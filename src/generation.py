"""Stage 02: synthetic twins. Prompts contain only the article TITLE (no human text), so a twin cannot copy its source."""
import hashlib
import random

import numpy as np

from .io_utils import append_jsonl, read_jsonl, set_seed
from .text_utils import clean_generation

PROMPT = ('Write the opening section of a Wikipedia article titled "{title}". '
          'Write about {n} words of plain encyclopedic prose as one continuous piece, '
          'with no headings, no lists and no introduction or closing remark.')


def build_prompt(title, n_words):
    return PROMPT.format(title=title, n=n_words)


def assign_twins(human, generators, twins_s4, seed):
    """Decide which (article, generator) pairs must be generated. Pure and deterministic.

    S1, S2: every article gets ONE twin, generators round-robin (balanced).
    S3:     every article gets a twin from EVERY generator (test set for per-generator metrics and audits).
    S4:     the first `twins_s4` articles (seeded order) get one twin each, round-robin (downstream pool).
    S5:     no twins (held-out human evaluation text).
    Returns list of dicts {article_id, split, generator, order}.
    """
    rng = np.random.default_rng(seed)
    names = [g["name"] if isinstance(g, dict) else g for g in generators]
    by_split = {}
    for r in human:
        by_split.setdefault(r["split"], []).append(r["article_id"])
    out = []
    for s in ("S1", "S2", "S4"):
        ids = sorted(by_split.get(s, []))
        perm = rng.permutation(len(ids))
        ids = [ids[i] for i in perm]
        if s == "S4":
            ids = ids[:twins_s4]
        for k, a in enumerate(ids):
            out.append({"article_id": a, "split": s, "generator": names[k % len(names)], "order": k})
    for a in sorted(by_split.get("S3", [])):
        for n in names:
            out.append({"article_id": a, "split": "S3", "generator": n, "order": -1})
    return out


def _mock_seed(article_id, generator, attempt):
    return int(hashlib.sha1(f"{article_id}|{generator}|{attempt}".encode()).hexdigest()[:8], 16)


class MockGenerator:
    """Word-sampling stand-in used ONLY by offline plumbing tests. Output is not language; records carry backend='mock'."""

    def __init__(self, vocab_texts, fail_rate=0.0):
        words = " ".join(vocab_texts).split()
        vals, counts = np.unique([w.lower().strip('.,;:()"') for w in words], return_counts=True)
        self.vocab, self.p = vals, counts / counts.sum()
        self.fail_rate = fail_rate

    def generate(self, article_id, generator, attempt, n_words=210):
        rng = np.random.default_rng(_mock_seed(article_id, generator, attempt))
        if attempt == 0 and rng.random() < self.fail_rate:
            return "I'm sorry, I cannot write that."
        sents, total = [], 0
        while total < n_words:
            k = int(rng.integers(8, 20))
            w = list(rng.choice(self.vocab, size=k, p=self.p))
            w[0] = w[0].capitalize()
            sents.append(" ".join(w) + ".")
            total += k
        return " ".join(sents)


class HFGenerator:
    def __init__(self, hf_id, device, gcfg):
        from .models import load_causal_lm, load_tokenizer
        self.tok = load_tokenizer(hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load_causal_lm(hf_id, device, gcfg.get("dtype", "float16"))
        self.device = device
        self.g = gcfg

    def _format(self, prompt):
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                add_generation_prompt=True)
        return prompt

    def generate_batch(self, prompts, seed):
        import torch
        set_seed(seed)
        texts = [self._format(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=True, temperature=self.g["temperature"], top_p=self.g["top_p"],
                                      max_new_tokens=self.g["max_new_tokens"], pad_token_id=self.tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(new, skip_special_tokens=True)

    def generate(self, prompts, seed, batch_size):
        import torch
        res, i, bs = [], 0, batch_size
        while i < len(prompts):
            try:
                res.extend(self.generate_batch(prompts[i:i + bs], seed + i))
                i += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"CUDA OOM: batch size reduced to {bs}")
        return res


def pending_pairs(pairs, records, max_attempts, gcfg):
    """Pairs with no successfully-cleaned record and fewer than max_attempts attempts."""
    ok, tries = set(), {}
    for r in records:
        key = (r["article_id"], r["generator"])
        tries[key] = max(tries.get(key, 0), r["attempt"] + 1)
        clean, _ = clean_generation(r["raw"], gcfg["target_words"], gcfg["min_words"])
        if clean is not None:
            ok.add(key)
    return [(p, tries.get((p["article_id"], p["generator"]), 0)) for p in pairs
            if (p["article_id"], p["generator"]) not in ok
            and tries.get((p["article_id"], p["generator"]), 0) < max_attempts]


def run_generation(cfg, human, pairs, out_dir, device, log=print):
    """Resumable. For each generator: loop over attempts, appending raw outputs to generated_<name>.jsonl."""
    gcfg = cfg["generation"]
    title = {r["article_id"]: r["title"] for r in human}
    mock = gcfg["backend"] == "mock"
    summary = {}
    for g in gcfg["generators"]:
        name = g["name"]
        path = out_dir / f"generated_{name}.jsonl"
        mine = [p for p in pairs if p["generator"] == name]
        gen = None
        for attempt_round in range(gcfg["max_attempts"]):
            records = read_jsonl(path) if path.exists() else []
            todo = pending_pairs(mine, records, gcfg["max_attempts"], gcfg)
            todo = [(p, a) for p, a in todo if a == attempt_round]
            if not todo:
                continue
            log(f"[{name}] attempt {attempt_round}: {len(todo)} passages")
            if gen is None:
                if mock:
                    gen = MockGenerator([r["text"] for r in human], gcfg.get("mock_fail_rate", 0.0))
                else:
                    gen = HFGenerator(g["hf_id"], device, gcfg)
            bs = gcfg["batch_size"]
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                if mock:
                    raws = [gen.generate(p["article_id"], name, a, gcfg["target_words"] + 10) for p, a in chunk]
                else:
                    prompts = [build_prompt(title[p["article_id"]], gcfg["target_words"]) for p, _ in chunk]
                    raws = gen.generate(prompts, cfg["seed"] + 1000 * attempt_round + i, bs)
                append_jsonl(path, [{"article_id": p["article_id"], "split": p["split"], "generator": name,
                                     "attempt": a, "raw": raw, "backend": "mock" if mock else "hf",
                                     "model": g["hf_id"]} for (p, a), raw in zip(chunk, raws)])
                if (i // bs) % 10 == 0:
                    log(f"[{name}] {min(i + bs, len(todo))}/{len(todo)}")
        records = read_jsonl(path) if path.exists() else []
        ok = {(r["article_id"], r["generator"]) for r in records
              if clean_generation(r["raw"], gcfg["target_words"], gcfg["min_words"])[0] is not None}
        n_ok = sum(1 for p in mine if (p["article_id"], p["generator"]) in ok)
        summary[name] = {"requested": len(mine), "ok": n_ok, "failed_after_retries": len(mine) - n_ok}
        if gen is not None and not mock:
            del gen
            from .models import free_cuda
            free_cuda()
    return summary
