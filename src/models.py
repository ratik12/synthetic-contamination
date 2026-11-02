"""Model loading and LM token statistics (used by generation and feature extraction)."""
import os

import numpy as np


def hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def torch_dtype(name):
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_causal_lm(model_id, device, dtype="float32"):
    """model_id may be a hub id or a local directory. Works across transformers 4.x/5.x dtype kwarg rename."""
    from transformers import AutoModelForCausalLM
    kw = {"token": hf_token()} if not os.path.isdir(model_id) else {}
    dt = torch_dtype(dtype) if device == "cuda" else torch_dtype("float32")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)
    model.to(device)
    model.eval()
    return model


def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    kw = {"token": hf_token()} if not os.path.isdir(model_id) else {}
    return AutoTokenizer.from_pretrained(model_id, **kw)


def free_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


LM_FEATURE_NAMES = ["ll_mean", "ll_std", "ent_mean", "logrank_mean", "fdg", "n_tokens"]


def lm_token_stats(model, tok, texts, device, max_len=256, batch_size=8):
    """Per-text statistics of a causal LM over a text. A BOS/EOS token is prepended so token 1 is scored.

    ll_mean   mean token log-probability (higher => more LM-like)
    ll_std    std of token log-probability (burstiness)
    ent_mean  mean predictive entropy
    logrank_mean mean log rank of observed token
    fdg       analytic Fast-DetectGPT criterion with the same model as sampler and scorer:
              (sum log p(x_t) - sum E[log p]) / sqrt(sum Var[log p])
    """
    import torch
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if start is None:
        raise ValueError("tokenizer has neither BOS nor EOS token")
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        seqs = []
        for t in chunk:
            ids = tok(t, add_special_tokens=False, truncation=True, max_length=max_len - 1)["input_ids"]
            seqs.append([start] + ids)
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), 0, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            tgt = ids[:, 1:]
            m = mask[:, 1:].bool()
            logp = torch.log_softmax(logits, dim=-1)
            tlp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            p = logp.exp()
            mean_ref = (p * logp).sum(-1)
            ent = -mean_ref
            var_ref = (p * logp ** 2).sum(-1) - mean_ref ** 2
            rank = (logp > tlp.unsqueeze(-1)).sum(-1) + 1
        for j in range(len(seqs)):
            mj = m[j]
            lp = tlp[j][mj].cpu().numpy().astype(np.float64)
            if len(lp) < 2:
                out.append({k: float("nan") for k in LM_FEATURE_NAMES})
                continue
            v = var_ref[j][mj].clamp(min=1e-8).sum().item()
            out.append({
                "ll_mean": float(lp.mean()),
                "ll_std": float(lp.std()),
                "ent_mean": float(ent[j][mj].mean().item()),
                "logrank_mean": float(np.log(rank[j][mj].cpu().numpy()).mean()),
                "fdg": float((lp.sum() - mean_ref[j][mj].sum().item()) / np.sqrt(v)),
                "n_tokens": int(mj.sum().item()),
            })
    return out


def binoculars_scores(observer, performer, tok, texts, device, max_len=256, batch_size=4):
    """Binoculars (Hans et al., ICML 2024), following the authors' reference code.

        score = mean_t[-log p_performer(x_t)]  /  mean_t[-sum_v p_observer(v) log p_performer(v)]

    observer = base model, performer = its instruction-tuned sibling (same tokenizer required).
    LOWER score => more likely machine-generated. Returns a list of floats (NaN for texts under 2 tokens)."""
    import torch
    if observer.config.vocab_size != performer.config.vocab_size:
        raise ValueError("observer and performer must share a vocabulary")
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        seqs = []
        for t in chunk:
            ids = tok(t, add_special_tokens=False, truncation=True, max_length=max_len - 1)["input_ids"]
            seqs.append([start] + ids)
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.no_grad():
            lo = observer(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            lp = performer(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            tgt = ids[:, 1:]
            m = mask[:, 1:].bool()
            logp_perf = torch.log_softmax(lp, -1)
            nll = -logp_perf.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            xent = -(torch.softmax(lo, -1) * logp_perf).sum(-1)
        for j in range(len(seqs)):
            mj = m[j]
            if int(mj.sum()) < 2:
                out.append(float("nan"))
            else:
                out.append(float(nll[j][mj].mean().item() / xent[j][mj].mean().item()))
    return out
