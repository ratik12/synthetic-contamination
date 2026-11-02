"""Stage 01: sample Wikipedia articles, extract fixed-length human passages, deduplicate, assign splits."""
import numpy as np

from .io_utils import read_jsonl
from .splits import assign_splits
from .text_utils import (extract_passage, is_bad_article, near_duplicate_pairs, n_words, text_hash)


def load_articles(dcfg, seed):
    """Return (list of {id,title,text}, info dict). Never silently changes the dump: deviations are recorded in info."""
    info = {"source": dcfg["source"], "fallback_used": False}
    if dcfg["source"] == "local_jsonl":
        rows = read_jsonl(dcfg["local_path"])
        info.update(name="local_jsonl", path=dcfg["local_path"], n_scanned=len(rows))
        return rows, info

    from datasets import load_dataset
    name, conf = dcfg["hf_name"], dcfg["hf_config"]
    n_pool = dcfg["pool_size"]
    try:
        if dcfg.get("hf_streaming"):
            ds = load_dataset(name, conf, split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(seed=seed, buffer_size=10000).take(n_pool)
            rows = [{"id": str(r.get("id", i)), "title": r["title"], "text": r["text"]} for i, r in enumerate(ds)]
        else:
            ds = load_dataset(name, conf, split="train", trust_remote_code=True)
            idx = np.sort(np.random.default_rng(seed).choice(len(ds), size=min(n_pool, len(ds)), replace=False))
            rows = []
            for i in idx:
                r = ds[int(i)]
                rows.append({"id": str(r.get("id", i)), "title": r["title"], "text": r["text"]})
        info.update(name=name, config=conf, n_scanned=len(rows))
    except Exception as e:
        if not dcfg.get("allow_fallback_dump"):
            raise RuntimeError(
                f"Could not load {name}/{conf}: {type(e).__name__}: {e}\n"
                "Set data.allow_fallback_dump: true to use wikimedia/wikipedia 20231101.simple (post-ChatGPT dump; "
                "this is a deviation that must be reported), or set data.source: local_jsonl.") from e
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
        idx = np.sort(np.random.default_rng(seed).choice(len(ds), size=min(n_pool, len(ds)), replace=False))
        rows = [{"id": str(ds[int(i)]["id"]), "title": ds[int(i)]["title"], "text": ds[int(i)]["text"]} for i in idx]
        info.update(name="wikimedia/wikipedia", config="20231101.simple", n_scanned=len(rows), fallback_used=True,
                    warning="DEVIATION: 2023-11 dump may itself contain LLM-written text; the 1 Mar 2022 pre-ChatGPT guarantee does not hold")
        print("WARNING:", info["warning"])
    return rows, info


def build_human_passages(rows, dcfg):
    """Filter, extract passages, drop exact and near duplicates. Returns (passages, stats)."""
    stats = {"scanned": len(rows), "bad_article": 0, "too_short": 0, "exact_dup": 0, "near_dup": 0}
    cand, seen = [], set()
    for r in rows:
        if is_bad_article(r["title"], r["text"]):
            stats["bad_article"] += 1
            continue
        p = extract_passage(r["text"], dcfg["passage_words"], dcfg["min_words"])
        if p is None:
            stats["too_short"] += 1
            continue
        h = text_hash(p)
        if h in seen:
            stats["exact_dup"] += 1
            continue
        seen.add(h)
        cand.append({"article_id": f"a{r['id']}", "title": r["title"], "text": p, "n_words": n_words(p), "hash": h})
    drop = set()
    for i, j in near_duplicate_pairs([c["text"] for c in cand], dcfg.get("dedup_ngram", 8), dcfg.get("dedup_threshold", 0.5)):
        drop.add(max(i, j))
    stats["near_dup"] = len(drop)
    cand = [c for k, c in enumerate(cand) if k not in drop]
    stats["eligible"] = len(cand)
    return cand, stats


def split_passages(cand, dcfg, seed):
    splits = assign_splits([c["article_id"] for c in cand], dcfg["sizes"], seed)
    where = {a: s for s, ids in splits.items() for a in ids}
    out = []
    for c in cand:
        if c["article_id"] in where:
            out.append({**c, "split": where[c["article_id"]]})
    out.sort(key=lambda r: (r["split"], r["article_id"]))
    return out, splits
