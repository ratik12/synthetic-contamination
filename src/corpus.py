"""Stage 03: turn raw generations into a clean labelled corpus and report quality problems."""
import numpy as np

from .text_utils import clean_generation, near_duplicate_pairs, n_words, shares_ngram, text_hash


def build_corpus(human, gen_records, gcfg, qcfg, mock_flag):
    """human: list of human passage rows. gen_records: raw generation records (all generators).

    Returns (corpus rows, report dict). Each (article, generator) uses its first attempt that cleans successfully."""
    backends = {r.get("backend") for r in gen_records}
    if "mock" in backends and not mock_flag:
        raise RuntimeError("mock-generated records found but config has mock: false; refusing to build a corpus")
    if mock_flag and backends - {"mock"}:
        raise RuntimeError("config has mock: true but real model outputs are present")

    src = {r["article_id"]: r for r in human}
    corpus = [{"doc_id": f"{r['article_id']}|human", "article_id": r["article_id"], "split": r["split"],
               "generator": "human", "label": 0, "text": r["text"], "n_words": r["n_words"], "backend": "wikipedia"}
              for r in human]
    rep = {"dropped": {"refusal": 0, "empty": 0, "too_short": 0, "copied_from_source": 0, "near_dup_synthetic": 0,
                       "unknown_article": 0},
           "per_generator": {}}
    chosen = {}
    for r in sorted(gen_records, key=lambda x: x["attempt"]):
        key = (r["article_id"], r["generator"])
        if key in chosen:
            continue
        if r["article_id"] not in src:
            rep["dropped"]["unknown_article"] += 1
            continue
        clean, why = clean_generation(r["raw"], gcfg["target_words"], gcfg["min_words"])
        if clean is None:
            rep["dropped"][why] = rep["dropped"].get(why, 0) + 1
            continue
        chosen[key] = (r, clean)

    syn = []
    for (a, g), (r, clean) in chosen.items():
        if shares_ngram(clean, src[a]["text"], qcfg["copy_ngram"]):
            rep["dropped"]["copied_from_source"] += 1
            continue
        syn.append({"doc_id": f"{a}|{g}", "article_id": a, "split": src[a]["split"], "generator": g, "label": 1,
                    "text": clean, "n_words": n_words(clean), "backend": r.get("backend", "hf")})
    syn.sort(key=lambda d: d["doc_id"])
    drop = set()
    for i, j in near_duplicate_pairs([d["text"] for d in syn], qcfg["copy_ngram"], 0.3):
        drop.add(max(i, j))
    rep["dropped"]["near_dup_synthetic"] = len(drop)
    syn = [d for k, d in enumerate(syn) if k not in drop]
    corpus += syn

    h_len = np.mean([d["n_words"] for d in corpus if d["label"] == 0])
    for g in sorted({d["generator"] for d in syn}):
        L = [d["n_words"] for d in syn if d["generator"] == g]
        rep["per_generator"][g] = {"n": len(L), "mean_words": float(np.mean(L)), "std_words": float(np.std(L)),
                                   "len_ratio_vs_human": float(np.mean(L) / h_len)}
        rep["per_generator"][g]["length_confound_flag"] = bool(
            not (1 / qcfg["max_len_ratio"] <= np.mean(L) / h_len <= qcfg["max_len_ratio"]))
    # passages cut mid-sentence are a possible shortcut feature; report the rate per group
    term = lambda t: t.rstrip()[-1:] in (".", "!", "?", '"', "'", ")")
    groups = {"human": [d for d in corpus if d["label"] == 0]}
    for g in sorted({d["generator"] for d in syn}):
        groups[g] = [d for d in syn if d["generator"] == g]
    rep["ends_with_terminal_punct"] = {k: float(np.mean([term(d["text"]) for d in v])) for k, v in groups.items() if v}
    rep["human_mean_words"] = float(h_len)
    rep["n_human"] = sum(1 for d in corpus if d["label"] == 0)
    rep["n_synthetic"] = len(syn)
    return corpus, rep
