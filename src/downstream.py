"""Stage 08: build the downstream training sets (D arms) and size-matched human-only controls (H arms).

NO training happens here. D_f = N documents of which round(f*N) are synthetic twins; the remaining human documents come
from articles whose twin is NOT in the arm. H_(100-f) is exactly the human part of D_f (same documents, no synthetic),
so D_f vs H_(100-f) isolates the effect of the synthetic documents, and D_f vs D0 the effect of replacement."""
import numpy as np

from .checks import check_arms


def arm_names(fractions):
    d = [f"D{int(round(f * 100))}" for f in fractions]
    h = [f"H{100 - int(round(f * 100))}" for f in fractions if f > 0]
    return d, h


def build_arms(corpus, total_docs, fractions, seed):
    s4h = sorted([d for d in corpus if d["split"] == "S4" and d["label"] == 0], key=lambda d: d["article_id"])
    s4s = {d["article_id"]: d for d in corpus if d["split"] == "S4" and d["label"] == 1}
    if not s4h:
        raise ValueError("no S4 human documents; run stages 01-03")
    if len(s4h) < total_docs:
        raise ValueError(f"S4 has {len(s4h)} human documents but total_docs={total_docs}")
    rng = np.random.default_rng(seed)
    twin_order = [str(a) for a in rng.permutation(sorted(s4s))]          # nested: D15's twins are the first 15% of D30's
    human_order = [s4h[i] for i in rng.permutation(len(s4h))]
    arms = {}
    for f in fractions:
        n_syn = int(round(f * total_docs))
        if n_syn > len(twin_order):
            raise ValueError(f"fraction {f} needs {n_syn} synthetic docs but only {len(twin_order)} S4 twins exist; "
                             "raise data.twins_S4 or lower total_docs/fractions")
        syn_arts = list(twin_order[:n_syn])
        syn = [s4s[a] for a in syn_arts]
        banned = set(syn_arts)
        hum = [d for d in human_order if d["article_id"] not in banned][: total_docs - n_syn]
        if len(hum) < total_docs - n_syn:
            raise ValueError(f"arm D{int(round(f * 100))}: only {len(hum)} eligible human docs for {total_docs - n_syn} slots")
        arms[f"D{int(round(f * 100))}"] = {"human": hum, "synthetic": syn}
        if f > 0:
            arms[f"H{100 - int(round(f * 100))}"] = {"human": hum, "synthetic": []}
    check_arms(arms, "S4")
    return arms


def arm_summary(arms):
    out = {}
    for name, a in arms.items():
        docs = a["human"] + a["synthetic"]
        gens = {}
        for d in a["synthetic"]:
            gens[d["generator"]] = gens.get(d["generator"], 0) + 1
        out[name] = {"n_docs": len(docs), "n_human": len(a["human"]), "n_synthetic": len(a["synthetic"]),
                     "synthetic_fraction": len(a["synthetic"]) / max(1, len(docs)),
                     "total_words": int(sum(d["n_words"] for d in docs)), "synthetic_by_generator": gens}
    return out
