"""Stages 06-07: detector evaluation (RQ1) on S3, and dataset-level prevalence audits (RQ2/RQ3) on S3 audit sets."""
import hashlib

import numpy as np
import pandas as pd

from .estimators import estimate_all
from .metrics import bootstrap_ci, detection_metrics


def _cond_gen(cond):
    return cond.split(":", 1)[1] if cond.startswith("unseen:") else None


def detection_table(scores, n_boot=500, seed=0):
    """AUROC / AUPRC / TPR@1%FPR on S3 with 95% bootstrap CI for AUROC.

    'seen'      : one row per generator (human vs that generator) plus an 'all' row.
    'unseen:g'  : human vs held-out generator g only (the detector never saw g in training)."""
    s3 = scores[scores["split"] == "S3"]
    if s3.empty:
        raise ValueError("no S3 scores")
    rows = []
    for (det, cond), g in s3.groupby(["detector", "condition"]):
        held = _cond_gen(cond)
        gens = sorted(set(g.loc[g.label == 1, "generator"]))
        targets = [held] if held else gens + ["all"]
        for t in targets:
            sub = g if t == "all" else g[(g.generator == t) | (g.label == 0)]
            y, s = sub["label"].values, sub["score"].values
            m = detection_metrics(y, s)
            lo, hi = bootstrap_ci(y, s, lambda a, b: detection_metrics(a, b)["auroc"], n_boot, seed)
            rows.append({"detector": det, "condition": cond, "eval_generator": t, "n": len(sub), **m,
                         "auroc_lo": lo, "auroc_hi": hi})
    return pd.DataFrame(rows)


def _stable_seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def make_audit_set(s3_by_article, gens, alpha, size, rng):
    """Pick `size` S3 articles; round(alpha*size) of them contribute a synthetic twin (generator drawn from `gens`),
    the rest contribute their human text. An article contributes exactly one document. Returns list of doc_ids, true alpha."""
    arts = sorted(s3_by_article)
    if size > len(arts):
        raise ValueError(f"audit set size {size} exceeds the {len(arts)} S3 articles")
    pick = rng.choice(len(arts), size=size, replace=False)
    k = int(round(alpha * size))
    ids = []
    for pos, ai in enumerate(pick):
        a = arts[ai]
        if pos < k:
            g = gens[int(rng.integers(len(gens)))]
            ids.append(s3_by_article[a][g])
        else:
            ids.append(s3_by_article[a]["human"])
    return ids, k / size


def run_audits(scores, acfg, gens, seed, log=print):
    """For every detector x condition x alpha, draw audit sets and estimate alpha with CC/PCC/ACC/MLE.

    Calibration scores come from S2. In an 'unseen:g' condition the calibration set excludes g's twins and the audit
    sets contain synthetic text from g only; in 'seen' they come from all generators."""
    s3 = scores[scores["split"] == "S3"]
    rows = []
    for (det, cond), g in scores.groupby(["detector", "condition"]):
        held = _cond_gen(cond)
        cal = g[g.split == "S2"]
        if held:
            cal = cal[(cal.label == 0) | (cal.generator != held)]
        cal_h = cal.loc[cal.label == 0, "score"].values
        cal_s = cal.loc[cal.label == 1, "score"].values
        s3g = g[g.split == "S3"]
        by_art = {}
        sc = {}
        for r in s3g.itertuples():
            aid = r.doc_id.split("|")[0]
            by_art.setdefault(aid, {})[("human" if r.label == 0 else r.generator)] = r.doc_id
            sc[r.doc_id] = r.score
        gens_used = [held] if held else gens
        by_art = {a: d for a, d in by_art.items() if "human" in d and all(x in d for x in gens_used)}
        for alpha in acfg["alphas"]:
            for draw in range(acfg["draws_per_alpha"]):
                rng = np.random.default_rng(_stable_seed(seed, cond, alpha, draw))      # same draws for every detector
                ids, true_a = make_audit_set(by_art, gens_used, alpha, acfg["set_size"], rng)
                brng = np.random.default_rng(_stable_seed(seed, det, cond, alpha, draw, "boot"))
                res = estimate_all([sc[i] for i in ids], cal_h, cal_s, acfg["n_boot"], brng, acfg["alpha_grid_step"])
                for est in ("cc", "pcc", "acc", "mle"):
                    lo, hi = res["ci"][est]
                    p = res["point"][est]
                    rows.append({"detector": det, "condition": cond, "alpha": true_a, "draw": draw, "estimator": est,
                                 "estimate": p, "ci_lo": lo, "ci_hi": hi, "tau": res["tau"], "tpr": res["tpr"],
                                 "fpr": res["fpr"], "n_cal_h": len(cal_h), "n_cal_s": len(cal_s)})
        log(f"audited {det} / {cond}")
    return pd.DataFrame(rows)


def summarize_estimates(est):
    d = est.copy()
    d["err"] = d["estimate"] - d["alpha"]
    d["covered"] = (d["ci_lo"] <= d["alpha"]) & (d["alpha"] <= d["ci_hi"])
    d["width"] = d["ci_hi"] - d["ci_lo"]
    g = d.groupby(["detector", "condition", "estimator", "alpha"])
    out = g.agg(n=("err", "size"), n_nan=("estimate", lambda x: int(x.isna().sum())), bias=("err", "mean"),
                mae=("err", lambda x: float(np.nanmean(np.abs(x)))),
                rmse=("err", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
                coverage=("covered", "mean"), mean_ci_width=("width", "mean")).reset_index()
    return out
