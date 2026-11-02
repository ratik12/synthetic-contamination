"""Detection metrics with bootstrap intervals."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def tpr_at_fpr(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= fpr_target)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def detection_metrics(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        raise ValueError("need both classes to compute detection metrics")
    return {"auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01)}


def bootstrap_ci(y, s, fn, n_boot=500, seed=0, alpha=0.05):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))
