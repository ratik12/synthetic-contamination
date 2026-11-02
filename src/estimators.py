"""Prevalence estimators: given detector scores on an audit set and labelled calibration scores, estimate the
synthetic fraction alpha. Pure numpy; unit-tested on simulated Gaussian scores with known alpha."""
import numpy as np
from sklearn.linear_model import LogisticRegression

MIN_TPR_GAP = 0.05          # below this, ACC divides by ~0 and is reported as NaN


def choose_threshold(cal_h, cal_s):
    """Threshold maximising TPR - FPR (Youden J) on calibration scores."""
    cand = np.unique(np.concatenate([cal_h, cal_s]))
    cand = np.concatenate([[cand[0] - 1.0], (cand[:-1] + cand[1:]) / 2.0, [cand[-1] + 1.0]])
    tpr = (cal_s[None, :] > cand[:, None]).mean(1)
    fpr = (cal_h[None, :] > cand[:, None]).mean(1)
    return float(cand[np.argmax(tpr - fpr)])


def platt_fit(cal_h, cal_s):
    x = np.concatenate([cal_h, cal_s])
    y = np.concatenate([np.zeros(len(cal_h)), np.ones(len(cal_s))])
    mu, sd = x.mean(), x.std() + 1e-12
    m = LogisticRegression(C=100.0, max_iter=1000).fit(((x - mu) / sd)[:, None], y)
    return float(m.coef_[0, 0] / sd), float(m.intercept_[0] - m.coef_[0, 0] * mu / sd)


def platt_apply(x, ab):
    z = ab[0] * np.asarray(x) + ab[1]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def rates_at(cal_h, cal_s, tau):
    return float((cal_s > tau).mean()), float((cal_h > tau).mean())


def acc_estimate(cc, tpr, fpr):
    if tpr - fpr < MIN_TPR_GAP:
        return float("nan")
    return float(np.clip((cc - fpr) / (tpr - fpr), 0.0, 1.0))


def _bandwidth(x):
    return max(1e-6, float(np.std(x)) * len(x) ** (-1.0 / 5.0))


def kernel_matrix(points, centers, bw):
    z = (points[:, None] - centers[None, :]) / bw
    return np.exp(-0.5 * z ** 2) / (bw * np.sqrt(2 * np.pi))


def alpha_grid(step):
    return np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1)


def mle_from_densities(fh, fs, grid, weights=None):
    """argmax_alpha sum_i w_i log((1-alpha) fh_i + alpha fs_i). fh, fs: shape (M,) or (M,B); weights (M,) or (M,B)."""
    fh = np.maximum(fh, 1e-12)
    fs = np.maximum(fs, 1e-12)
    if fh.ndim == 1:
        ll = np.log((1 - grid)[None, :] * fh[:, None] + grid[None, :] * fs[:, None])
        if weights is not None:
            ll = ll * weights[:, None]
        return float(grid[np.argmax(ll.sum(0))])
    M, B = fh.shape
    out = np.empty(B)
    for b in range(B):                                  # loop over replicates keeps memory at M x G
        ll = np.log((1 - grid)[None, :] * fh[:, b:b + 1] + grid[None, :] * fs[:, b:b + 1])
        if weights is not None:
            ll = ll * weights[:, b:b + 1]
        out[b] = grid[np.argmax(ll.sum(0))]
    return out


def estimate_all(audit, cal_h, cal_s, n_boot, rng, step, level=0.95):
    """Point estimates and percentile bootstrap CIs for CC, PCC, ACC, MLE.

    Bootstrap: CC and PCC resample the audit set only (threshold and Platt fit fixed from the full calibration set).
    ACC and MLE resample the audit set AND the calibration sets (threshold fixed; TPR/FPR and KDE densities recomputed)."""
    audit, cal_h, cal_s = (np.asarray(a, dtype=float) for a in (audit, cal_h, cal_s))
    if min(len(audit), len(cal_h), len(cal_s)) < 5:
        raise ValueError("audit and calibration sets must each have at least 5 scores")
    M = len(audit)
    tau = choose_threshold(cal_h, cal_s)
    ab = platt_fit(cal_h, cal_s)
    tpr, fpr = rates_at(cal_h, cal_s, tau)
    grid = alpha_grid(step)
    bw_h, bw_s = _bandwidth(cal_h), _bandwidth(cal_s)
    Kh, Ks = kernel_matrix(audit, cal_h, bw_h), kernel_matrix(audit, cal_s, bw_s)   # (M, n)
    fh, fs = Kh.mean(1), Ks.mean(1)
    cc_ind = (audit > tau).astype(float)
    pcc_ind = platt_apply(audit, ab)
    point = {"cc": float(cc_ind.mean()), "pcc": float(pcc_ind.mean()),
             "acc": acc_estimate(cc_ind.mean(), tpr, fpr), "mle": mle_from_densities(fh, fs, grid)}

    cnt_a = rng.multinomial(M, np.full(M, 1.0 / M), size=n_boot).T.astype(float)                       # (M, B)
    cnt_h = rng.multinomial(len(cal_h), np.full(len(cal_h), 1.0 / len(cal_h)), size=n_boot).T.astype(float)
    cnt_s = rng.multinomial(len(cal_s), np.full(len(cal_s), 1.0 / len(cal_s)), size=n_boot).T.astype(float)
    boot = {"cc": (cc_ind @ cnt_a) / M, "pcc": (pcc_ind @ cnt_a) / M}
    # ACC: per replicate TPR/FPR from resampled calibration scores
    s_ind = (cal_s > tau).astype(float)
    h_ind = (cal_h > tau).astype(float)
    tpr_b = (s_ind @ cnt_s) / len(cal_s)
    fpr_b = (h_ind @ cnt_h) / len(cal_h)
    cc_b = (cc_ind @ cnt_a) / M
    boot["acc"] = np.array([acc_estimate(c, t, f) for c, t, f in zip(cc_b, tpr_b, fpr_b)])
    fh_b = (Kh @ cnt_h) / len(cal_h)
    fs_b = (Ks @ cnt_s) / len(cal_s)
    boot["mle"] = mle_from_densities(fh_b, fs_b, grid, weights=cnt_a)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    ci = {k: (float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))) if np.isfinite(v).any()
          else (float("nan"), float("nan")) for k, v in boot.items()}
    return {"point": point, "ci": ci, "tau": tau, "tpr": tpr, "fpr": fpr}
