import numpy as np
import pytest

from src.estimators import (acc_estimate, alpha_grid, choose_threshold, estimate_all, mle_from_densities, platt_apply,
                            platt_fit)
from src.metrics import bootstrap_ci, detection_metrics, tpr_at_fpr


def sim(rng, n_h, n_s, sep=1.5):
    return rng.normal(0, 1, n_h), rng.normal(sep, 1, n_s)


def audit_scores(rng, m, alpha, sep=1.5):
    k = int(round(alpha * m))
    return np.concatenate([rng.normal(0, 1, m - k), rng.normal(sep, 1, k)])


def test_detection_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    assert detection_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert detection_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 0.0
    with pytest.raises(ValueError):
        detection_metrics(np.array([0, 0, 0]), np.array([1., 2., 3.]))


def test_tpr_at_fpr_perfect_and_random():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)]
    assert tpr_at_fpr(y, np.r_[rng.normal(0, 1, 1000), rng.normal(10, 1, 1000)]) == 1.0
    assert tpr_at_fpr(y, rng.normal(0, 1, 2000)) < 0.1


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(200), np.ones(200)]
    s = np.r_[rng.normal(0, 1, 200), rng.normal(1, 1, 200)]
    lo, hi = bootstrap_ci(y, s, lambda a, b: detection_metrics(a, b)["auroc"], 200, 0)
    assert lo < detection_metrics(y, s)["auroc"] < hi


def test_threshold_and_platt_on_separated_scores():
    rng = np.random.default_rng(2)
    h, s = sim(rng, 500, 500, 4.0)
    assert 1.0 < choose_threshold(h, s) < 3.0
    ab = platt_fit(h, s)
    assert platt_apply(np.array([-2.0]), ab)[0] < 0.05 and platt_apply(np.array([6.0]), ab)[0] > 0.95


def test_acc_nan_when_detector_uninformative():
    assert np.isnan(acc_estimate(0.3, 0.51, 0.50))
    assert acc_estimate(0.3, 0.9, 0.1) == pytest.approx(0.25)
    assert acc_estimate(2.0, 0.9, 0.1) == 1.0                                   # clipped


def test_mle_recovers_alpha_from_densities():
    from scipy.stats import norm
    rng = np.random.default_rng(3)
    x = audit_scores(rng, 4000, 0.3)
    a = mle_from_densities(norm.pdf(x, 0, 1), norm.pdf(x, 1.5, 1), alpha_grid(0.005))
    assert abs(a - 0.3) < 0.05


@pytest.mark.parametrize("alpha", [0.0, 0.15, 0.5])
def test_acc_and_mle_much_less_biased_than_cc(alpha):
    rng = np.random.default_rng(10)
    errs = {"cc": [], "pcc": [], "acc": [], "mle": []}
    for r in range(30):
        h, s = sim(rng, 300, 300)
        res = estimate_all(audit_scores(rng, 200, alpha), h, s, 60, rng, 0.01)
        for k in errs:
            errs[k].append(res["point"][k] - alpha)
    bias = {k: np.nanmean(v) for k, v in errs.items()}
    if alpha >= 0.15:
        assert abs(bias["acc"]) < 0.05 and abs(bias["mle"]) < 0.05
        assert abs(bias["cc"]) > abs(bias["acc"])
    else:
        assert abs(bias["acc"]) < 0.06 and abs(bias["mle"]) < 0.06


def test_bootstrap_ci_covers_truth_reasonably_often():
    rng = np.random.default_rng(11)
    cover = {"acc": 0, "mle": 0}
    n = 40
    for r in range(n):
        h, s = sim(rng, 300, 300)
        res = estimate_all(audit_scores(rng, 200, 0.3), h, s, 150, rng, 0.01)
        for k in cover:
            lo, hi = res["ci"][k]
            cover[k] += lo <= 0.3 <= hi
    assert cover["acc"] / n >= 0.75 and cover["mle"] / n >= 0.75


def test_estimate_all_rejects_tiny_inputs():
    with pytest.raises(ValueError):
        estimate_all([0.1, 0.2], [0.0] * 10, [1.0] * 10, 10, np.random.default_rng(0), 0.01)
