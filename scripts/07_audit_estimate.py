"""Stage 07: RQ2/RQ3 dataset-level prevalence estimation on S3 audit sets."""
import _boot  # noqa
import pandas as pd
from src.evaluate import run_audits, summarize_estimates
from src.io_utils import results_dir, stage_main


def run(cfg, opts):
    res = results_dir(cfg)
    scores = pd.read_csv(res / "scores.csv")
    gens = [g["name"] for g in cfg["generation"]["generators"]]
    acfg = dict(cfg["audit"])
    if "draws" in opts:
        acfg["draws_per_alpha"] = int(opts["draws"])
    est = run_audits(scores, acfg, gens, cfg["seed"])
    est.to_csv(res / "estimates.csv", index=False)
    summ = summarize_estimates(est)
    summ.to_csv(res / "estimator_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    return f"{len(est)} estimates"


if __name__ == "__main__":
    stage_main("07_audit_estimate", run)
