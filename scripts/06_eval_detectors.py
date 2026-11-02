"""Stage 06: RQ1 detection metrics on S3 with bootstrap CIs."""
import _boot  # noqa
import pandas as pd
from src.evaluate import detection_table
from src.io_utils import results_dir, stage_main


def run(cfg, opts):
    res = results_dir(cfg)
    tab = detection_table(pd.read_csv(res / "scores.csv"), seed=cfg["seed"])
    tab.to_csv(res / "detection_metrics.csv", index=False)
    print(tab.round(3).to_string(index=False))
    return f"{len(tab)} rows"


if __name__ == "__main__":
    stage_main("06_eval_detectors", run)
