"""Stage 09: write LaTeX tables, figures and factual sentences for the report from the pipeline outputs.

Usage: python scripts/09_make_report_tables.py --config configs/mid.yaml --extra out=report
Then copy <out>/gen and <out>/fig into the Overleaf project (next to main.tex)."""
import _boot  # noqa
from src.io_utils import data_dir, results_dir, stage_main
from src.report_tables import build_all


def run(cfg, opts):
    out = opts.get("out", "report")
    written = build_all(cfg, data_dir(cfg), results_dir(cfg), out, allow_mock=opts.get("allow_mock") == "1")
    for w in written:
        print("wrote", w)
    missing = {"data_corpus", "data_synth", "det_seen", "det_unseen", "est_all", "est_sel", "results_text"} - set(written)
    if missing:
        print("NOT written (inputs missing; the report will show PENDING boxes):", sorted(missing))
    return f"{len(written)} files in {out}/"


if __name__ == "__main__":
    stage_main("09_make_report_tables", run)
