"""Stage 05: train detectors on S1 (seen / leave-one-generator-out), score S2 and S3."""
import _boot  # noqa
from src.detectors import run_detectors
from src.io_utils import data_dir, pick_device, read_jsonl, results_dir, ensure, stage_main


def run(cfg, opts):
    out = data_dir(cfg)
    corpus = read_jsonl(out / "corpus.jsonl")
    feats = {r["doc_id"]: r for r in read_jsonl(out / "features.jsonl")}
    missing = [d["doc_id"] for d in corpus if d["doc_id"] not in feats]
    if missing:
        raise ValueError(f"{len(missing)} documents lack features; rerun stage 04")
    df = run_detectors(corpus, feats, cfg, pick_device("auto"))
    res = ensure(results_dir(cfg))
    df.to_csv(res / "scores.csv", index=False)
    return f"{len(df)} score rows"


if __name__ == "__main__":
    stage_main("05_train_detectors", run)
