"""Stage 04: LM log-likelihood / Fast-DetectGPT statistics and stylometric features for every document."""
import _boot  # noqa
from src.features import extract_all
from src.io_utils import data_dir, pick_device, read_jsonl, stage_main, write_jsonl


def run(cfg, opts):
    out = data_dir(cfg)
    corpus = read_jsonl(out / "corpus.jsonl")
    s = cfg["scorer"]
    rows = extract_all(corpus, s["hf_id"], pick_device("auto"), s["max_len"], s["batch_size"], bino=cfg.get("binoculars"))
    write_jsonl(out / "features.jsonl", rows)
    return f"{len(rows)} documents"


if __name__ == "__main__":
    stage_main("04_extract_features", run)
