"""Stage 01: sample Wikipedia, extract human passages, dedup, assign splits S1-S5."""
import _boot  # noqa
from src.data_prep import build_human_passages, load_articles, split_passages
from src.io_utils import data_dir, stage_main, write_json, write_jsonl


def run(cfg, opts):
    rows, info = load_articles(cfg["data"], cfg["seed"])
    cand, stats = build_human_passages(rows, cfg["data"])
    print("filter stats:", stats)
    human, splits = split_passages(cand, cfg["data"], cfg["seed"])
    out = data_dir(cfg)
    write_jsonl(out / "human.jsonl", human)
    write_json(out / "data_manifest.json", {"source": info, "filter_stats": stats,
                                            "split_sizes": {s: len(v) for s, v in splits.items()},
                                            "config_sha256": cfg["_config_sha256"], "mock": bool(cfg.get("mock"))})
    print({s: len(v) for s, v in splits.items()})
    return f"{len(human)} human passages; fallback_dump={info.get('fallback_used')}"


if __name__ == "__main__":
    stage_main("01_prepare_data", run)
