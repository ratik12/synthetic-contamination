"""Stage 08: build D-arms and size-matched H-arms for downstream training. Does NOT train anything."""
import _boot  # noqa
import hashlib
from src.downstream import arm_summary, build_arms
from src.io_utils import data_dir, ensure, read_jsonl, stage_main, write_json, write_jsonl


def run(cfg, opts):
    out = data_dir(cfg)
    corpus = read_jsonl(out / "corpus.jsonl")
    dc = cfg["downstream"]
    arms = build_arms(corpus, dc["total_docs"], dc["d_fractions"], cfg["seed"])
    ddir = ensure(out / "downstream")
    for name, a in arms.items():
        docs = [{"text": d["text"], "article_id": d["article_id"], "generator": d["generator"], "label": d["label"]}
                for d in a["human"] + a["synthetic"]]
        write_jsonl(ddir / f"{name}.jsonl", docs)
    s5 = [{"text": d["text"], "article_id": d["article_id"]} for d in corpus if d["split"] == "S5" and d["label"] == 0]
    if not s5:
        raise ValueError("no S5 human documents for held-out evaluation")
    write_jsonl(ddir / "eval_human_S5.jsonl", s5)
    man = {"arms": arm_summary(arms), "n_eval_S5": len(s5), "config_sha256": cfg["_config_sha256"],
           "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ddir.glob("*.jsonl"))},
           "mock": bool(cfg.get("mock"))}
    write_json(ddir / "manifest.json", man)
    for k, v in man["arms"].items():
        print(f"{k}: {v['n_docs']} docs, {v['n_synthetic']} synthetic, {v['total_words']} words")
    return f"{len(arms)} arms written; downstream training not run"


if __name__ == "__main__":
    stage_main("08_prepare_downstream", run)
