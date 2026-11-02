"""Stage 03: clean generations, quality-filter, run leakage checks, write corpus.jsonl."""
import _boot  # noqa
from src.checks import check_corpus
from src.corpus import build_corpus
from src.io_utils import data_dir, read_jsonl, stage_main, write_json, write_jsonl


def run(cfg, opts):
    out = data_dir(cfg)
    human = read_jsonl(out / "human.jsonl")
    recs = []
    for g in cfg["generation"]["generators"]:
        p = out / f"generated_{g['name']}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run stage 02 for generator {g['name']} first")
        recs += read_jsonl(p)
    corpus, rep = build_corpus(human, recs, cfg["generation"], cfg["quality"], bool(cfg.get("mock")))
    rep["checks"] = check_corpus(corpus, cfg["quality"]["copy_ngram"])
    counts = {}
    for d in corpus:
        counts[f"{d['split']}/{d['generator']}"] = counts.get(f"{d['split']}/{d['generator']}", 0) + 1
    rep["counts"] = dict(sorted(counts.items()))
    write_jsonl(out / "corpus.jsonl", corpus)
    write_json(out / "quality_report.json", rep)
    print("dropped:", rep["dropped"])
    for g, v in rep["per_generator"].items():
        flag = "  <-- LENGTH CONFOUND" if v["length_confound_flag"] else ""
        print(f"{g}: n={v['n']} mean_words={v['mean_words']:.0f} ratio={v['len_ratio_vs_human']:.2f}{flag}")
    print("share of passages ending in terminal punctuation:", {k: round(v, 3) for k, v in rep["ends_with_terminal_punct"].items()})
    return f"{rep['n_human']} human, {rep['n_synthetic']} synthetic; checks passed"


if __name__ == "__main__":
    stage_main("03_build_corpus", run)
