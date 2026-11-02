"""Print a few human and synthetic passages. READ THEM before trusting any result."""
import _boot  # noqa
import argparse
import random

from src.io_utils import data_dir, load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--raw", action="store_true", help="show raw generations (use after a smoke run, before stage 03)")
a = ap.parse_args()
cfg = load_config(a.config)
if a.raw:
    for g in cfg["generation"]["generators"]:
        for r in read_jsonl(data_dir(cfg) / f"generated_{g['name']}.jsonl")[: a.n]:
            print(f"\n=== RAW {g['name']} | {r['article_id']} | attempt {r['attempt']} ===\n{r['raw']}")
    raise SystemExit(0)
corpus = read_jsonl(data_dir(cfg) / "corpus.jsonl")
rng = random.Random(0)
for g in sorted({d["generator"] for d in corpus}):
    docs = [d for d in corpus if d["generator"] == g]
    for d in rng.sample(docs, min(a.n, len(docs))):
        print(f"\n=== {g} | {d['doc_id']} | {d['n_words']} words ===\n{d['text']}")
