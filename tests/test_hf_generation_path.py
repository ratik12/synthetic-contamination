"""Exercises the real HFGenerator code (chat template, left padding, sampling, decode, resume, retries) with tiny random models.
The generated text is meaningless; only mechanics are checked."""
from src.generation import assign_twins, run_generation
from src.io_utils import read_jsonl
from tests.helpers import prose_articles


def test_hf_backend_mechanics_and_resume(tiny, tmp_path):
    arts = prose_articles(8, seed=7)
    human = [{"article_id": f"a{i}", "title": a["title"], "text": "x", "split": "S1"} for i, a in enumerate(arts)]
    cfg = {"seed": 0, "generation": {"backend": "hf", "target_words": 40, "min_words": 15, "max_new_tokens": 90,
                                     "temperature": 0.8, "top_p": 0.95, "batch_size": 3, "max_attempts": 2,
                                     "dtype": "float32",
                                     "generators": [{"name": "A", "hf_id": tiny["genA"]}, {"name": "B", "hf_id": tiny["genB"]}]}}
    pairs = assign_twins(human, cfg["generation"]["generators"], 0, 0)
    assert len(pairs) == 8
    summ = run_generation(cfg, human, pairs, tmp_path, "cpu", log=lambda *a: None)
    for g, s in summ.items():
        assert s["ok"] + s["failed_after_retries"] == s["requested"] == 4
    for g in ("A", "B"):
        recs = read_jsonl(tmp_path / f"generated_{g}.jsonl")
        assert recs and all(r["backend"] == "hf" and r["model"] for r in recs)
        assert max(r["attempt"] for r in recs) <= 1
    n_before = {g: len(read_jsonl(tmp_path / f"generated_{g}.jsonl")) for g in ("A", "B")}
    run_generation(cfg, human, pairs, tmp_path, "cpu", log=lambda *a: None)          # resume: nothing new to do
    assert n_before == {g: len(read_jsonl(tmp_path / f"generated_{g}.jsonl")) for g in ("A", "B")}


def test_hf_generation_is_seed_reproducible(tiny):
    from src.generation import HFGenerator
    g = HFGenerator(tiny["genA"], "cpu", {"dtype": "float32", "temperature": 0.8, "top_p": 0.95, "max_new_tokens": 30})
    p = ['Write the opening section of a Wikipedia article titled "River".'] * 2
    assert g.generate_batch(p, 5) == g.generate_batch(p, 5)
    assert g.generate_batch(p, 5) != g.generate_batch(p, 6)
