from collections import Counter

from src.generation import MockGenerator, assign_twins, build_prompt, pending_pairs

GENS = [{"name": "A"}, {"name": "B"}]
GCFG = {"target_words": 50, "min_words": 20}


def human(split_sizes):
    return [{"article_id": f"{s}_{i}", "split": s, "title": f"T{i}", "text": "x"} for s, n in split_sizes.items() for i in range(n)]


def test_assign_twins_structure():
    h = human({"S1": 20, "S2": 10, "S3": 8, "S4": 30, "S5": 6})
    pairs = assign_twins(h, GENS, twins_s4=12, seed=0)
    by = Counter((p["split"], p["generator"]) for p in pairs)
    assert by[("S1", "A")] == by[("S1", "B")] == 10
    assert by[("S2", "A")] == by[("S2", "B")] == 5
    assert by[("S3", "A")] == by[("S3", "B")] == 8
    assert by[("S4", "A")] + by[("S4", "B")] == 12
    assert not any(p["split"] == "S5" for p in pairs)
    # S1/S2/S4: one twin per article; S3: one per generator per article
    per_art = Counter(p["article_id"] for p in pairs)
    assert all(per_art[a] == 1 for a in per_art if not a.startswith("S3"))
    assert all(per_art[a] == 2 for a in per_art if a.startswith("S3"))


def test_assign_twins_deterministic():
    h = human({"S1": 20, "S2": 10, "S3": 8, "S4": 30, "S5": 6})
    assert assign_twins(h, GENS, 12, 0) == assign_twins(h, GENS, 12, 0)
    assert assign_twins(h, GENS, 12, 0) != assign_twins(h, GENS, 12, 1)


def test_prompt_contains_title_but_no_article_text():
    p = build_prompt("Mount Everest", 200)
    assert "Mount Everest" in p and "200" in p


def test_pending_pairs_retry_logic():
    good = " ".join(["word"] * 40) + "."
    pairs = [{"article_id": "a", "generator": "A", "split": "S1"}, {"article_id": "b", "generator": "A", "split": "S1"},
             {"article_id": "c", "generator": "A", "split": "S1"}]
    recs = [{"article_id": "a", "generator": "A", "attempt": 0, "raw": good},
            {"article_id": "b", "generator": "A", "attempt": 0, "raw": "I'm sorry, I cannot."}]
    todo = pending_pairs(pairs, recs, 3, GCFG)
    assert {(p["article_id"], a) for p, a in todo} == {("b", 1), ("c", 0)}
    recs += [{"article_id": "b", "generator": "A", "attempt": i, "raw": "I'm sorry, I cannot."} for i in (1, 2)]
    assert {p["article_id"] for p, _ in pending_pairs(pairs, recs, 3, GCFG)} == {"c"}   # b exhausted its attempts


def test_mock_generator_deterministic_and_long_enough():
    g = MockGenerator(["alpha beta gamma delta epsilon"] * 10, fail_rate=0.0)
    t1, t2 = g.generate("a1", "A", 0, 210), g.generate("a1", "A", 0, 210)
    assert t1 == t2 and len(t1.split()) >= 210
    assert g.generate("a1", "A", 1, 210) != t1
