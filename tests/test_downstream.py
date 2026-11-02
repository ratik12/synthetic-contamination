import pytest

from src.downstream import arm_names, arm_summary, build_arms


def corpus(n_h=100, n_twins=60):
    c = []
    for i in range(n_h):
        c.append({"doc_id": f"a{i}|human", "article_id": f"a{i}", "split": "S4", "generator": "human", "label": 0,
                  "text": f"human {i}", "n_words": 200})
        if i < n_twins:
            c.append({"doc_id": f"a{i}|G{i % 2}", "article_id": f"a{i}", "split": "S4", "generator": f"G{i % 2}",
                      "label": 1, "text": f"synthetic {i}", "n_words": 200})
    c.append({"doc_id": "z|human", "article_id": "z", "split": "S1", "generator": "human", "label": 0, "text": "x", "n_words": 5})
    return c


FR = [0.0, 0.05, 0.15, 0.30, 0.50]


def test_arm_names():
    assert arm_names(FR) == (["D0", "D5", "D15", "D30", "D50"], ["H95", "H85", "H70", "H50"])


def test_arms_sizes_fractions_and_matched_controls():
    arms = build_arms(corpus(), 100, FR, 0)
    assert set(arms) == {"D0", "D5", "D15", "D30", "D50", "H95", "H85", "H70", "H50"}
    summ = arm_summary(arms)
    for f in FR:
        name = f"D{int(f * 100)}"
        assert summ[name]["n_docs"] == 100 and summ[name]["n_synthetic"] == round(f * 100)
    for d, h in (("D5", "H95"), ("D15", "H85"), ("D30", "H70"), ("D50", "H50")):
        assert arms[h]["human"] == arms[d]["human"] and arms[h]["synthetic"] == []      # H = exactly the human part of D
        assert summ[h]["n_docs"] == summ[d]["n_human"]


def test_arms_nested_synthetic_and_no_twin_with_its_own_source():
    arms = build_arms(corpus(), 100, FR, 0)
    ids = lambda a: {d["article_id"] for d in a["synthetic"]}
    assert ids(arms["D5"]) <= ids(arms["D15"]) <= ids(arms["D30"]) <= ids(arms["D50"])
    for a in arms.values():
        assert not ({d["article_id"] for d in a["human"]} & ids(a))
        assert all(d["split"] == "S4" for d in a["human"] + a["synthetic"])


def test_arms_deterministic_and_seed_dependent():
    a, b, c = build_arms(corpus(), 100, FR, 0), build_arms(corpus(), 100, FR, 0), build_arms(corpus(), 100, FR, 1)
    assert [d["doc_id"] for d in a["D30"]["synthetic"]] == [d["doc_id"] for d in b["D30"]["synthetic"]]
    assert [d["doc_id"] for d in a["D30"]["synthetic"]] != [d["doc_id"] for d in c["D30"]["synthetic"]]


def test_fails_when_not_enough_twins_or_humans_or_empty():
    with pytest.raises(ValueError, match="twins"):
        build_arms(corpus(n_twins=20), 100, FR, 0)
    with pytest.raises(ValueError, match="human documents"):
        build_arms(corpus(n_h=40, n_twins=40), 100, FR, 0)
    with pytest.raises(ValueError):
        build_arms([], 100, FR, 0)
    with pytest.raises(ValueError):
        build_arms([d for d in corpus() if d["split"] != "S4"], 100, FR, 0)
