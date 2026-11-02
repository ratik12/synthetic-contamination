import copy

import pytest

from src.checks import check_arms, check_corpus
from src.corpus import build_corpus
from tests.helpers import prose_articles

GC = {"target_words": 200, "min_words": 150}
QC = {"copy_ngram": 8, "max_len_ratio": 1.5}


def make_human(n=6):
    arts = prose_articles(n * 2, seed=5)
    out = []
    for i in range(n):
        words = arts[i]["text"].split()[:200]
        out.append({"article_id": f"a{i}", "title": f"T{i}", "text": " ".join(words), "n_words": 200,
                    "split": "S1" if i < n // 2 else "S3"})
    return out


def good_gen(art, tag):
    # distinct text per tag; no 8-gram overlap with anything else
    return " ".join(f"{tag}{art}w{i}" for i in range(210)) + "."


def records(human, gens=("A",), backend="hf"):
    return [{"article_id": h["article_id"], "split": h["split"], "generator": g, "attempt": 0,
             "raw": good_gen(h["article_id"], g), "backend": backend} for h in human for g in gens]


def test_build_corpus_happy_path_passes_checks():
    h = make_human()
    corpus, rep = build_corpus(h, records(h, ("A", "B")), GC, QC, False)
    assert rep["n_human"] == 6 and rep["n_synthetic"] == 12
    assert check_corpus(corpus)["n_synthetic"] == 12


def test_build_corpus_drops_refusals_and_copies():
    h = make_human()
    recs = records(h)
    recs[0]["raw"] = "I'm sorry, I cannot do that."
    recs[1]["raw"] = h[1]["text"]                               # verbatim copy of its own source
    corpus, rep = build_corpus(h, recs, GC, QC, False)
    assert rep["dropped"]["refusal"] == 1 and rep["dropped"]["copied_from_source"] == 1
    assert rep["n_synthetic"] == 4


def test_build_corpus_uses_retry_attempt_when_first_failed():
    h = make_human(2)
    recs = records(h)
    recs[0]["raw"] = "I'm sorry, I cannot."
    recs.append({**recs[0], "attempt": 1, "raw": good_gen("retry", "A")})
    corpus, rep = build_corpus(h, recs, GC, QC, False)
    assert rep["n_synthetic"] == 2


def test_mock_records_refused_unless_mock_flag():
    h = make_human(2)
    with pytest.raises(RuntimeError):
        build_corpus(h, records(h, backend="mock"), GC, QC, False)
    with pytest.raises(RuntimeError):
        build_corpus(h, records(h, backend="hf"), GC, QC, True)
    build_corpus(h, records(h, backend="mock"), GC, QC, True)


def test_length_confound_flag_raised_only_when_lengths_differ():
    h = make_human(2)
    recs = records(h)
    for r in recs:
        r["raw"] = " ".join(f"w{r['article_id']}x{i}" for i in range(400)) + "."
    _, rep = build_corpus(h, recs, {"target_words": 400, "min_words": 150}, QC, False)        # 400 vs 200 words
    assert rep["per_generator"]["A"]["length_confound_flag"] is True
    _, rep = build_corpus(h, recs, GC, QC, False)                                             # cut to 200 words
    assert rep["per_generator"]["A"]["length_confound_flag"] is False


# ---- the checks must FAIL on bad input, including empty input -------------------------------------------------------
def test_check_corpus_fails_on_empty():
    with pytest.raises(AssertionError):
        check_corpus([])


def test_check_corpus_fails_when_no_human_docs():
    h = make_human(2)
    corpus, _ = build_corpus(h, records(h), GC, QC, False)
    with pytest.raises(AssertionError):
        check_corpus([d for d in corpus if d["label"] == 1])


def _corpus():
    h = make_human()
    return build_corpus(h, records(h), GC, QC, False)[0]


def test_check_corpus_detects_article_in_two_splits():
    c = _corpus()
    c[0] = {**c[0], "split": "S2"}
    other = next(d for d in c if d["article_id"] == c[0]["article_id"] and d is not c[0])
    with pytest.raises(AssertionError, match="appears in splits"):
        check_corpus(c)


def test_check_corpus_detects_identical_text_across_splits():
    c = _corpus()
    humans = [d for d in c if d["label"] == 0]
    a, b = next(d for d in humans if d["split"] == "S1"), next(d for d in humans if d["split"] == "S3")
    i = c.index(b)
    c[i] = {**b, "text": a["text"]}
    with pytest.raises(AssertionError):
        check_corpus(c)


def test_check_corpus_detects_twin_copying_its_source():
    c = _corpus()
    syn = next(d for d in c if d["label"] == 1)
    src = next(d for d in c if d["label"] == 0 and d["article_id"] == syn["article_id"])
    c[c.index(syn)] = {**syn, "text": src["text"]}
    with pytest.raises(AssertionError, match="copies its source"):
        check_corpus(c)


def test_check_corpus_detects_duplicate_ids():
    c = _corpus()
    with pytest.raises(AssertionError, match="duplicate doc_id"):
        check_corpus(c + [copy.deepcopy(c[0])])


def test_check_arms_fails_on_empty_and_on_leak():
    with pytest.raises(AssertionError):
        check_arms({})
    doc = lambda a, lab: {"article_id": a, "doc_id": f"{a}|{lab}", "split": "S4", "label": lab}
    ok = {"D5": {"human": [doc("a1", 0)], "synthetic": [doc("a2", 1)]}}
    assert check_arms(ok)
    leak = {"D5": {"human": [doc("a1", 0)], "synthetic": [doc("a1", 1)]}}
    with pytest.raises(AssertionError, match="both human text and synthetic twin"):
        check_arms(leak)
    wrong_split = {"D5": {"human": [{**doc("a1", 0), "split": "S1"}], "synthetic": []}}
    with pytest.raises(AssertionError, match="from split"):
        check_arms(wrong_split)
