import pytest

from src.splits import assign_splits
from src.text_utils import (clean_generation, cut_to_sentence_end, extract_passage, is_bad_article, near_duplicate_pairs,
                            n_words, shares_ngram, split_sentences)
from tests.helpers import prose_articles


def test_extract_passage_length_and_sentence_end():
    art = "\n".join(f"Sentence number {i} talks about the river and the old stone bridge near town." for i in range(60))
    p = extract_passage(art, 200, 150)
    assert p is not None and 150 <= n_words(p) <= 200
    assert p.rstrip().endswith(".")


def test_extract_passage_too_short_returns_none():
    assert extract_passage("Short text. Only a few words here.", 200, 150) is None


def test_extract_passage_skips_headings():
    body = " ".join(["This is a sentence about a topic."] * 40)
    raw = f"Intro\n{body}\nHistory\n{body}"
    p = extract_passage(raw, 200, 150)
    assert p is not None and "History" not in p.split(".")[0]


def test_bad_articles():
    assert is_bad_article("Mercury (disambiguation)", "x")
    assert is_bad_article("List of rivers", "x")
    assert is_bad_article("Mercury", "Mercury may refer to: a planet")
    assert not is_bad_article("Mercury", "Mercury is a planet in the Solar System.")


def test_clean_generation_strips_chat_and_markdown():
    body = " ".join(["The river flows through the valley and into the sea."] * 25)
    raw = f"Sure! Here is the article:\n\n**River**\n\n# Heading\n{body}"
    out, why = clean_generation(raw, 200, 150)
    assert why is None and "**" not in out and "Sure" not in out and "Heading" not in out
    assert n_words(out) <= 200


def test_clean_generation_rejects_refusal_empty_short():
    assert clean_generation("I'm sorry, I cannot help with that.", 200, 150) == (None, "refusal")
    assert clean_generation("   ", 200, 150)[0] is None
    assert clean_generation("Too short.", 200, 150) == (None, "too_short")
    assert clean_generation(None, 200, 150) == (None, "empty")


def test_cut_to_sentence_end_drops_partial_sentence():
    sent = "The old river runs south past the mill and the town."          # 11 words, ends with a period
    text = " ".join([sent] * 12)                                         # 132 words
    cut = text.split()
    cut = " ".join(cut[:100])                                            # mid-sentence cut would end in a fragment
    assert not cut.endswith(".")
    out = cut_to_sentence_end(text, 100, 50)
    assert out.endswith(".") and n_words(out) <= 100 and n_words(out) >= 50


def test_shares_ngram_and_near_duplicates():
    a = "one two three four five six seven eight nine ten eleven"
    b = "zero one two three four five six seven eight nine"
    c = "completely different words appear in this sentence here today again later"
    assert shares_ngram(a, b, 8) and not shares_ngram(a, c, 8)
    assert near_duplicate_pairs([a, b, c], 8, 0.3) == [(0, 1)]


def test_split_sentences_basic():
    assert split_sentences("A cat sat. A dog ran! Is it? Yes.") == ["A cat sat.", "A dog ran!", "Is it?", "Yes."]


SIZES = {"S1": 6, "S2": 4, "S3": 5, "S4": 7, "S5": 3}


def test_splits_disjoint_exact_sizes_deterministic():
    ids = [f"a{i}" for i in range(40)]
    s = assign_splits(ids, SIZES, 1)
    assert {k: len(v) for k, v in s.items()} == SIZES
    flat = [x for v in s.values() for x in v]
    assert len(flat) == len(set(flat))
    assert s == assign_splits(list(reversed(ids)), SIZES, 1)
    assert s != assign_splits(ids, SIZES, 2)


def test_splits_raise_when_too_few():
    with pytest.raises(ValueError):
        assign_splits([f"a{i}" for i in range(10)], SIZES, 1)
