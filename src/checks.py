"""Leakage and integrity checks. Every check raises AssertionError on violation AND on empty input
(a check that passes on nothing is worthless)."""
from .text_utils import near_duplicate_pairs, shares_ngram, text_hash


def _nonempty(rows, what):
    if not rows:
        raise AssertionError(f"{what}: no documents to check (empty input must fail, not pass)")


def check_corpus(corpus, copy_ngram=8):
    _nonempty(corpus, "corpus")
    human = [d for d in corpus if d["label"] == 0]
    syn = [d for d in corpus if d["label"] == 1]
    _nonempty(human, "human documents")
    # 1. each article lives in exactly one split
    art_split = {}
    for d in corpus:
        s = art_split.setdefault(d["article_id"], d["split"])
        assert s == d["split"], f"article {d['article_id']} appears in splits {s} and {d['split']}"
    # 2. doc ids unique
    ids = [d["doc_id"] for d in corpus]
    assert len(ids) == len(set(ids)), "duplicate doc_id in corpus"
    # 3. identical text must not appear in two different splits
    seen = {}
    for d in corpus:
        h = text_hash(d["text"])
        if h in seen and seen[h] != d["split"]:
            raise AssertionError(f"identical text appears in splits {seen[h]} and {d['split']} (doc {d['doc_id']})")
        seen.setdefault(h, d["split"])
    # 4. human passages from different splits must not be near duplicates of each other
    pairs = near_duplicate_pairs([d["text"] for d in human], copy_ngram, 0.3)
    for i, j in pairs:
        assert human[i]["split"] == human[j]["split"], (
            f"near-duplicate human passages across splits: {human[i]['doc_id']} / {human[j]['doc_id']}")
    # 5. a twin never shares an n-gram with its own source article
    src = {d["article_id"]: d["text"] for d in human}
    for d in syn:
        assert d["article_id"] in src, f"synthetic doc {d['doc_id']} has no human source"
        assert not shares_ngram(d["text"], src[d["article_id"]], copy_ngram), f"{d['doc_id']} copies its source"
    return {"n_human": len(human), "n_synthetic": len(syn), "near_dup_pairs_checked": len(pairs)}


def check_arms(arms, split_required="S4"):
    """arms: {name: {"human": [doc dicts], "synthetic": [doc dicts]}}. Checks the downstream contamination design.

    - no article contributes BOTH its human text and its synthetic twin to the same arm
    - every document comes from the required split
    - every arm is non-empty
    """
    _nonempty(list(arms.values()), "arms")
    for name, a in arms.items():
        h_art = {d["article_id"] for d in a["human"]}
        s_art = {d["article_id"] for d in a["synthetic"]}
        both = h_art & s_art
        assert not both, f"arm {name}: {len(both)} articles contribute both human text and synthetic twin"
        for d in a["human"] + a["synthetic"]:
            assert d["split"] == split_required, f"arm {name}: doc {d['doc_id']} from split {d['split']}"
        assert len(a["human"]) + len(a["synthetic"]) > 0, f"arm {name} is empty"
    return True
