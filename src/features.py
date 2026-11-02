"""Stage 04: stylometric + LM-based features. POS features from the research plan are NOT implemented (disclosed)."""
import math
import re

import numpy as np

from .text_utils import split_sentences

FUNCTION_WORDS = set("""the of and to in a is that for it as was with be by on not he this are or his from at which but have an had
they you were her all she there would their we him been has when who will more no if out so said what up its about into than them
can only other new some could time these two may then do first any my now such like our over man me even most made after also did many
before must through back years where much your way well down should because each just those people how too little state good very make
world still own see men work long get here between both life being under never day same another know while last might us great old year
off come since against go came right used take three""".split())

STYLE_FEATURES = ["ttr", "rep3", "sent_mean", "sent_std", "func_rate", "comma_rate", "paren_rate", "digit_rate",
                  "quote_rate", "semi_rate", "word_len", "flesch"]


def _syllables(w):
    w = w.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def stylometric(text):
    toks = re.findall(r"[A-Za-z0-9']+", text)
    low = [t.lower() for t in toks]
    n = max(1, len(low))
    first = low[:150]
    ttr = len(set(first)) / max(1, len(first))
    tri = [tuple(low[i:i + 3]) for i in range(len(low) - 2)]
    rep3 = 1 - len(set(tri)) / max(1, len(tri))
    sents = split_sentences(text)
    sl = np.array([len(s.split()) for s in sents]) if sents else np.array([n])
    alpha = [t for t in toks if t.isalpha()]
    syl = sum(_syllables(t) for t in alpha) / max(1, len(alpha))
    flesch = 206.835 - 1.015 * sl.mean() - 84.6 * syl
    nw = max(1, len(text.split()))
    return {
        "ttr": ttr, "rep3": rep3, "sent_mean": float(sl.mean()), "sent_std": float(sl.std()),
        "func_rate": sum(1 for t in low if t in FUNCTION_WORDS) / n,
        "comma_rate": text.count(",") / nw, "paren_rate": text.count("(") / nw,
        "digit_rate": sum(c.isdigit() for c in text) / max(1, len(text)),
        "quote_rate": (text.count('"') + text.count("'")) / nw, "semi_rate": (text.count(";") + text.count(":")) / nw,
        "word_len": float(np.mean([len(t) for t in alpha])) if alpha else 0.0, "flesch": flesch,
    }


def extract_all(corpus, scorer_id, device, max_len=256, batch_size=8, log=print, bino=None):
    """Return list of feature dicts aligned with corpus order. Imports torch lazily.

    bino: optional dict {observer_id, performer_id, max_len, batch_size}; adds the Binoculars score ('bino', lower = machine)."""
    from .models import free_cuda, lm_token_stats, load_causal_lm, load_tokenizer
    tok = load_tokenizer(scorer_id)
    model = load_causal_lm(scorer_id, device, "float32")
    texts = [d["text"] for d in corpus]
    lm = []
    for i in range(0, len(texts), 400):
        lm.extend(lm_token_stats(model, tok, texts[i:i + 400], device, max_len, batch_size))
        log(f"scored {min(i + 400, len(texts))}/{len(texts)}")
    del model
    free_cuda()
    bino_scores = [float("nan")] * len(texts)
    if bino:
        from .models import binoculars_scores
        btok = load_tokenizer(bino["observer_id"])
        obs = load_causal_lm(bino["observer_id"], device, bino.get("dtype", "float32"))
        perf = load_causal_lm(bino["performer_id"], device, bino.get("dtype", "float32"))
        bino_scores = []
        for i in range(0, len(texts), 400):
            bino_scores.extend(binoculars_scores(obs, perf, btok, texts[i:i + 400], device, bino.get("max_len", 256),
                                                 bino.get("batch_size", 4)))
            log(f"binoculars {min(i + 400, len(texts))}/{len(texts)}")
        del obs, perf
        free_cuda()
    rows = []
    for d, l, b in zip(corpus, lm, bino_scores):
        row = {"doc_id": d["doc_id"], "bino": b}
        row.update(l)
        row.update(stylometric(d["text"]))
        rows.append(row)
    return rows


FEATURE_COLUMNS = ["ll_mean", "ll_std", "ent_mean", "logrank_mean", "fdg", "bino"] + STYLE_FEATURES
