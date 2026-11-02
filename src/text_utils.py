"""Pure text helpers: passage extraction, sentence splitting, cleaning of generated text, overlap tests."""
import hashlib
import re

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")
_REFUSAL = re.compile(r"\b(i cannot|i can't|i can not|i'm sorry|i am sorry|as an ai|i'm unable|i am unable)\b", re.I)
_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|okay|ok|here(?:'s| is| are)|absolutely)\b[^\n]*?(:|\n)\s*", re.I)


def normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return text.split()


def n_words(text):
    return len(text.split())


def split_sentences(text):
    text = normalize_ws(text)
    if not text:
        return []
    parts = _SENT_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def first_sentence(text):
    s = split_sentences(text)
    return s[0] if s else ""


def text_hash(text):
    return hashlib.sha1(normalize_ws(text).lower().encode("utf-8")).hexdigest()


def looks_like_heading(line):
    line = line.strip()
    if not line:
        return False
    return len(line.split()) <= 6 and not re.search(r"[.!?]$", line)


def is_bad_article(title, text):
    t = title.lower()
    if "disambiguation" in t or t.startswith("list of") or t.startswith("lists of"):
        return True
    head = text[:400].lower()
    if " may refer to" in head or "may also refer to" in head:
        return True
    return False


def cut_to_sentence_end(text, max_words, min_words):
    """Keep at most max_words words, backing up to the last sentence end if that keeps >= min_words."""
    ws = text.split()
    if len(ws) <= max_words:
        cut = " ".join(ws)
    else:
        cut = " ".join(ws[:max_words])
    sents = split_sentences(cut)
    # drop a trailing partial sentence (no terminal punctuation) when we truncated
    if len(ws) > max_words and sents and not re.search(r"[.!?][\"')\]]*$", sents[-1]):
        trimmed = " ".join(sents[:-1])
        if n_words(trimmed) >= min_words:
            cut = trimmed
    return cut


def extract_passage(raw_text, passage_words=200, min_words=150, stop_at_heading=False):
    """Opening text of an article, cut to ~passage_words at a sentence end.

    Heading-like lines (short, no terminal punctuation) are skipped, or end the passage when
    stop_at_heading=True. Simple-English articles often have a short lead, so skipping is the default.
    Returns None when the article is too short."""
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    lead = []
    for l in lines:
        if looks_like_heading(l):
            if lead and stop_at_heading:
                break
            continue
        lead.append(l)
        if n_words(" ".join(lead)) >= passage_words + 40:
            break
    text = normalize_ws(" ".join(lead))
    if n_words(text) < min_words:
        return None
    passage = cut_to_sentence_end(text, passage_words, min_words)
    return passage if n_words(passage) >= min_words else None


def clean_generation(raw, target_words=200, min_words=150):
    """Strip chat preambles, markdown and refusals from a generated passage; cut to target length.

    Returns (clean_text or None, reason or None)."""
    if raw is None:
        return None, "empty"
    t = raw.replace("\r", "")
    if _REFUSAL.search(t[:300]):
        return None, "refusal"
    t = _PREAMBLE.sub("", t, count=1)
    # markdown: headings, bold/italic markers, bullets, numbered lists, code fences
    t = re.sub(r"^\s{0,3}#{1,6}\s*.*$", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__|\*|`+)", "", t)
    t = re.sub(r"^\s*([-•]|\d+[.)])\s+", "", t, flags=re.M)
    paragraphs = [normalize_ws(p) for p in t.split("\n") if normalize_ws(p)]
    # drop a leading "Title:" style line
    if paragraphs and looks_like_heading(paragraphs[0]) and len(paragraphs) > 1:
        paragraphs = paragraphs[1:]
    t = normalize_ws(" ".join(paragraphs))
    if not t:
        return None, "empty"
    t = cut_to_sentence_end(t, target_words, min_words)
    if n_words(t) < min_words:
        return None, "too_short"
    return t, None


def word_ngrams(text, n):
    ws = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", text)]
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def shares_ngram(a, b, n=8):
    """True if texts a and b share any run of n consecutive words (memorisation / copying test)."""
    return len(word_ngrams(a, n) & word_ngrams(b, n)) > 0


def near_duplicate_pairs(texts, n=8, threshold=0.5):
    """Indices (i, j) whose n-gram sets overlap by >= threshold of the smaller set. Inverted-index, O(total shingles)."""
    sets = [word_ngrams(t, n) for t in texts]
    index = {}
    for i, s in enumerate(sets):
        for g in s:
            index.setdefault(g, []).append(i)
    counts = {}
    for g, ids in index.items():
        if len(ids) < 2:
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (ids[a], ids[b])
                counts[key] = counts.get(key, 0) + 1
    out = []
    for (i, j), c in counts.items():
        denom = max(1, min(len(sets[i]), len(sets[j])))
        if c / denom >= threshold:
            out.append((i, j))
    return out
