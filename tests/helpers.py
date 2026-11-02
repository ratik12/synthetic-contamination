"""Test fixtures: real English prose from the Python stdlib docstrings, and tiny random local models.
Nothing here needs network. Tiny models produce meaningless numbers; tests check code paths, not science."""
import importlib
import inspect
import random
import sys
from pathlib import Path

SKIP = {"antigravity", "this", "idlelib", "tkinter", "turtle", "turtledemo", "pydoc_data", "lib2to3", "test", "ensurepip",
        "venv", "msvcrt", "winreg", "_winapi", "winsound", "nt", "readline", "curses", "ctypes", "multiprocessing"}


def _first_paragraph(obj):
    """All prose paragraphs of a docstring (>=12 words, no doctest/rst markup), joined."""
    doc = inspect.getdoc(obj)
    if not doc:
        return None
    keep = []
    for para in doc.split("\n\n"):
        para = " ".join(para.split())
        if len(para.split()) >= 12 and ">>>" not in para and "----" not in para and "::" not in para and "`" not in para:
            keep.append(para)
    return " ".join(keep) if keep else None


def _docstrings(limit=100000):
    """First paragraphs of docstrings from the stdlib and a few installed scientific packages (real English prose)."""
    out = []
    names = sorted(n for n in sys.stdlib_module_names if not n.startswith("_") and n.split(".")[0] not in SKIP)
    names += ["numpy", "scipy.stats", "scipy.optimize", "sklearn.linear_model", "sklearn.ensemble", "sklearn.metrics",
              "sklearn.feature_extraction.text", "pandas", "yaml", "tokenizers", "torch.nn", "torch.optim"]
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        objs = [o for _, o in inspect.getmembers(mod) if inspect.isfunction(o) or inspect.isclass(o)]
        for obj in objs:
            para = _first_paragraph(obj)
            if para:
                out.append(para)
            if inspect.isclass(obj):
                for _, m in inspect.getmembers(obj, inspect.isfunction):
                    para = _first_paragraph(m)
                    if para:
                        out.append(para)
        if len(out) >= limit:
            break
    return list(dict.fromkeys(out))


_CACHE = {}


def prose_articles(n, seed=0, words=(230, 320)):
    """n articles, each a distinct run of docstring sentences; text has newlines like raw Wikipedia dumps."""
    if "docs" not in _CACHE:
        _CACHE["docs"] = _docstrings()
    docs = list(_CACHE["docs"])
    random.Random(seed).shuffle(docs)
    arts, i = [], 0
    rng = random.Random(seed + 1)
    for k in range(n):
        target = rng.randint(*words)
        parts, total = [], 0
        while total < target:
            if i >= len(docs):
                raise RuntimeError("not enough docstring prose for requested article count")
            parts.append(docs[i])
            total += len(docs[i].split())
            i += 1
        arts.append({"id": str(1000 + k), "title": f"Topic {k} on {parts[0].split()[0].lower()}", "text": "\n".join(parts)})
    return arts


def build_tiny_models(root):
    """Writes tiny GPT-2 scorer, two tiny GPT-2 'generators' (with a chat template) and a tiny DistilBERT. Returns paths."""
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import (DistilBertConfig, DistilBertForSequenceClassification, GPT2Config, GPT2LMHeadModel,
                              PreTrainedTokenizerFast)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    corpus = [a["text"] for a in prose_articles(60, seed=99)]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=600, min_frequency=2, special_tokens=["<|endoftext|>", "<|pad|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                                  pad_token="<|pad|>", model_input_names=["input_ids", "attention_mask"])
    tok.chat_template = ("{% for m in messages %}User: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}Assistant:{% endif %}")
    V = len(tok)
    paths = {}
    for name, seed in (("scorer", 1), ("genA", 2), ("genB", 3)):
        torch.manual_seed(seed)
        cfg = GPT2Config(vocab_size=V, n_positions=512, n_embd=32, n_layer=2, n_head=2, bos_token_id=0, eos_token_id=0)
        m = GPT2LMHeadModel(cfg)
        d = root / name
        m.save_pretrained(d)
        tok.save_pretrained(d)
        paths[name] = str(d)
    torch.manual_seed(4)
    dcfg = DistilBertConfig(vocab_size=V, dim=32, n_layers=2, n_heads=2, hidden_dim=64, max_position_embeddings=512,
                            pad_token_id=1)
    d = root / "distil"
    DistilBertForSequenceClassification(dcfg).save_pretrained(d)
    tok.save_pretrained(d)
    paths["distil"] = str(d)
    return paths
