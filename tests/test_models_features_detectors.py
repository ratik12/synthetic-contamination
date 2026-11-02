import numpy as np
import pytest
import torch

from src.detectors import ZERO_SHOT, fit_predict_distilbert, fit_predict_features, fit_predict_tfidf, make_conditions, run_detectors
from src.features import FEATURE_COLUMNS, stylometric
from src.models import LM_FEATURE_NAMES, binoculars_scores, load_causal_lm, load_tokenizer, lm_token_stats
from tests.helpers import prose_articles


@pytest.fixture(scope="module")
def lm(tiny):
    return load_causal_lm(tiny["scorer"], "cpu"), load_tokenizer(tiny["scorer"])


TEXTS = [a["text"][:700] for a in prose_articles(4, seed=3)] + ["short text here for padding test"]


def test_lm_stats_hand_computed(lm):
    model, tok = lm
    text = TEXTS[0]
    got = lm_token_stats(model, tok, [text], "cpu", 256, 1)[0]
    ids = [tok.bos_token_id] + tok(text, add_special_tokens=False, truncation=True, max_length=255)["input_ids"]
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([ids])).logits[0, :-1].float(), -1)
    tgt = torch.tensor(ids[1:])
    tlp = logp[torch.arange(len(tgt)), tgt]
    p = logp.exp()
    mu = (p * logp).sum(-1)
    var = (p * logp ** 2).sum(-1) - mu ** 2
    assert got["ll_mean"] == pytest.approx(tlp.mean().item(), abs=1e-5)
    assert got["ll_std"] == pytest.approx(tlp.std(unbiased=False).item(), abs=1e-5)
    assert got["ent_mean"] == pytest.approx((-mu).mean().item(), abs=1e-5)
    assert got["fdg"] == pytest.approx(((tlp.sum() - mu.sum()) / var.clamp(min=1e-8).sum().sqrt()).item(), abs=1e-4)
    assert got["n_tokens"] == len(tgt)


def test_lm_stats_padding_invariant(lm):
    model, tok = lm
    batch = lm_token_stats(model, tok, TEXTS, "cpu", 256, 5)
    single = [lm_token_stats(model, tok, [t], "cpu", 256, 1)[0] for t in TEXTS]
    for b, s in zip(batch, single):
        for k in LM_FEATURE_NAMES:
            assert b[k] == pytest.approx(s[k], rel=1e-3, abs=1e-3), k


def test_binoculars_hand_computed_and_padding_invariant(tiny):
    obs, perf = load_causal_lm(tiny["scorer"], "cpu"), load_causal_lm(tiny["genA"], "cpu")
    tok = load_tokenizer(tiny["scorer"])
    text = TEXTS[1]
    got = binoculars_scores(obs, perf, tok, [text], "cpu", 256, 1)[0]
    ids = [tok.bos_token_id] + tok(text, add_special_tokens=False, truncation=True, max_length=255)["input_ids"]
    with torch.no_grad():
        lo = obs(torch.tensor([ids])).logits[0, :-1].float()
        lp = torch.log_softmax(perf(torch.tensor([ids])).logits[0, :-1].float(), -1)
    tgt = torch.tensor(ids[1:])
    ppl = -lp[torch.arange(len(tgt)), tgt].mean()
    xppl = -(torch.softmax(lo, -1) * lp).sum(-1).mean()
    assert got == pytest.approx((ppl / xppl).item(), rel=1e-5)
    batch = binoculars_scores(obs, perf, tok, TEXTS, "cpu", 256, 5)
    single = [binoculars_scores(obs, perf, tok, [t], "cpu", 256, 1)[0] for t in TEXTS]
    assert batch == pytest.approx(single, rel=1e-3)


def test_binoculars_same_model_equals_nll_over_entropy(lm):
    """With observer == performer the denominator is the model's own mean predictive entropy."""
    model, tok = lm
    b = binoculars_scores(model, model, tok, [TEXTS[0]], "cpu", 256, 1)[0]
    s = lm_token_stats(model, tok, [TEXTS[0]], "cpu", 256, 1)[0]
    assert b == pytest.approx(-s["ll_mean"] / s["ent_mean"], rel=1e-4)


def test_binoculars_rejects_mismatched_vocab(tiny, lm):
    from transformers import GPT2Config, GPT2LMHeadModel
    other = GPT2LMHeadModel(GPT2Config(vocab_size=50, n_positions=64, n_embd=16, n_layer=1, n_head=1))
    with pytest.raises(ValueError, match="vocabulary"):
        binoculars_scores(lm[0], other, lm[1], ["x y z"], "cpu")


def test_binoculars_detector_orientation_is_negated():
    from src.detectors import ZERO_SHOT
    assert ZERO_SHOT["binoculars"] == ("bino", -1.0)           # lower raw score = machine => higher detector score


def test_stylometric_finite_and_sensible():
    f = stylometric("The cat sat on the mat. It was a sunny day, and the cat was happy (very happy); 3 birds sang.")
    assert set(f) >= {"ttr", "rep3", "flesch", "func_rate"} and all(np.isfinite(list(f.values())))
    rep = stylometric(" ".join(["same words again"] * 50))
    assert rep["rep3"] > 0.9 and rep["ttr"] < 0.1


# ---------- detectors on synthetic features: separable signal must be found, noise must not -------------------------
def toy_corpus(n=60, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    docs, feats = [], {}
    for split in ("S1", "S2", "S3"):
        for i in range(n):
            for lab, gen in ((0, "human"), (1, "A" if i % 2 else "B")):
                if split == "S3" and lab == 1:
                    pass
                did = f"{split}{i}|{gen}{lab}"
                text = ("lorem ipsum dolor " * 20 + ("zebra quantum " * 10 if lab and signal else "apple river " * 10))
                docs.append({"doc_id": did, "article_id": f"{split}{i}", "split": split, "generator": gen, "label": lab,
                             "text": text})
                f = {c: float(rng.normal(lab * 2.0 if signal else 0.0, 1)) for c in FEATURE_COLUMNS}
                feats[did] = f
    return docs, feats


def test_feature_detectors_separate_and_noise_does_not():
    from sklearn.metrics import roc_auc_score
    for signal, lo, hi in ((True, 0.95, 1.01), (False, 0.3, 0.7)):
        docs, feats = toy_corpus(signal=signal)
        tr = [d for d in docs if d["split"] == "S1"]
        te = [d for d in docs if d["split"] == "S3"]
        y = [d["label"] for d in te]
        for kind in ("lr_features", "gbdt_features"):
            s = fit_predict_features(kind, tr, [te], feats, 0)[0]
            assert lo <= roc_auc_score(y, s) <= hi, (kind, signal)


def test_tfidf_detector_uses_text():
    from sklearn.metrics import roc_auc_score
    docs, _ = toy_corpus(signal=True)
    tr = [d for d in docs if d["split"] == "S1"]
    te = [d for d in docs if d["split"] == "S3"]
    s = fit_predict_tfidf(tr, [te], 0)[0]
    assert roc_auc_score([d["label"] for d in te], s) > 0.99


def test_nan_features_are_imputed_not_crashed():
    docs, feats = toy_corpus(n=20)
    k = docs[0]["doc_id"]
    feats[k]["ll_std"] = float("nan")
    tr = [d for d in docs if d["split"] == "S1"]
    te = [d for d in docs if d["split"] == "S3"]
    s = fit_predict_features("lr_features", tr, [te], feats, 0)[0]
    assert np.isfinite(s).all()


def test_distilbert_code_path_runs_on_tiny_model(tiny):
    docs, _ = toy_corpus(n=16)
    tr = [d for d in docs if d["split"] == "S1"]
    te = [d for d in docs if d["split"] == "S3"]
    cfg = {"hf_id": tiny["distil"], "epochs": 1, "lr": 1e-3, "batch_size": 8, "max_len": 64}
    s = fit_predict_distilbert(tr, [te], cfg, 0, "cpu")[0]
    assert len(s) == len(te) and np.isfinite(s).all()


def test_conditions_and_run_detectors_schema(tiny):
    assert make_conditions(["A"]) == ["seen"]
    assert make_conditions(["A", "B"]) == ["seen", "unseen:A", "unseen:B"]
    docs, feats = toy_corpus(n=30)
    for d in docs:                       # zero-shot columns
        feats[d["doc_id"]].update({"ll_mean": feats[d["doc_id"]]["ll_mean"], "fdg": feats[d["doc_id"]]["fdg"]})
    cfg = {"seed": 0, "generation": {"generators": [{"name": "A"}, {"name": "B"}]},
           "detectors": {"names": list(ZERO_SHOT) + ["lr_features", "tfidf_lr", "distilbert"],
                         "distilbert": {"enabled": True, "hf_id": tiny["distil"], "epochs": 1, "lr": 1e-3,
                                        "batch_size": 16, "max_len": 64}}}
    df = run_detectors(docs, feats, cfg, "cpu", log=lambda *a: None)
    assert set(df.columns) == {"doc_id", "split", "label", "generator", "detector", "condition", "score"}
    assert set(df.split) == {"S2", "S3"} and set(df.condition) == {"seen", "unseen:A", "unseen:B"}
    assert len(df) == 3 * 6 * 120          # 3 conditions x 6 detectors x (S2+S3: 2 splits x 30 articles x 2 docs)
    assert df.score.notna().all()


def test_run_detectors_fails_without_training_data():
    docs, feats = toy_corpus(n=10)
    docs = [d for d in docs if d["split"] != "S1"]
    cfg = {"seed": 0, "generation": {"generators": [{"name": "A"}]}, "detectors": {"names": ["lr_features"], "distilbert": {"enabled": False}}}
    with pytest.raises(ValueError):
        run_detectors(docs, feats, cfg, "cpu", log=lambda *a: None)
