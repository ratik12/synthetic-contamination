"""Stage 05: detectors. Every detector returns a real-valued score where HIGHER means "more synthetic".

Zero-shot detectors (gpt2_loglik, fast_detectgpt, binoculars) use a fixed, a-priori orientation; nothing is learned.
Fusion from the research plan is not implemented (disclosed in README)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

# name -> (feature column, sign). Binoculars is LOWER for machine text, so its sign is -1.
ZERO_SHOT = {"gpt2_loglik": ("ll_mean", 1.0), "fast_detectgpt": ("fdg", 1.0), "binoculars": ("bino", -1.0)}


def _matrix(docs, feats):
    return np.array([[feats[d["doc_id"]][c] for c in FEATURE_COLUMNS] for d in docs], dtype=float)


def _impute(Xtr, Xs):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fix = lambda X: np.where(np.isnan(X), med, X)
    return fix(Xtr), [fix(X) for X in Xs]


def _balanced_weights(y):
    y = np.asarray(y)
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n:
            w[y == c] = len(y) / (2.0 * n)
    return w


def fit_predict_features(kind, train, evals, feats, seed):
    Xtr, Xev = _impute(_matrix(train, feats), [_matrix(e, feats) for e in evals])
    ytr = np.array([d["label"] for d in train])
    if kind == "lr_features":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
        return [m.decision_function(sc.transform(X)) for X in Xev]
    if kind == "gbdt_features":
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=seed)
        m.fit(Xtr, ytr, sample_weight=_balanced_weights(ytr))
        out = []
        for X in Xev:
            p = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
            out.append(np.log(p / (1 - p)))
        return out
    raise ValueError(kind)


def fit_predict_tfidf(train, evals, seed):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    Xtr = vec.fit_transform([d["text"] for d in train])
    ytr = np.array([d["label"] for d in train])
    m = LogisticRegression(C=3.0, class_weight="balanced", max_iter=2000, random_state=seed).fit(Xtr, ytr)
    return [m.decision_function(vec.transform([d["text"] for d in e])) for e in evals]


def fit_predict_distilbert(train, evals, dcfg, seed, device):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from .models import free_cuda, hf_token
    import os
    kw = {"token": hf_token()} if not os.path.isdir(dcfg["hf_id"]) else {}
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(dcfg["hf_id"], **kw)
    model = AutoModelForSequenceClassification.from_pretrained(dcfg["hf_id"], num_labels=2, **kw).to(device)
    ytr = np.array([d["label"] for d in train])
    cw = torch.tensor([len(ytr) / (2.0 * max(1, (ytr == c).sum())) for c in (0, 1)], dtype=torch.float, device=device)
    texts = [d["text"] for d in train]
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(list(zip(texts, ytr.tolist())), batch_size=dcfg["batch_size"], shuffle=True, generator=g,
                        collate_fn=lambda b: b)
    opt = torch.optim.AdamW(model.parameters(), lr=dcfg["lr"], weight_decay=0.01)
    steps = max(1, len(loader) * dcfg["epochs"])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    model.train()
    for ep in range(dcfg["epochs"]):
        for batch in loader:
            enc = tok([t for t, _ in batch], truncation=True, max_length=dcfg["max_len"], padding=True,
                      return_tensors="pt").to(device)
            y = torch.tensor([l for _, l in batch], device=device)
            loss = lossf(model(**enc).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    outs = []
    with torch.no_grad():
        for e in evals:
            s = []
            for i in range(0, len(e), 64):
                enc = tok([d["text"] for d in e[i:i + 64]], truncation=True, max_length=dcfg["max_len"], padding=True,
                          return_tensors="pt").to(device)
                lg = model(**enc).logits.float()
                s.extend((lg[:, 1] - lg[:, 0]).cpu().numpy().tolist())
            outs.append(np.array(s))
    del model
    free_cuda()
    return outs


def make_conditions(generators):
    """'seen' = train on all generators; 'unseen:g' = train without g's twins (leave-one-generator-out)."""
    conds = ["seen"]
    if len(generators) >= 2:
        conds += [f"unseen:{g}" for g in generators]
    return conds


def run_detectors(corpus, feats, cfg, device, log=print):
    """Returns DataFrame [doc_id, split, label, generator, detector, condition, score] for every S2 and S3 document."""
    gens = [g["name"] for g in cfg["generation"]["generators"]]
    seed = cfg["seed"]
    train_pool = [d for d in corpus if d["split"] == "S1"]
    ev = [d for d in corpus if d["split"] in ("S2", "S3")]
    if not train_pool or not ev:
        raise ValueError("S1 / S2 / S3 documents missing; run stages 01-03 first")
    rows = []
    for cond in make_conditions(gens):
        held = cond.split(":", 1)[1] if cond.startswith("unseen:") else None
        train = [d for d in train_pool if held is None or d["generator"] != held]
        if len({d["label"] for d in train}) < 2:
            raise ValueError(f"condition {cond}: training set lacks one class")
        log(f"condition {cond}: train {len(train)} docs ({sum(d['label'] for d in train)} synthetic)")
        for name in cfg["detectors"]["names"]:
            if name in ZERO_SHOT:
                col, sign = ZERO_SHOT[name]
                s = sign * np.array([feats[d["doc_id"]][col] for d in ev], dtype=float)
            elif name in ("lr_features", "gbdt_features"):
                s = fit_predict_features(name, train, [ev], feats, seed)[0]
            elif name == "tfidf_lr":
                s = fit_predict_tfidf(train, [ev], seed)[0]
            elif name == "distilbert":
                if not cfg["detectors"]["distilbert"]["enabled"]:
                    continue
                s = fit_predict_distilbert(train, [ev], cfg["detectors"]["distilbert"], seed, device)[0]
            else:
                raise ValueError(f"unknown detector {name}")
            for d, v in zip(ev, s):
                rows.append({"doc_id": d["doc_id"], "split": d["split"], "label": d["label"],
                             "generator": d["generator"], "detector": name, "condition": cond, "score": float(v)})
            log(f"  {name}: done")
    df = pd.DataFrame(rows)
    if df["score"].isna().any():
        n = int(df["score"].isna().sum())
        raise ValueError(f"{n} NaN detector scores; check feature extraction")
    return df
