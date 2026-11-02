"""Stage 09: turn pipeline outputs into LaTeX tables, figures and factual sentences for the report.

Design rule: every number in the report comes from a file in data/ or results/ (or from the config for planned design
counts). Sentences are generated mechanically from those numbers; interpretation is left to the author.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DET_NAMES = {
    "gpt2_loglik": "GPT-2 log-likelihood",
    "fast_detectgpt": "Fast-DetectGPT (analytic)",
    "binoculars": "Binoculars (SmolLM2-360M)",
    "lr_features": "Logistic regression (features)",
    "gbdt_features": "Gradient boosting (features)",
    "tfidf_lr": "TF-IDF logistic regression",
    "distilbert": "DistilBERT (fine-tuned)",
}
EST_NAMES = {"cc": "CC", "pcc": "PCC", "acc": "ACC", "mle": "MLE"}
EST_ORDER = ["cc", "pcc", "acc", "mle"]


def tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def dname(d):
    return DET_NAMES.get(d, tex(d))


def f3(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pp(x, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"


def _cap(caption, mock):
    return (r"\textbf{[MOCK DATA]} " if mock else "") + caption


def _short(caption):
    """Entry for the list of tables: the first sentence only (braces keep a ']' from ending the option early)."""
    first = caption.split(". ")[0].rstrip(".")
    return "{" + first + "}"


def _table(label, caption, colspec, header, rows, mock, size="footnotesize", extra_after="", pre_header=None):
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\{size}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if pre_header:
        lines.append(pre_header)
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption[{_short(caption)}]{{{_cap(caption, mock)}}}", rf"\label{{{label}}}"]
    if extra_after:
        lines.append(extra_after)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------- design tables
def design_splits(cfg):
    d, g = cfg["data"], len(cfg["generation"]["generators"])
    s = d["sizes"]
    purpose = {"S1": "Detector training", "S2": "Calibration of thresholds and score densities",
               "S3": "Test set and audit pool", "S4": "Downstream training pool",
               "S5": "Held-out human text for downstream evaluation"}
    syn = {"S1": s["S1"], "S2": s["S2"], "S3": s["S3"] * g, "S4": d["twins_S4"], "S5": 0}
    rows = [[k, purpose[k], f"{s[k]:,}", f"{syn[k]:,}"] for k in ("S1", "S2", "S3", "S4", "S5")]
    rows.append(r"\midrule")
    rows.append([r"\textbf{Total}", "", f"{sum(s.values()):,}", f"{sum(syn.values()):,}"])
    return _table("tab:splits", "Splits of the human passages and the synthetic twins to be generated (planned counts "
                  "from the configuration). Articles are assigned to a split before any text is generated.",
                  "@{}llrr@{}", ["Split", "Purpose", "Human", "Synthetic"], rows, cfg.get("mock", False), "small")


def design_levels(cfg):
    a, dn = cfg["audit"], cfg["downstream"]
    rows = []
    for f in dn["d_fractions"]:
        na, nd = int(round(f * a["set_size"])), int(round(f * dn["total_docs"]))
        pct = int(round(f * 100))
        ctrl = "--" if f == 0 else f"H{100 - pct}: {dn['total_docs'] - nd:,} human"
        rows.append([f"{pct}\\%", f"{na} / {a['set_size'] - na}", f"{nd:,} / {dn['total_docs'] - nd:,}", ctrl])
    return _table("tab:levels", "Contamination levels. Audit sets are used to test the share estimators; training sets "
                  "D are built for the downstream study. Each control H keeps exactly the human documents of its "
                  "training set and drops the synthetic ones. Cells give synthetic / human document counts.",
                  "@{}rccl@{}", ["Level", f"Audit set ({a['set_size']} docs)", f"Training set D ({dn['total_docs']:,} docs)",
                                 "Control"], rows, cfg.get("mock", False), "small")


def design_settings(cfg):
    g, d, a = cfg["generation"], cfg["data"], cfg["audit"]
    dd = cfg["detectors"]["distilbert"]
    b = cfg.get("binoculars") or {}
    rows = [
        ["Reference dump", tex(f"{d.get('hf_name', '')}/{d.get('hf_config', '')}") if d["source"] == "hf_wikipedia"
         else tex(d["source"])],
        ["Passage", f"{d['passage_words']} words, at least {d['min_words']}; near-duplicate threshold {d['dedup_threshold']}"],
        ["Generators", ", ".join(tex(x["hf_id"]) for x in g["generators"])],
        ["Decoding", f"temperature {g['temperature']}, top-$p$ {g['top_p']}, at most {g['max_new_tokens']} new tokens, "
                     f"up to {g['max_attempts']} attempts per passage"],
        ["Prompt", "article title only; no human text is shown to the generator"],
        ["Scoring model", tex(cfg["scorer"]["hf_id"]) + f", {cfg['scorer']['max_len']} tokens"],
        ["Binoculars pair", (tex(b["observer_id"]) + " / " + tex(b["performer_id"])) if b else "not configured"],
        ["DistilBERT", f"{dd['epochs']} epochs, learning rate {dd['lr']:g}, batch {dd['batch_size']}, {dd['max_len']} tokens"],
        ["Audit", f"{a['set_size']} documents per set, {a['draws_per_alpha']} draws per level, "
                  f"{a['n_boot']} bootstrap replicates"],
        ["Random seed", str(cfg["seed"])],
    ]
    return _table("tab:settings", "Experimental settings, copied from the configuration file that produced the results "
                  f"(SHA-256 prefix {cfg['_config_sha256'][:12]}).", "@{}lp{10.6cm}@{}", ["Setting", "Value"], rows,
                  cfg.get("mock", False), "footnotesize")


def design_macros(cfg):
    """LaTeX macros so that numbers quoted in the prose always equal the configuration."""
    d, a, dn = cfg["data"], cfg["audit"], cfg["downstream"]
    g = len(cfg["generation"]["generators"])
    sz = d["sizes"]
    twins = sz["S1"] + sz["S2"] + sz["S3"] * g + d["twins_S4"]
    n_det = sum(1 for n in cfg["detectors"]["names"] if n != "distilbert" or cfg["detectors"]["distilbert"]["enabled"])
    mac = {
        "nArticles": f"{sum(sz.values()):,}", "nTwins": f"{twins:,}", "nGenerators": str(g), "nDetectors": str(n_det),
        "nLevels": str(len(a["alphas"])), "auditSize": str(a["set_size"]), "nDraws": str(a["draws_per_alpha"]),
        "nBoot": str(a["n_boot"]), "dsTotal": f"{dn['total_docs']:,}", "passageWords": str(d["passage_words"]),
        "minWords": str(d["min_words"]), "nSplitOne": str(sz["S1"]), "nSplitTwo": str(sz["S2"]),
        "nSplitThree": str(sz["S3"]), "nSplitFour": f"{sz['S4']:,}", "nSplitFive": str(sz["S5"]),
        "nTwinsFour": str(d["twins_S4"]), "seed": str(cfg["seed"]),
    }
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items())


# ---------------------------------------------------------------------------------------------- measured: dataset
def data_corpus(dm, cfg):
    st = dm["filter_stats"]
    rows = [["Articles sampled", f"{st['scanned']:,}"], ["Disambiguation and list pages removed", f"{st['bad_article']:,}"],
            ["Shorter than the minimum length", f"{st['too_short']:,}"], ["Exact duplicates", f"{st['exact_dup']:,}"],
            ["Near duplicates", f"{st['near_dup']:,}"], ["Eligible articles", f"{st['eligible']:,}"]]
    src = dm["source"]
    note = ""
    if src.get("fallback_used"):
        note = r"\par\smallskip\textcolor{red}{\footnotesize Deviation: the fallback dump (" + tex(str(src.get("config"))) + \
               r") was used instead of the 1 March 2022 dump.}"
    return _table("tab:corpus", "Construction of the human corpus (measured).", "@{}lr@{}", ["Step", "Articles"], rows,
                  dm.get("mock", False), "small", note)


def data_synth(qr, gen_stats):
    rows = []
    h_term = qr["ends_with_terminal_punct"].get("human")
    rows.append(["human", f"{qr['n_human']:,}", f"{qr['human_mean_words']:.0f}", "1.00", pp(h_term), "--"])
    for g, v in qr["per_generator"].items():
        gs = gen_stats.get(g, {})
        rows.append([tex(g), f"{v['n']:,}", f"{v['mean_words']:.0f}", f"{v['len_ratio_vs_human']:.2f}",
                     pp(qr["ends_with_terminal_punct"].get(g)),
                     "--" if "retried" not in gs else f"{gs['retried']:,}"])
    dr = qr["dropped"]
    note = (r"\par\smallskip{\footnotesize Rejected generations: " +
            ", ".join(f"{tex(k.replace('_', ' '))} {v}" for k, v in dr.items() if v) + ".}") if any(dr.values()) else ""
    return _table("tab:synth", "Accepted passages per source (measured). Length ratio is the mean synthetic length "
                  "divided by the mean human length; ``ends cleanly'' is the percentage of passages ending in "
                  "terminal punctuation.", "@{}lrrrrr@{}", ["Source", "Passages", "Mean words", "Length ratio",
                                                            "Ends cleanly (\\%)", "Needed retry"], rows,
                  qr.get("mock", False), "small", note)


# ------------------------------------------------------------------------------------------ measured: detectors
def _order(dets, cfg):
    names = [n for n in cfg["detectors"]["names"] if n in dets]
    return names + [d for d in sorted(dets) if d not in names]


def det_seen_table(det, cfg, mock):
    seen = det[(det.condition == "seen") & (det.eval_generator == "all")].set_index("detector")
    dets = _order(list(seen.index), cfg)
    best = {c: seen[c].max() for c in ("auroc", "auprc", "tpr_at_1pct_fpr")}
    b = lambda c, v: (r"\textbf{" + f3(v) + "}") if v == best[c] else f3(v)
    rows = []
    for d in dets:
        r = seen.loc[d]
        rows.append([dname(d), b("auroc", r.auroc), f"[{f3(r.auroc_lo)}, {f3(r.auroc_hi)}]", b("auprc", r.auprc),
                     b("tpr_at_1pct_fpr", r.tpr_at_1pct_fpr)])
    return _table("tab:det_seen", "Detection on the test split S3, detectors trained on all generators (measured). "
                  "AUROC with 95\\% bootstrap interval, area under the precision--recall curve, and true-positive "
                  "rate at 1\\% false-positive rate. Best value per column in bold.", "@{}lcccc@{}",
                  ["Detector", "AUROC", "95\\% CI", "AUPRC", "TPR@1\\%FPR"], rows, mock)


def det_unseen_table(det, cfg, mock):
    unseen = det[det.condition.str.startswith("unseen:")]
    if unseen.empty:
        return None
    seen_g = det[(det.condition == "seen") & (det.eval_generator != "all")]
    gens = sorted(unseen.eval_generator.unique())
    dets = _order(list(unseen.detector.unique()), cfg)
    header = ["Detector"]
    for g in gens:
        header += [f"{tex(g)}: seen", "unseen", r"$\Delta$"]
    rows = []
    for d in dets:
        row = [dname(d)]
        for g in gens:
            s = seen_g[(seen_g.detector == d) & (seen_g.eval_generator == g)].auroc
            u = unseen[(unseen.detector == d) & (unseen.eval_generator == g)].auroc
            sv, uv = (s.iloc[0] if len(s) else float("nan")), (u.iloc[0] if len(u) else float("nan"))
            row += [f3(sv), f3(uv), f"{uv - sv:+.3f}" if not (math.isnan(sv) or math.isnan(uv)) else "--"]
        rows.append(row)
    return _table("tab:det_unseen", "AUROC on S3 for each generator when the detector has seen it in training "
                  "(``seen'') and when it was left out of training (``unseen''), and the change $\\Delta$ "
                  "(measured). With two generators, ``unseen'' means trained on the other generator only.",
                  "@{}l" + "ccc" * len(gens) + "@{}", header, rows, mock, "scriptsize")


# ------------------------------------------------------------------------------------------ measured: estimators
def _overall(summ, **filt):
    s = summ
    for k, v in filt.items():
        s = s[s[k] == v]
    return float(np.sqrt(np.mean(np.square(s["rmse"])))) if len(s) else float("nan")


def select_detector(scores, det_seen_auroc_s2=None):
    """Detector with the highest AUROC on S2 (seen condition). S2 is calibration data, never the audit pool S3."""
    from sklearn.metrics import roc_auc_score
    s2 = scores[(scores.split == "S2") & (scores.condition == "seen")]
    best, best_auc = None, -1
    for d, g in s2.groupby("detector"):
        auc = roc_auc_score(g.label, g.score)
        if auc > best_auc:
            best, best_auc = d, auc
    return best, best_auc


def est_all_table(summ, cfg, mock):
    dets = _order(list(summ.detector.unique()), cfg)
    conds_unseen = [c for c in summ.condition.unique() if c.startswith("unseen:")]
    rows = []
    for d in dets:
        row = [dname(d)]
        for cond_sel in ("seen", "unseen"):
            for e in EST_ORDER:
                sub = summ[(summ.detector == d) & (summ.estimator == e)]
                sub = sub[sub.condition == "seen"] if cond_sel == "seen" else sub[sub.condition.isin(conds_unseen)]
                row.append(pp(_overall(sub)) if len(sub) else "--")
        rows.append(row)
    nan_total = int(summ[summ.estimator == "acc"]["n_nan"].sum())
    n_total = int(summ[summ.estimator == "acc"]["n"].sum())
    note = rf"\par\smallskip{{\footnotesize ACC was undefined (TPR$-$FPR$<0.05$) in {nan_total} of {n_total} audits; " \
           r"those audits are left out of its error.}" if nan_total else ""
    header = ["Detector"] + [EST_NAMES[e] for e in EST_ORDER] * 2
    pre = (r" & \multicolumn{4}{c}{Seen generators} & \multicolumn{4}{c}{Unseen generator} \\" + "\n" +
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    return _table("tab:est_all", "Root-mean-square error of the estimated synthetic share, in percentage points, pooled "
                  "over all contamination levels (measured). Left: detector trained on all generators. Right: audit "
                  "sets contain only a generator left out of detector training and calibration.",
                  "@{}l" + "r" * 8 + "@{}", header, rows, mock, "scriptsize", note, pre_header=pre)


def est_selected_table(summ, detector, cfg, mock):
    s = summ[(summ.detector == detector) & (summ.condition == "seen")]
    alphas = sorted(s.alpha.unique())
    rows = []
    for a in alphas:
        row = [f"{100 * a:.0f}\\%"]
        for e in EST_ORDER:
            r = s[(s.estimator == e) & (np.isclose(s.alpha, a))]
            row.append("--" if r.empty else f"{pp(r.bias.iloc[0], True)} / {pp(r.rmse.iloc[0])}")
        rows.append(row)
    rows.append(r"\midrule")
    rows.append(["Coverage of 95\\% CI"] + [
        f"{100 * s[s.estimator == e].coverage.mean():.0f}\\%" if (s.estimator == e).any() else "--" for e in EST_ORDER])
    return _table("tab:est_sel", f"Estimation error for {dname(detector)} (the detector with the highest AUROC on the "
                  "calibration split S2), detector trained on all generators (measured). Cells give bias / RMSE in "
                  "percentage points over the audit draws; the last row is the share of 95\\% bootstrap intervals "
                  "that contain the true share, averaged over levels.", "@{}lrrrr@{}",
                  ["True share"] + [EST_NAMES[e] for e in EST_ORDER], rows, mock)


# ------------------------------------------------------------------------------------------------------ figures
def make_figures(det, est, summ, detector, out_dir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    INK, SEC, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
                         "xtick.color": SEC, "ytick.color": SEC, "axes.labelcolor": SEC, "text.color": INK,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    out_dir = Path(out_dir)
    # Figure 1: AUROC per detector, seen vs unseen
    seen = det[(det.condition == "seen") & (det.eval_generator == "all")].set_index("detector")
    dets = _order(list(seen.index), cfg)[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 0.45 * len(dets) + 1.1))
    ax.axvline(0.5, color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=1)
    unseen = det[det.condition.str.startswith("unseen:")]
    markers = ["^", "s", "D", "v"]
    gens = sorted(unseen.eval_generator.unique())
    for i, d in enumerate(dets):
        r = seen.loc[d]
        ax.plot([r.auroc_lo, r.auroc_hi], [i + 0.12] * 2, color=BLUE, lw=1.4, zorder=2)
        ax.scatter([r.auroc], [i + 0.12], s=26, color=BLUE, edgecolors="white", linewidths=0.6, zorder=3,
                   label="Seen generators, all (95% CI)" if i == 0 else None)
        for k, g in enumerate(gens):
            u = unseen[(unseen.detector == d) & (unseen.eval_generator == g)]
            if u.empty:
                continue
            u = u.iloc[0]
            y = i - 0.12 - 0.1 * k
            ax.plot([u.auroc_lo, u.auroc_hi], [y] * 2, color=ORANGE, lw=1.4, zorder=2)
            ax.scatter([u.auroc], [y], s=26, color=ORANGE, marker=markers[k % 4], edgecolors="white", linewidths=0.6,
                       zorder=3, label=f"Unseen: {g} (95% CI)" if i == 0 else None)
    ax.set_yticks(range(len(dets)))
    ax.set_yticklabels([DET_NAMES.get(d, d) for d in dets])
    ax.set_xlabel("AUROC on test split S3 (dashed line: chance)")
    ax.set_xlim(0.3, 1.02)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1 + len(gens), frameon=False, fontsize=7,
              handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "auroc.pdf", bbox_inches="tight")
    plt.close(fig)
    # Figure 2: estimated vs true share for the selected detector
    est_s = est[(est.detector == detector) & (est.condition == "seen")]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=0.9, ls=(0, (3, 3)), zorder=1)
    cols = {"cc": BLUE, "pcc": ORANGE, "acc": AQUA, "mle": VIOLET}
    mk = {"cc": "o", "pcc": "^", "acc": "s", "mle": "D"}
    for e in EST_ORDER:
        g = est_s[est_s.estimator == e].groupby("alpha")["estimate"]
        x, m = np.array(sorted(g.groups)) * 100, np.array([g.get_group(a).mean() for a in sorted(g.groups)]) * 100
        lo = np.array([g.get_group(a).quantile(0.05) for a in sorted(g.groups)]) * 100
        hi = np.array([g.get_group(a).quantile(0.95) for a in sorted(g.groups)]) * 100
        ax.fill_between(x, lo, hi, color=cols[e], alpha=0.12, lw=0, zorder=1)
        ax.plot(x, m, color=cols[e], lw=1.6, marker=mk[e], ms=4.5, mec="white", mew=0.6, zorder=3, label=EST_NAMES[e])
        ax.annotate(EST_NAMES[e], (x[-1], m[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center",
                    color=INK)
    ax.set_xlabel("True synthetic share (%)")
    ax.set_ylabel("Estimated share (%): mean, 5th to 95th percentile")
    ax.set_xlim(-2, 58)
    ax.set_ylim(-5, 105 if est_s["estimate"].max() > 0.6 else 70)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "estimates.pdf", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------- factual sentences
def results_text(det, summ, scores, cfg, detector):
    """Returns {"all": paragraph, "abstract": two sentences}. Every sentence restates numbers from the tables."""
    seen = det[(det.condition == "seen") & (det.eval_generator == "all")].set_index("detector")
    best, worst = seen.auroc.idxmax(), seen.auroc.idxmin()
    det_s = (f"On the test split S3, with detectors trained on all generators, AUROC ranged from "
             f"{f3(seen.auroc[worst])} ({dname(worst)}) to {f3(seen.auroc[best])} ({dname(best)}). "
             f"The best detector reached a true-positive rate of {f3(seen.tpr_at_1pct_fpr[best])} at a 1\\% "
             f"false-positive rate.")
    s = [det_s]
    un = det[det.condition.str.startswith("unseen:")]
    if not un.empty:
        sg = det[(det.condition == "seen") & (det.eval_generator != "all")]
        m = un.merge(sg, on=["detector", "eval_generator"], suffixes=("_u", "_s"))
        m["drop"] = m.auroc_s - m.auroc_u
        top = m.loc[m["drop"].idxmax()]
        low = m.loc[m["drop"].idxmin()]
        s.append(f"When the generator was left out of training, the largest AUROC change was "
                 f"{-top['drop']:+.3f} ({dname(top.detector)} on {tex(top.eval_generator)}) and the smallest "
                 f"{-low['drop']:+.3f} ({dname(low.detector)} on {tex(low.eval_generator)}).")
    sel = summ[(summ.detector == detector) & (summ.condition == "seen")]
    rm = {e: _overall(sel, estimator=e) for e in EST_ORDER if (sel.estimator == e).any()}
    bestE = min(rm, key=lambda k: rm[k] if not math.isnan(rm[k]) else 9)
    est_s = (f"For {dname(detector)}, the detector with the highest AUROC on the calibration split, the "
             f"root-mean-square error of the estimated synthetic share, pooled over contamination levels, was "
             + ", ".join(f"{pp(rm[e])} points for {EST_NAMES[e]}" for e in rm) + f"; the lowest was {EST_NAMES[bestE]}.")
    s.append(est_s)
    z = sel[np.isclose(sel.alpha, 0.0)]
    if not z.empty:
        s.append("On audit sets with no synthetic text the mean bias was " +
                 ", ".join(f"{pp(z[z.estimator == e].bias.iloc[0], True)} points for {EST_NAMES[e]}"
                           for e in EST_ORDER if (z.estimator == e).any()) + ".")
    cov = {e: sel[sel.estimator == e].coverage.mean() for e in rm}
    s.append("Averaged over levels, the 95\\% bootstrap intervals contained the true share in " +
             ", ".join(f"{100 * cov[e]:.0f}\\% of audits for {EST_NAMES[e]}" for e in cov) +
             ". Because every audit set is drawn from the same S3 articles, these coverage figures are optimistic.")
    return {"all": " ".join(s) + "\n", "abstract": det_s + " " + est_s + "\n"}


# ------------------------------------------------------------------------------------------------ orchestrator
def generation_stats(data_dir, generators, gcfg):
    from .text_utils import clean_generation
    out = {}
    for g in generators:
        p = Path(data_dir) / f"generated_{g}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        retried = {r["article_id"] for r in recs if r["attempt"] >= 1}
        out[g] = {"retried": len(retried)}
    return out


def build_all(cfg, data_dir, results_dir, out_dir, allow_mock=False):
    data_dir, results_dir, out_dir = Path(data_dir), Path(results_dir), Path(out_dir)
    mock = bool(cfg.get("mock"))
    if mock and not allow_mock:
        raise RuntimeError("config has mock: true. Mock outputs are plumbing checks, not results. "
                           "Pass allow_mock=True only to test the LaTeX build; tables will be stamped [MOCK DATA].")
    gen, fig = out_dir / "gen", out_dir / "fig"
    gen.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    written = []

    def w(name, text):
        (gen / f"{name}.tex").write_text(text)
        written.append(name)

    w("design_splits", design_splits(cfg))
    w("design_levels", design_levels(cfg))
    w("design_settings", design_settings(cfg))
    w("design_macros", design_macros(cfg))
    dm_p, qr_p = data_dir / "data_manifest.json", data_dir / "quality_report.json"
    if dm_p.exists():
        dm = json.loads(dm_p.read_text())
        dm["mock"] = dm.get("mock", mock)
        w("data_corpus", data_corpus(dm, cfg))
    if qr_p.exists():
        qr = json.loads(qr_p.read_text())
        qr["mock"] = mock
        gens = [g["name"] for g in cfg["generation"]["generators"]]
        w("data_synth", data_synth(qr, generation_stats(data_dir, gens, cfg["generation"])))
    det_p, est_p, sm_p, sc_p = (results_dir / n for n in ("detection_metrics.csv", "estimates.csv",
                                                          "estimator_summary.csv", "scores.csv"))
    if det_p.exists() and sc_p.exists():
        det, scores = pd.read_csv(det_p), pd.read_csv(sc_p)
        w("det_seen", det_seen_table(det, cfg, mock))
        t = det_unseen_table(det, cfg, mock)
        if t:
            w("det_unseen", t)
        if sm_p.exists() and est_p.exists():
            summ, est = pd.read_csv(sm_p), pd.read_csv(est_p)
            sel, _ = select_detector(scores)
            w("est_all", est_all_table(summ, cfg, mock))
            w("est_sel", est_selected_table(summ, sel, cfg, mock))
            rt = results_text(det, summ, scores, cfg, sel)
            w("results_text", rt["all"])
            w("abstract_results", rt["abstract"])
            make_figures(det, est, summ, sel, fig, cfg)
            written += ["fig/auroc.pdf", "fig/estimates.pdf"]
    return written
