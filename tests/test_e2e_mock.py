"""End-to-end run of stages 01-08 through the real scripts on stdlib prose, a mock generator and tiny random models.
Verifies plumbing, file contracts, leakage checks and mock-safety. The numbers are NOT results."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.helpers import prose_articles

REPO = Path(__file__).resolve().parents[1]


def cfg_dict(tiny, local):
    return {
        "experiment_name": "e2e", "seed": 3, "mock": True, "paths": {"data": "data", "results": "results"},
        "data": {"source": "local_jsonl", "local_path": str(local), "passage_words": 200, "min_words": 150,
                 "dedup_ngram": 8, "dedup_threshold": 0.3,
                 "sizes": {"S1": 40, "S2": 20, "S3": 30, "S4": 50, "S5": 15}, "twins_S4": 25},
        "generation": {"backend": "mock", "target_words": 200, "min_words": 150, "max_new_tokens": 10, "temperature": 1,
                       "top_p": 1, "batch_size": 16, "max_attempts": 3, "dtype": "float32", "mock_fail_rate": 0.3,
                       "generators": [{"name": "mockA", "hf_id": "mock"}, {"name": "mockB", "hf_id": "mock"}]},
        "quality": {"copy_ngram": 8, "max_len_ratio": 1.5},
        "scorer": {"hf_id": tiny["scorer"], "max_len": 128, "batch_size": 8},
        "binoculars": {"observer_id": tiny["scorer"], "performer_id": tiny["genA"], "dtype": "float32", "max_len": 128,
                       "batch_size": 4},
        "detectors": {"names": ["gpt2_loglik", "fast_detectgpt", "binoculars", "lr_features", "gbdt_features", "tfidf_lr", "distilbert"],
                      "distilbert": {"enabled": True, "hf_id": tiny["distil"], "epochs": 1, "lr": 0.001, "batch_size": 16,
                                     "max_len": 64}},
        "audit": {"alphas": [0.0, 0.2, 0.5], "set_size": 25, "draws_per_alpha": 3, "n_boot": 30, "alpha_grid_step": 0.02},
        "downstream": {"total_docs": 50, "d_fractions": [0.0, 0.1, 0.3, 0.5]},
    }


@pytest.fixture(scope="module")
def run_dir(tiny, tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    arts = prose_articles(200, seed=21)
    (d / "arts.jsonl").write_text("\n".join(json.dumps(a) for a in arts))
    (d / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict(tiny, d / "arts.jsonl")))
    return d


def run(d, script, *extra):
    env = {**os.environ, "CAPSTONE_LOG": str(d / "LOG.md")}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--config", str(d / "cfg.yaml"), *extra],
                       cwd=d, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}"
    return r.stdout


def test_pipeline_end_to_end(run_dir):
    d = run_dir
    out = run(d, "01_prepare_data.py")
    assert "MOCK / TEST RUN" in out
    man = json.loads((d / "data/e2e/data_manifest.json").read_text())
    assert man["split_sizes"] == {"S1": 40, "S2": 20, "S3": 30, "S4": 50, "S5": 15} and man["mock"] is True

    run(d, "02_generate.py")
    gen = json.loads((d / "data/e2e/generation_summary_all.json").read_text())
    assert all(v["requested"] > 0 and v["ok"] + v["failed_after_retries"] == v["requested"] for v in gen.values())
    raw = [json.loads(l) for l in (d / "data/e2e/generated_mockA.jsonl").read_text().splitlines()]
    assert any(r["attempt"] >= 1 for r in raw), "mock_fail_rate=0.3 should have forced retries"
    assert all(r["backend"] == "mock" for r in raw)

    run(d, "03_build_corpus.py")
    corpus = [json.loads(l) for l in (d / "data/e2e/corpus.jsonl").read_text().splitlines()]
    assert {x["split"] for x in corpus} == {"S1", "S2", "S3", "S4", "S5"}
    assert not any(x["split"] == "S5" and x["label"] == 1 for x in corpus)
    s3_syn = [x for x in corpus if x["split"] == "S3" and x["label"] == 1]
    assert {x["generator"] for x in s3_syn} == {"mockA", "mockB"}

    run(d, "04_extract_features.py")
    run(d, "05_train_detectors.py")
    scores = pd.read_csv(d / "results/e2e/scores.csv")
    assert set(scores.condition) == {"seen", "unseen:mockA", "unseen:mockB"} and set(scores.detector) >= {"distilbert", "tfidf_lr", "binoculars"}
    assert set(scores.split) == {"S2", "S3"} and scores.score.notna().all()

    run(d, "06_eval_detectors.py")
    det = pd.read_csv(d / "results/e2e/detection_metrics.csv")
    assert {"auroc", "auprc", "tpr_at_1pct_fpr", "auroc_lo", "auroc_hi"} <= set(det.columns)
    unseen = det[det.condition == "unseen:mockA"]
    assert set(unseen.eval_generator) == {"mockA"}

    run(d, "07_audit_estimate.py")
    est = pd.read_csv(d / "results/e2e/estimates.csv")
    assert set(est.estimator) == {"cc", "pcc", "acc", "mle"}
    assert set(est.alpha.round(2)) == {0.0, 0.2, 0.48}          # round(alpha * 25) / 25
    assert (d / "results/e2e/estimator_summary.csv").exists()

    run(d, "08_prepare_downstream.py")
    man = json.loads((d / "data/e2e/downstream/manifest.json").read_text())
    assert set(man["arms"]) == {"D0", "D10", "D30", "D50", "H90", "H70", "H50"}
    assert all(v["n_docs"] == 50 for k, v in man["arms"].items() if k.startswith("D"))
    assert man["arms"]["D50"]["n_synthetic"] == 25 and man["n_eval_S5"] == 15

    log = (d / "LOG.md").read_text()
    for stage in ("01_prepare_data", "02_generate", "03_build_corpus", "04_extract_features", "05_train_detectors",
                  "06_eval_detectors", "07_audit_estimate", "08_prepare_downstream"):
        assert stage in log


def test_stage_03_refuses_mock_records_when_mock_flag_off(run_dir, tiny):
    d = run_dir
    cfg = yaml.safe_load((d / "cfg.yaml").read_text())
    cfg["mock"] = False
    (d / "cfg_real.yaml").write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "CAPSTONE_LOG": str(d / "LOG.md")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/03_build_corpus.py"), "--config", str(d / "cfg_real.yaml")],
                       cwd=d, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "mock" in r.stderr.lower()
    assert "FAILED" in (d / "LOG.md").read_text()


def test_stage_02_real_backend_refuses_cpu(run_dir):
    d = run_dir
    cfg = yaml.safe_load((d / "cfg.yaml").read_text())
    cfg["generation"]["backend"] = "hf"
    cfg["mock"] = False
    (d / "cfg_hf.yaml").write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "CAPSTONE_LOG": str(d / "LOG.md")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/02_generate.py"), "--config", str(d / "cfg_hf.yaml")],
                       cwd=d, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "GPU" in r.stderr


def test_smoke_limit_generates_only_n_per_generator(tiny, tmp_path):
    arts = prose_articles(200, seed=21)
    (tmp_path / "arts.jsonl").write_text("\n".join(json.dumps(a) for a in arts))
    cfg = cfg_dict(tiny, tmp_path / "arts.jsonl")
    cfg["generation"]["mock_fail_rate"] = 0.0
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    run(tmp_path, "01_prepare_data.py")
    out = run(tmp_path, "02_generate.py", "--extra", "limit=5")
    assert "SMOKE MODE" in out
    for g in ("mockA", "mockB"):
        assert len((tmp_path / f"data/e2e/generated_{g}.jsonl").read_text().splitlines()) == 5
    assert not (tmp_path / "data/e2e/pairs.jsonl").exists()
    run(tmp_path, "02_generate.py")                                   # full run resumes: smoke records are kept
    for g in ("mockA", "mockB"):
        recs = [json.loads(l) for l in (tmp_path / f"data/e2e/generated_{g}.jsonl").read_text().splitlines()]
        assert len({r["article_id"] for r in recs}) > 5


def test_report_tables_refuse_mock_and_stamp_when_allowed(run_dir):
    """Runs after test_pipeline_end_to_end (same module-scoped directory)."""
    d = run_dir
    env = {**os.environ, "CAPSTONE_LOG": str(d / "LOG.md")}
    cmd = [sys.executable, str(REPO / "scripts/09_make_report_tables.py"), "--config", str(d / "cfg.yaml")]
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "mock" in r.stderr.lower()
    assert not (d / "report" / "gen" / "det_seen.tex").exists()
    run(d, "09_make_report_tables.py", "--extra", "out=report", "allow_mock=1")
    gen = d / "report" / "gen"
    for name in ("design_splits", "design_levels", "design_macros", "data_corpus", "data_synth", "det_seen",
                 "det_unseen", "est_all", "est_sel"):
        text = (gen / f"{name}.tex").read_text()
        assert text.strip(), name
        if name != "design_macros":
            assert "MOCK DATA" in text, name
    assert "\\nArticles" in (gen / "design_macros.tex").read_text()
    assert (d / "report" / "fig" / "auroc.pdf").stat().st_size > 1000
    assert "Assigned" not in (gen / "data_corpus.tex").read_text()
