"""File, config and logging helpers shared by every pipeline stage."""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def data_dir(cfg):
    return Path(cfg["paths"]["data"]) / cfg["experiment_name"]


def results_dir(cfg):
    return Path(cfg["paths"]["results"]) / cfg["experiment_name"]


def ensure(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """Atomic write so an interrupted run never leaves a half-written file."""
    path = Path(path)
    ensure(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    path = Path(path)
    ensure(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path, obj):
    path = Path(path)
    ensure(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def log_run(cfg, stage, status, note=""):
    """Append one line per stage execution to EXPERIMENT_LOG.md (written by code, never by hand)."""
    log = Path(os.environ.get("CAPSTONE_LOG", REPO / "EXPERIMENT_LOG.md"))
    if not log.exists():
        log.write_text("# Experiment log\n\n| UTC time | experiment | stage | config sha256 | commit | status | note |\n|---|---|---|---|---|---|---|\n")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = (f"| {ts} | {cfg['experiment_name']} | {stage} | {cfg['_config_sha256'][:12]} | "
            f"{git_commit()[:10]} | {status} | {note} |\n")
    with open(log, "a", encoding="utf-8") as f:
        f.write(line)


def pick_device(pref="auto"):
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def banner_if_mock(cfg):
    """Mock runs must never be mistaken for results."""
    if cfg.get("mock", False):
        print("=" * 70)
        print("MOCK / TEST RUN: outputs are plumbing checks, NOT experimental results")
        print("=" * 70)
        return True
    return False


def stage_main(stage_name, fn):
    """Run a stage, log success/failure into EXPERIMENT_LOG.md."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="stage-specific key=value options")
    args = ap.parse_args()
    cfg = load_config(args.config)
    banner_if_mock(cfg)
    set_seed(cfg["seed"])
    opts = dict(kv.split("=", 1) for kv in args.extra)
    try:
        note = fn(cfg, opts) or ""
    except Exception as e:
        log_run(cfg, stage_name, "FAILED", f"{type(e).__name__}: {str(e)[:120]}")
        raise
    log_run(cfg, stage_name, "ok", note)
