"""Run this FIRST on Colab/Kaggle. Reports every prerequisite as OK/FAIL without crashing."""
import _boot  # noqa
import argparse
import os
import sys

from src.io_utils import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
cfg = load_config(ap.parse_args().config)
fails = 0


def check(name, fn):
    global fails
    try:
        msg = fn()
        print(f"[ OK ] {name}: {msg}")
    except Exception as e:
        fails += 1
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}")


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU. Colab: Runtime > Change runtime type > T4 GPU. Needed for stage 02 and 05 (DistilBERT).")
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 2**30:.1f} GiB"


def pkgs():
    import datasets, sklearn, torch, transformers, yaml  # noqa
    return f"torch {torch.__version__}, transformers {transformers.__version__}, datasets {datasets.__version__}"


def token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return "HF_TOKEN set" if t else "HF_TOKEN not set (the default models are not gated, but anonymous downloads are rate-limited)"


def hub(model_id):
    def f():
        from transformers import AutoConfig
        from src.models import hf_token
        c = AutoConfig.from_pretrained(model_id, token=hf_token())
        return f"{model_id} reachable ({c.model_type})"
    return f


def wiki():
    d = cfg["data"]
    if d["source"] != "hf_wikipedia":
        return f"source is {d['source']}"
    from datasets import load_dataset_builder
    b = load_dataset_builder(d["hf_name"], d["hf_config"], trust_remote_code=True)
    return f"{d['hf_name']}/{d['hf_config']} builder ok"


check("python", lambda: sys.version.split()[0])
check("packages", pkgs)
check("gpu", gpu)
check("hf token", token)
if cfg["generation"]["backend"] == "hf":
    for g in cfg["generation"]["generators"]:
        check(f"generator {g['name']}", hub(g["hf_id"]))
check("scorer", hub(cfg["scorer"]["hf_id"]))
if cfg.get("binoculars"):
    check("binoculars observer", hub(cfg["binoculars"]["observer_id"]))
    check("binoculars performer", hub(cfg["binoculars"]["performer_id"]))
if cfg["detectors"]["distilbert"]["enabled"]:
    check("distilbert", hub(cfg["detectors"]["distilbert"]["hf_id"]))
check("wikipedia dataset", wiki)
print("\nALL CHECKS PASSED" if not fails else f"\n{fails} CHECK(S) FAILED: fix before continuing")
sys.exit(1 if fails else 0)
