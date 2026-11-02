#!/usr/bin/env bash
# Run the smoke test by hand FIRST (see README step 3). Full pipeline up to (not including) downstream training. Resumable: rerun after a Colab disconnect.
set -euo pipefail
CFG=${1:-configs/mid.yaml}
python scripts/00_check_env.py --config "$CFG"
python scripts/01_prepare_data.py --config "$CFG"
python scripts/02_generate.py --config "$CFG"                          # full generation (resumes from the smoke output)
python scripts/03_build_corpus.py --config "$CFG"
python scripts/show_samples.py --config "$CFG"
python scripts/04_extract_features.py --config "$CFG"
python scripts/05_train_detectors.py --config "$CFG"
python scripts/06_eval_detectors.py --config "$CFG"
python scripts/07_audit_estimate.py --config "$CFG"
python scripts/08_prepare_downstream.py --config "$CFG"
