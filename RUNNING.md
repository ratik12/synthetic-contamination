# Running on Colab

Step-by-step instructions for running the pipeline on Google Colab (T4 GPU).

## 1. Open the notebook

Open `notebooks/run_on_colab.ipynb` directly in Colab:
- Direct link: `https://colab.research.google.com/github/ratik12/synthetic-contamination/blob/main/notebooks/run_on_colab.ipynb`
- Or go to [colab.research.google.com](https://colab.research.google.com), select the GitHub tab, enter `ratik12/synthetic-contamination`, and pick `run_on_colab.ipynb`.

Make sure GPU is enabled: `Runtime > Change runtime type > T4 GPU`.

## 2. Execution order

The notebook runs the pipeline cells sequentially:

1. **Check GPU:** Runs `nvidia-smi` and checks CUDA is available.
2. **Setup:** Clones repo, installs `requirements.txt`.
3. **Drive mount & symlinks:** Mounts Drive to `/content/drive/MyDrive/synthetic_contamination_backup` and symlinks `data/` and `results/` so shard progress is preserved across disconnects.
4. **00_check_env:** Verifies dependencies and Hugging Face model IDs.
5. **01_prepare_data:** Downloads Wikipedia dump (`20220301.simple`), cleans text, dedups, and writes split metadata (S1 to S5).
6. **Smoke test (02_generate --extra limit=16):** Generates 16 passages per generator and prints 4 samples with `show_samples.py`. Check the text before continuing: confirm no refusals, chat preambles ("Sure, here is..."), or markdown headings.
7. **Full pipeline:** Runs `bash run_all.sh configs/mid.yaml`. This resumes from the smoke output and runs full generation (02), corpus merge and leakage checks (03), feature extraction (04), detector training (05), evaluation (06), audit estimation (07), and downstream dataset prep (08).
8. **09_make_report_tables:** Writes LaTeX tables to `report/gen/` and figures to `report/fig/`.
9. **Display results:** Prints `results/mid_today/detection_metrics.csv` and `estimator_summary.csv`.

## 3. Expected runtimes (estimates until measured)

Times projected on a free Colab T4 GPU:

- `00_check_env`: ~30s
- `01_prepare_data`: ~3-5 min (dump download and filtering)
- `02_generate` (smoke test, limit=16): ~2 min
- `02_generate` (full run): ~45-75 min (Qwen2.5-1.5B ~15-20 min, Phi-3.5-mini ~30-50 min; resumable)
- `03_build_corpus`: ~1-2 min
- `04_extract_features`: ~15-20 min (GPT-2 stats, Binoculars, stylometric features)
- `05_train_detectors`: ~5-10 min (LR, GBDT, TF-IDF, DistilBERT seen/unseen)
- `06_eval_detectors`: ~2-3 min (AUROC, AUPRC, TPR@1% FPR with bootstrap)
- `07_audit_estimate`: ~5-10 min (CC, PCC, ACC, KDE MLE across alphas and draws)
- `08_prepare_downstream`: ~1-2 min (D arms and matching H controls)
- `09_make_report_tables`: ~1 min
- **Total estimated time:** ~1.5 to 2.5 hours

## 4. Fallback if Wikipedia 20220301 loader fails

The Wikipedia 20220301 snapshot uses an older loading script that requires `datasets==2.21.0`.

If `01_prepare_data.py` fails to download or parse the dump due to Hugging Face changes, you can enable the fallback dump in `configs/mid.yaml`:
```yaml
data:
  allow_fallback_dump: true
```
This loads `wikimedia/wikipedia` snapshot `20231101.simple`.

**Deviation note:** The November 2023 dump was created after ChatGPT was released (November 2022). It may contain unknown amounts of synthetic web text. If `allow_fallback_dump: true` is used, note it as a design deviation in the report.
