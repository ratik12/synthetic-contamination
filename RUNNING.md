# Running the Synthetic Data Contamination Pipeline on Google Colab

This guide describes how to launch and execute the full experimental pipeline on a remote GPU (Google Colab T4) using `notebooks/run_on_colab.ipynb`.

---

## 1. Opening the Notebook in Colab

1. Open your browser and navigate to Google Colab:
   - Direct link: `https://colab.research.google.com/github/ratik12/synthetic-contamination/blob/main/notebooks/run_on_colab.ipynb`
   - Alternatively, open [Google Colab](https://colab.research.google.com/), select the **GitHub** tab, enter `ratik12/synthetic-contamination`, and select `notebooks/run_on_colab.ipynb`.
2. **Select GPU Hardware Accelerator:**
   - Go to the menu: `Runtime` -> `Change runtime type`.
   - Under **Hardware accelerator**, select **T4 GPU**.
   - Click **Save**.

---

## 2. Execution Flow

The notebook contains individual cells that correspond to each stage of the research pipeline:

1. **GPU Check (`nvidia-smi` & PyTorch CUDA check):** Confirms CUDA T4 is available.
2. **Clone & Install:** Clones the repository and installs `requirements.txt` (including pinned `datasets==2.21.0` and `matplotlib`).
3. **Mount Drive & Symlink:** Mounts Google Drive (`/content/drive/MyDrive/synthetic_contamination_backup`) and symlinks `data/` and `results/` so that if the Colab session disconnects, all shards and outputs are preserved.
4. **Environment Check:** Runs `scripts/00_check_env.py` to test HuggingFace model accessibility.
5. **Data Preparation:** Runs `scripts/01_prepare_data.py` to sample Wikipedia, clean, dedup, and partition S1–S5 splits.
6. **Smoke Test:** Runs `scripts/02_generate.py --extra limit=16` and prints 4 raw samples via `scripts/show_samples.py --raw --n 4`.
   > **Critical Check:** Inspect the smoke test samples by eye! Verify there are no AI chat preambles ("Sure, here is..."), markdown headings, or refusals.
7. **Full Execution:** Executes `bash run_all.sh configs/mid.yaml`, which resumes generation from the smoke test and proceeds through feature extraction, detector training, evaluation, audit estimation, and downstream dataset preparation.
8. **Report Generation:** Runs `scripts/09_make_report_tables.py --extra out=report` to generate LaTeX tables and summary figures.
9. **Display Results:** Reads and displays `detection_metrics.csv` and `estimator_summary.csv`.

---

## 3. Estimated Runtime per Stage

> *Note: All runtimes below are estimates measured/projected on a Google Colab free-tier T4 GPU (16 GB VRAM) until formally measured during your runs.*

| Stage | Script / Task | Estimated Runtime | Notes |
|---|---|---|---|
| **00** | `00_check_env.py` | ~30 seconds | Checks environment, CUDA, and HF weights connectivity |
| **01** | `01_prepare_data.py` | ~3–5 minutes | Downloads `20220301.simple`, cleans, dedups, assigns S1–S5 splits |
| **02 (Smoke)** | `02_generate.py --extra limit=16` | ~2 minutes | Generates 16 samples per generator |
| **02 (Full)** | `02_generate.py` | ~45–75 minutes | Generates synthetic twins with Qwen2.5-1.5B and Phi-3.5-mini in fp16; resumable |
| **03** | `03_build_corpus.py` | ~1–2 minutes | Merges corpus, executes 4 leakage checks, computes length stats |
| **04** | `04_extract_features.py` | ~15–20 minutes | GPT-2 metrics + Fast-DetectGPT + Binoculars + 12 stylometric features |
| **05** | `05_train_detectors.py` | ~5–10 minutes | Trains LR, GBDT, TF-IDF LR, and DistilBERT across seen/unseen setups |
| **06** | `06_eval_detectors.py` | ~2–3 minutes | AUROC, AUPRC, TPR@1% FPR with 200 bootstrap resamples |
| **07** | `07_audit_estimate.py` | ~5–10 minutes | Ratio estimation across 5 alphas, 50 draws, KDE/CC/PCC/ACC/MLE |
| **08** | `08_prepare_downstream.py` | ~1–2 minutes | Assembles nested contaminated D arms and matching H controls |
| **09** | `09_make_report_tables.py` | ~1 minute | Compiles LaTeX tables and figures |
| **Total** | Full Pipeline | **~1.5 – 2 hours** | |

---

## 4. Troubleshooting: Wikipedia 20220301 Loader

- **Dataset Pinning:** The project requires `datasets==2.21.0` because the pre-ChatGPT Wikipedia snapshot (`legacy-datasets/wikipedia`, config `20220301.simple`) relies on dataset loading scripts with `trust_remote_code=True`.
- **If the 20220301 Loader Fails:**
  If network or Hugging Face repository issues prevent loading `20220301.simple`, open `configs/mid.yaml` and set:
  ```yaml
  data:
    allow_fallback_dump: true
  ```
  This directs the pipeline to load `wikimedia/wikipedia` (config `20231101.simple`).
- **Research Deviation Warning:**
  > **IMPORTANT:** The November 2023 dump was created **after** the release of ChatGPT (November 2022). Consequently, the "human" text itself may contain unknown amounts of synthetic web text. If `allow_fallback_dump: true` is used, you **must explicitly declare this as a methodological deviation** in all capstone presentations, logs, and reports.
