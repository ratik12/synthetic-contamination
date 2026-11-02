# Synthetic-data contamination: pipeline up to downstream training

Stages 01-08 of the research plan, scaled to finish in one day on a free Colab/Kaggle T4. **Downstream training (SmolLM2) is not included**; stage 08 only builds the training sets.

## What was and was not verified

Verified here (no GPU, no internet): 64 offline tests, `python -m pytest tests -q`. They cover text cleaning, splits, twin assignment, leakage checks (which fail on empty input), estimators on simulated scores with known alpha, LM statistics checked by hand against a manual computation and for padding invariance, every detector (DistilBERT on a tiny random model), the Hugging Face generation code path on tiny random models, downstream-arm construction, and stages 01-08 end to end with a mock generator.

**Not verified** (needs your Colab session; the first run is the test):
- Loading `wikipedia/20220301.simple` (`datasets` is not installed in the build sandbox).
- Real Qwen2.5 / Phi-3.5 generation: output quality, speed, fp16 behaviour.
- Real GPT-2 / DistilBERT downloads.
- Every runtime estimate below is a guess, not a measurement.

Mock and tiny-model runs print `MOCK / TEST RUN` and tag records `backend: mock`. Stage 03 refuses mock records unless `mock: true`. **Never quote a number from a mock run.**

## Deviations from the research plan (state these in the report)

| Plan | Here | Why |
|---|---|---|
| 4 generators | 2: Qwen2.5-1.5B-Instruct, Phi-3.5-mini-instruct | Llama-3.2 is gated; Mistral-7B too slow for today |
| English Wikipedia 1 Mar 2022 | `20220301.simple` (same date, Simple English) | download and speed; text is simpler than normal Wikipedia, which makes detection easier or harder in ways you have not measured |
| Binoculars with a 7B pair | implemented with a SmolLM2-360M base/instruct pair (not one of the generators) | memory; a weaker variant |
| Detector fusion | not implemented | time |
| XGBoost | sklearn `HistGradientBoostingClassifier` | no extra dependency |
| Fast-DetectGPT with sampling + scoring model pair | analytic form, single model (GPT-2) | time; a weaker variant |
| POS features | not implemented | time |
| Llama-scale audit | 200-document audit sets, 50 draws per alpha, 5 alpha levels | time |
| Leave-one-generator-out | with 2 generators this is train-on-one, test-on-the-other | the "unseen" claim rests on a single pair |

## Splits (articles are assigned before anything is generated)

S1 600 detector training - S2 200 calibration - S3 300 test / audit pool - S4 1000 downstream pool - S5 300 held-out human evaluation text.
Twins: S1, S2 one per article (generators alternate); S3 one per article per generator; S4 first 500 articles only; S5 none. Prompts contain only the article title, never human text.

Known limit of the audit design: every audit set is drawn from the same 300 S3 articles, so the 50 draws per alpha are strongly overlapping. Coverage and RMSE estimates are optimistic for that reason. Say so.

## Run it (Colab, T4)

1. Runtime > Change runtime type > T4 GPU. Upload this folder (zip) and `unzip` it.
2. `pip install -r requirements.txt`. Optional: `import os; os.environ["HF_TOKEN"]="hf_..."` (the default models are not gated; a token only avoids rate limits).
3. `python scripts/00_check_env.py --config configs/mid.yaml`. Fix every FAIL before going on.
4. `python scripts/01_prepare_data.py --config configs/mid.yaml`. If it says too few eligible articles, raise `data.pool_size`. If the dump will not load, either fix it or set `allow_fallback_dump: true`, which uses a Nov 2023 dump and must be reported as a deviation.
5. **Smoke test:** `python scripts/02_generate.py --config configs/mid.yaml --extra limit=16` then `python scripts/show_samples.py --config configs/mid.yaml --raw --n 4`. Read the text. Look for chat preambles, markdown, refusals, truncation, wrong length. Note the seconds it took and multiply to estimate the full run.
6. `bash run_all.sh` (it repeats 01 harmlessly; it resumes generation from the smoke output). If Colab disconnects, rerun it: each stage resumes or overwrites cleanly.
7. Read `data/mid_today/quality_report.json` and the stage-03 printout:
   - `LENGTH CONFOUND`: detectors can win on length alone.
   - `ends_with_terminal_punct` differing between human and synthetic is a shortcut feature.
   - Large drops from refusals or copying mean something is wrong with the prompt.
8. Copy `data/` and `results/` and `EXPERIMENT_LOG.md` to Google Drive before the session ends.

To use one generator per session (memory): `--extra only=qwen2.5-1.5b`, then `only=phi-3.5-mini`.

## Feeding the report

After stages 01-08: `python scripts/09_make_report_tables.py --config configs/mid.yaml --extra out=report`. Copy `report/gen/` and `report/fig/` next to `main.tex` in Overleaf; every red PENDING box in the report is then replaced by a real table or figure. The script refuses a config with `mock: true`. Do not submit while any red text remains.

## Time (estimates, unmeasured)

Setup and checks 10-15 min. Stage 01 5-10 min. Stage 02: Qwen about 6 min and Phi-3.5 about 15 min for about 950 passages each on a T4 if batching works as expected; if the smoke test says otherwise, believe the smoke test. Stages 03-04 about 5 min. Stage 05 about 5-10 min (DistilBERT, 3 conditions). Stage 06 under 1 min. Stage 07 about 5 min on CPU (measured about 4 min on this sandbox at full audit settings). Stage 08 under 1 min. Total roughly 1.5-2.5 hours if nothing breaks.

## Outputs

| File | Use |
|---|---|
| `data/mid_today/corpus.jsonl`, `quality_report.json` | dataset description, quality table |
| `results/mid_today/detection_metrics.csv` | RQ1: AUROC with 95% CI, AUPRC, TPR at 1% FPR, per detector x seen/unseen generator |
| `results/mid_today/estimator_summary.csv` | RQ2/RQ3: bias, MAE, RMSE, CI coverage for CC, PCC, ACC, MLE by alpha |
| `data/mid_today/downstream/*.jsonl`, `manifest.json` | D0, D5, D15, D30, D50 and size-matched H95, H85, H70, H50; S5 evaluation text |
| `report/gen/*.tex`, `report/fig/*.pdf` | written by stage 09 for the LaTeX report (see below) |
| `EXPERIMENT_LOG.md` | one line per stage run: config hash, status |

## Design notes you must understand before defending this

- D_f vs H_(100-f): H is exactly the human part of D_f. That isolates adding synthetic documents to a smaller human set. It does not match training steps; a step-matched control is not included.
- The estimators use S2 for calibration. In an unseen-generator condition the calibration set excludes that generator's twins and the audit sets contain only that generator. ACC and MLE intervals resample the audit and calibration sets; CC and PCC intervals resample the audit set only. The ACC threshold is fixed after calibration.
- PCC assumes the calibration class balance (50/50). Expect it to be biased at low and high alpha.
- ACC returns NaN when TPR - FPR < 0.05; NaNs are counted in `n_nan`.
- A mixture MLE with KDE densities is only as good as the score densities; check it against ACC.
- With 200 documents per audit set, expect interval widths of several percentage points even for a good detector.

## Layout

`configs/mid.yaml` - one config, hashed into the log. `src/` - logic (all pure functions are unit-tested). `scripts/` - one file per stage. `tests/` - offline tests.
