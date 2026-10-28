CAFA6 Protein Function Prediction Pipeline

Overview

- Implements a modular pipeline for CAFA6: data loading, preprocessing, model training, evaluation, and submission generation.
- Includes a minimal baseline mode that generates a valid submission in low‑resource environments.
- Provides scripts and modules under `src/` and `scripts/` for reproducible runs.

Project Structure

- `src/preprocessing.py`: FASTA parsing, label handling, k‑mer encoding, train/val split.
- `src/model.py`: training (OVR logistic), prediction, Fmax/auPR metrics, model I/O, submission writer.
- `scripts/run_pipeline.py`: end‑to‑end orchestration; supports minimal baseline mode.
- `submissions/submission_final.tsv`: final submission file written by the pipeline.
- `logs/`: run logs and experiment notes.
- `notebooks/notebook_cafa6_pipeline.ipynb`: reproducible notebook skeleton.

Quick Start

1) Install dependencies (already installed in this workspace):
   - `pip install -r requirements.txt`
2) Run the minimal baseline (default):
   - `python scripts/run_pipeline.py`
   - Writes `submissions/submission_final.tsv` with fixed frequent GO terms per protein.

Full Pipeline (Training + Evaluation)

- The script supports a full run that trains a One‑vs‑Rest Logistic Regression on k‑mer features and evaluates Fmax and auPR.
- Due to memory constraints observed locally, the script defaults to a low‑resource baseline. To attempt a full run:
  - Set environment variable `MINIMAL_BASELINE=0`.
  - Optionally reduce resource usage via script arguments/env (e.g., k‑mer size, max sequences, class frequency threshold).

Command Examples

- Minimal baseline (default): `python scripts/run_pipeline.py`
- Full run (potentially heavy): `set MINIMAL_BASELINE=0 && python scripts/run_pipeline.py`

Submission Format

- The pipeline writes `protein_id\tGO_term\tscore` lines in TSV.
- Valid example lines:
  - `A0A024R2T3\tGO:0005515\t0.100000`

Evaluation Metrics

- `src/model.py` includes functions for CAFA metrics:
  - Fmax: thresholds predictions to maximize F‑score.
  - auPR: area under precision‑recall curve.
- Metrics are computed on the validation split during full runs (disabled in minimal baseline).

Scientific Notes

- Minimal baseline: fixed terms (`GO:0005515`, `GO:0016020`, `GO:0005829`) with small priors; chosen as commonly occurring, biologically plausible terms for demonstration only.
- Full pipeline: supports scalable label handling (chunked reading), k‑mer encoding, and OVR logistic training.
- Future improvements: BLAST similarity propagation, transformer embeddings (ProtBERT/ESM), hierarchical GO constraints.

Reproducibility & Logs

- All outputs are deterministic under fixed seeds and parameters.
- Run logs are saved to `logs/`.

Data Requirements

- Place CAFA6 data under `data/Train` and `data/Test` (already present in this workspace):
  - Train: `train_sequences.fasta`, `train_terms.tsv`, `train_taxonomy.tsv`
  - Test: `testsuperset.fasta`, `sample_submission.tsv`

Notebook

- See `notebooks/notebook_cafa6_pipeline.ipynb` for a concise, reproducible walkthrough.

License & Competition Rules

- Respect CAFA/Kaggle rules: no data leakage, only permitted external resources.
- This repository is intended for educational and competition use.
