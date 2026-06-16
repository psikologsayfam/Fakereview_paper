# Leakage-Aware Review-Event Benchmark Audit

This repository contains the code used to reproduce the experiments and manuscript
figures for the study:

**Auditing duplicate leakage and photo-linked target construction in a food
delivery review-event benchmark**

The code audits the translated Yogiyo review-event dataset under two evaluation
protocols:

- a baseline stratified split
- a duplicate-aware group split based on canonical review strings

It also compares text-only, behavior-only, no-picture, raw-picture, and
picture-context feature sets with the same downstream classifier.

## Repository layout

- `submission_pipeline.py` runs the full experiment pipeline and exports tables,
  metrics, confidence intervals, and analysis notes.
- `make_manuscript_figures.py` regenerates the manuscript-ready figures from the
  exported outputs.
- `requirements.txt` lists the Python dependencies used in the rerun.
- `data/` is the expected location of the input CSV and is not versioned here.
- `outputs/` and `_paper_extract/` are generated when the scripts are executed.

## Data source

The code expects the translated Yogiyo review-event CSV:

- Data in Brief article: https://doi.org/10.1016/j.dib.2024.110598
- Mendeley Data record: https://doi.org/10.17632/rnyrpzyw3h.2

Place `Reviews Translated into English.csv` inside `data/` before running the
pipeline.

## Quick start

Create a virtual environment and install the requirements:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the full benchmark audit:

```powershell
python submission_pipeline.py
```

Regenerate the manuscript figures from the exported results:

```powershell
python make_manuscript_figures.py
```

## Optional command-line arguments

You can override the default paths if needed:

```powershell
python submission_pipeline.py `
  --project-dir . `
  --output-dir outputs `
  --data-dir data `
  --csv-name "Reviews Translated into English.csv"
```

To skip creation of the zip archive:

```powershell
python submission_pipeline.py --no-zip
```

## Reproducibility notes

- The positive class is fixed as `BiasFree = 1`.
- Duplicate-aware splitting is defined on canonical review strings.
- The Picture-Context Score is learned on the training partition only.
- The main experiment matrix uses the same logistic regression classifier across
  feature families and split protocols.
- A linear SVM robustness check is included to test whether the same qualitative
  conclusion holds under an alternative classifier.
