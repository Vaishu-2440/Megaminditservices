# Developer Round 1 - Coding and Data Analysis Submission

This repository is a complete, reproducible implementation for the assignment shown in your screenshots.
It includes:

- A **literature-driven problem framing**.
- A **proposed algorithm** with clear steps and architecture.
- **Baseline vs proposed model comparison**.
- **Cross-validation** and metrics on balanced and imbalanced datasets.
- **Visualizations** and exportable results.

## Project Structure

- `src/main.py` - end-to-end pipeline (data prep, modeling, evaluation, visualizations).
- `docs/case_study.md` - assignment-ready case study report content.
- `docs/reference_list.md` - reference list with journals/papers and DOIs.
- `requirements.txt` - Python dependencies.
- `outputs/` - generated metrics and figures.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Generated Artifacts

After running:

- `outputs/metrics_summary.csv` - comparative metrics table.
- `outputs/figures/f1_comparison.png` - model-wise F1 visualization.
- `outputs/figures/cm_*.png` - confusion matrix for best model per dataset type.

## Proposed Algorithm Name

**HybridFS-Stack (HFS-Stack)**

1. Standardize numeric features.
2. Apply mutual-information feature selection.
3. Train heterogeneous models (LR + RF + RBF-SVM).
4. Weighted soft voting for final prediction.

This aligns with the requirement to develop a unique or improved method based on existing research.
