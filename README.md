# Developer Round-1 Assignment (Complete Delivery)

This repository is structured to satisfy **all requirements visible in the three assignment pages**:
- unique solution + implementation,
- comprehensive case-study document sections,
- comparative analysis vs existing algorithms,
- cross-validation with metrics beyond accuracy,
- balanced/unbalanced dataset evaluation,
- visualizations,
- time/space complexity discussion,
- reference finalization and journal targeting.

## Selected Paper Used as Technical Anchor

- **Saito, T., & Rehmsmeier, M. (2015)**: PR curves are more informative than ROC in imbalanced scenarios.
- DOI: 10.1371/journal.pone.0118432

This is why `PR-AUC` is included in the pipeline and report.

## Repository Layout

- `src/main.py`: Full coding implementation (traditional baselines + proposed method + ablation).
- `docs/case_study.md`: Case-study report draft content.
- `docs/reference_list.md`: 25+ DOI-backed references.
- `docs/journal_plan.md`: 5-journal target list with Q2/Q3 priority and publication strategy.
- `docs/assignment_requirements_checklist.md`: one-to-one requirement completion matrix.
- `requirements.txt`: required packages.
- `outputs/`: generated tables/plots after execution.

## Proposed Algorithm (Named)

**HybridFS-Stack++**

1. Standardization
2. Mutual-information feature selection
3. Heterogeneous ensemble (LR + RF + RBF-SVM)
4. Weighted soft voting
5. Ablation against variant without feature selection

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Output Files Produced

- `outputs/metrics_summary.csv`
- `outputs/complexity_summary.csv`
- `outputs/final_recommendations.csv`
- `outputs/figures/f1_comparison.png`
- `outputs/figures/prauc_comparison.png`
- `outputs/figures/cm_best_*.png`

These outputs directly support the required report sections and video explanation points.
