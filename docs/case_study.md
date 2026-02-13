# Case Study Report

## Title
**HybridFS-Stack++: A Feature-Selection Guided Ensemble for Robust Binary Classification Under Class Imbalance**

## 1) Problem Statement and Objectives

### Problem Statement
Many real-world classification tasks (especially healthcare) face class imbalance and noisy features. Traditional single-model approaches can overfit dominant patterns and underperform on minority-sensitive metrics.

### Objectives
1. Build and compare traditional baseline models and one proposed model.
2. Evaluate on both balanced and imbalanced dataset variants.
3. Validate via cross-validation with multiple metrics (not only accuracy).
4. Provide reproducible visual and tabular evidence for draft-paper preparation.

### Research Questions
- RQ1: Does feature-selection guided ensembling improve robustness over traditional models?
- RQ2: Is PR-AUC/F1 stability better under class imbalance compared with traditional baselines?
- RQ3: What is the cost/benefit tradeoff in time-space complexity?

## 2) Dataset and Preprocessing

- Dataset: `sklearn.datasets.load_breast_cancer`
- Samples: 569
- Features: 30 numerical
- Task: Binary classification

### Preprocessing pipeline
1. Stratified split (train/test).
2. Standardization.
3. Balanced variant (original) and imbalanced variant (minority subsampling).
4. Reproducibility via fixed random seed.

## 3) Model Selection and Development

### Existing/Traditional models
- Logistic Regression
- Random Forest
- SVM (RBF)

### Proposed model (new)
**HybridFS-Stack++**
1. Standardize numeric features.
2. Select informative features via mutual information.
3. Train LR + RF + RBF-SVM.
4. Weighted soft voting fusion.

### Ablation (requested “can we develop a new method?” validation)
- `Ablation-Hybrid-NoFS`: same ensemble without feature selection.
- This isolates the contribution of the feature-selection stage.

## 4) Experimental Methodology

- Holdout evaluation (stratified).
- 5-fold stratified cross-validation.
- Metrics:
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - F1
  - ROC-AUC
  - PR-AUC
  - MCC
- Runtime:
  - training time
  - inference time

## 5) Comparative Analysis Plan

The following comparisons are exported in `outputs/metrics_summary.csv`:
1. Traditional models vs proposed model.
2. Proposed model vs ablation model.
3. Balanced vs imbalanced performance.
4. CV stability (`cv_f1_mean`, `cv_f1_std`).

## 6) Balanced vs Unbalanced Discussion

- Balanced dataset indicates general discriminative quality.
- Imbalanced dataset focuses on minority-sensitive behavior.
- Recommendation should prioritize PR-AUC/F1/Recall, not just accuracy.

## 7) Time and Space Complexity

Saved in `outputs/complexity_summary.csv`.

High-level:
- Logistic Regression: approximately linear with samples/features.
- SVM-RBF: quadratic-ish in samples.
- Random Forest: trees × split-search cost.
- HybridFS-Stack++: feature selection + multiple model costs (higher but often better robustness).

## 8) Visualizations and Insights

Generated:
- `f1_comparison.png`
- `prauc_comparison.png`
- `cm_best_balanced.png`
- `cm_best_imbalanced.png`

## 9) Recommendations

`outputs/final_recommendations.csv` stores best model per dataset variant with F1/PR-AUC/Recall/CV-F1. This supports decision-making for publication and implementation.

## 10) Video Explanation Checklist (up to 15 minutes)

1. Novelty and originality of HybridFS-Stack++.
2. Literature support and gap.
3. Coding implementation details.
4. Quality of visualizations and comparison.
5. End-to-end structure of document.
6. Data handling, preprocessing, and model development.

## 11) Submission Package

- Source code (`src/main.py`)
- Case-study report (`docs/case_study.md` + generated outputs)
- Reference list (`docs/reference_list.md`)
- Journal targeting (`docs/journal_plan.md`)
- Requirement matrix (`docs/assignment_requirements_checklist.md`)
