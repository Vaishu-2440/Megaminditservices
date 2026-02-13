"""Round-1 coding assignment implementation.

Implements all requested items from the assignment sheets:
- selected-paper-aware objective framing,
- unique proposed algorithm,
- baseline + traditional + proposed comparisons,
- balanced vs. imbalanced evaluation,
- cross-validation with metrics beyond accuracy,
- feature-selection ablation,
- time/space complexity estimation,
- output tables and figures for report inclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
SELECTED_PAPER = (
    "Saito & Rehmsmeier (2015) - Precision-Recall plot is more informative than ROC under "
    "class imbalance"
)


@dataclass(frozen=True)
class EvaluationResult:
    dataset_variant: str
    model_name: str
    strategy: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    mcc: float
    cv_f1_mean: float
    cv_f1_std: float
    train_time_sec: float
    infer_time_sec: float


class ProposedHybridClassifier(BaseEstimator, ClassifierMixin):
    """HybridFS-Stack++: MI-based feature selection + weighted soft voting ensemble."""

    def __init__(self, k_features: int = 15, use_feature_selection: bool = True):
        self.k_features = k_features
        self.use_feature_selection = use_feature_selection
        self.pipeline_: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, slice(0, None))], remainder="drop"
        )

        voting = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
                ("svm", SVC(probability=True, kernel="rbf", C=2.0, random_state=RANDOM_STATE)),
            ],
            voting="soft",
            weights=[2, 2, 3],
            n_jobs=-1,
        )

        steps = [("preprocessor", preprocessor)]
        if self.use_feature_selection:
            steps.append(("select", SelectKBest(score_func=mutual_info_classif, k=self.k_features)))
        steps.append(("model", voting))
        return Pipeline(steps=steps)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X)


def load_datasets() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    full_df = X.copy()
    full_df["target"] = y
    minority = full_df[full_df["target"] == 1].sample(frac=0.25, random_state=RANDOM_STATE)
    majority = full_df[full_df["target"] == 0]
    imbalanced_df = pd.concat([majority, minority], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)

    return {
        "balanced": (X, y),
        "imbalanced": (imbalanced_df.drop(columns=["target"]), imbalanced_df["target"]),
    }


def build_models() -> dict[str, tuple[str, BaseEstimator]]:
    def with_scaler(model):
        return Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])

    return {
        "Traditional-LogReg": (
            "traditional",
            with_scaler(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ),
        "Traditional-SVM-RBF": (
            "traditional",
            with_scaler(SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)),
        ),
        "Traditional-RandomForest": (
            "traditional",
            Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
        ),
        "Proposed-HybridFS-Stack++": (
            "proposed",
            ProposedHybridClassifier(k_features=15, use_feature_selection=True),
        ),
        "Ablation-Hybrid-NoFS": (
            "ablation",
            ProposedHybridClassifier(k_features=15, use_feature_selection=False),
        ),
    }


def estimate_complexity(model_name: str, n_samples: int, n_features: int) -> tuple[str, str]:
    if "LogReg" in model_name:
        return (f"O({n_samples}*{n_features})", f"O({n_features})")
    if "SVM" in model_name:
        return (f"~O({n_samples}^2*{n_features})", f"O({n_samples}^2)")
    if "RandomForest" in model_name:
        return (f"O(trees*{n_samples}*log({n_samples})*{n_features})", "O(trees*nodes)")
    if "Hybrid" in model_name:
        return (
            "O(MI feature selection + LR + RF + SVM training)",
            "O(selected_features + ensemble params)",
        )
    return ("N/A", "N/A")


def evaluate_model(model, X_train, X_test, y_train, y_test, dataset_variant: str, model_name: str, strategy: str) -> EvaluationResult:
    model_for_train = clone(model)
    start_train = perf_counter()
    model_for_train.fit(X_train, y_train)
    train_time = perf_counter() - start_train

    start_infer = perf_counter()
    pred = model_for_train.predict(X_test)
    proba = model_for_train.predict_proba(X_test)[:, 1]
    infer_time = perf_counter() - start_infer

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        n_jobs=-1,
        error_score="raise",
    )

    return EvaluationResult(
        dataset_variant=dataset_variant,
        model_name=model_name,
        strategy=strategy,
        accuracy=accuracy_score(y_test, pred),
        balanced_accuracy=balanced_accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, proba),
        pr_auc=average_precision_score(y_test, proba),
        mcc=matthews_corrcoef(y_test, pred),
        cv_f1_mean=float(cv_scores["test_f1"].mean()),
        cv_f1_std=float(cv_scores["test_f1"].std()),
        train_time_sec=float(train_time),
        infer_time_sec=float(infer_time),
    )


def save_plots(results_df: pd.DataFrame, best_models: dict[str, BaseEstimator], tests: dict[str, tuple[pd.DataFrame, pd.Series]], output_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="model_name", y="f1", hue="dataset_variant")
    plt.title("F1 Comparison: Existing vs Proposed vs Ablation")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "f1_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="model_name", y="pr_auc", hue="dataset_variant")
    plt.title("PR-AUC Comparison (Imbalance-sensitive metric)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "prauc_comparison.png", dpi=160)
    plt.close()

    for variant, model in best_models.items():
        X_test, y_test = tests[variant]
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix ({variant})")
        fig.tight_layout()
        fig.savefig(output_dir / "figures" / f"cm_best_{variant}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    output_dir = Path("outputs")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    datasets = load_datasets()
    models = build_models()
    results: list[EvaluationResult] = []
    complexity_rows: list[dict[str, str]] = []
    best_models: dict[str, BaseEstimator] = {}
    test_partitions: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for variant, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )
        test_partitions[variant] = (X_test, y_test)

        best_f1 = -1.0
        best_fitted = None

        for model_name, (strategy, model) in models.items():
            result = evaluate_model(model, X_train, X_test, y_train, y_test, variant, model_name, strategy)
            results.append(result)

            train_c, space_c = estimate_complexity(model_name, len(X_train), X_train.shape[1])
            complexity_rows.append(
                {
                    "dataset_variant": variant,
                    "model_name": model_name,
                    "train_complexity": train_c,
                    "space_complexity": space_c,
                }
            )

            fitted = clone(model).fit(X_train, y_train)
            if result.f1 > best_f1:
                best_f1 = result.f1
                best_fitted = fitted

        best_models[variant] = best_fitted

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        ["dataset_variant", "f1"], ascending=[True, False]
    )
    complexity_df = pd.DataFrame(complexity_rows).drop_duplicates()

    recommendation_df = (
        results_df.sort_values(["dataset_variant", "f1"], ascending=[True, False])
        .groupby("dataset_variant")
        .head(1)
        .loc[:, ["dataset_variant", "model_name", "strategy", "f1", "pr_auc", "recall", "cv_f1_mean"]]
    )

    results_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    complexity_df.to_csv(output_dir / "complexity_summary.csv", index=False)
    recommendation_df.to_csv(output_dir / "final_recommendations.csv", index=False)
    save_plots(results_df, best_models, test_partitions, output_dir)

    print(f"Selected guiding paper: {SELECTED_PAPER}")
    print("Saved outputs:")
    print("- outputs/metrics_summary.csv")
    print("- outputs/complexity_summary.csv")
    print("- outputs/final_recommendations.csv")
    print("- outputs/figures/f1_comparison.png")
    print("- outputs/figures/prauc_comparison.png")
    print("- outputs/figures/cm_best_*.png")


if __name__ == "__main__":
    main()
