"""Round-1 coding assignment implementation.

This script implements:
1. Baseline models and one proposed model.
2. Cross-validation and train/test evaluation.
3. Comparison on balanced and imbalanced datasets.
4. Result visualizations and exportable tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42


@dataclass(frozen=True)
class EvaluationResult:
    dataset_variant: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    train_time_sec: float


class ProposedHybridClassifier(BaseEstimator, ClassifierMixin):
    """Proposed algorithm: MI-based feature selection + weighted soft-voting stack.

    Steps:
    1) Standardize all numeric features.
    2) Select top-k informative features by mutual information.
    3) Fit three diverse learners and combine using weighted soft voting.
    """

    def __init__(self, k_features: int = 15):
        self.k_features = k_features
        self.pipeline_: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, slice(0, None))],
            remainder="drop",
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

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("select", SelectKBest(score_func=mutual_info_classif, k=self.k_features)),
                ("model", voting),
            ]
        )

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

    # Create an intentionally imbalanced version to test robustness.
    full_df = X.copy()
    full_df["target"] = y
    minority = full_df[full_df["target"] == 1].sample(frac=0.25, random_state=RANDOM_STATE)
    majority = full_df[full_df["target"] == 0]
    imbalanced_df = pd.concat([majority, minority], axis=0).sample(frac=1.0, random_state=RANDOM_STATE)

    return {
        "balanced": (X, y),
        "imbalanced": (imbalanced_df.drop(columns=["target"]), imbalanced_df["target"]),
    }


def build_baseline_models() -> dict[str, Pipeline]:
    def pipeline_with_scaler(model):
        return Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])

    return {
        "LogisticRegression": pipeline_with_scaler(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "SVM-RBF": pipeline_with_scaler(
            SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE)
        ),
        "RandomForest": Pipeline(
            steps=[("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        ),
        "Proposed-HybridFS-Stack": ProposedHybridClassifier(k_features=15),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test, dataset_variant: str, model_name: str) -> EvaluationResult:
    start = perf_counter()
    model.fit(X_train, y_train)
    train_time = perf_counter() - start

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        error_score="raise",
    )

    return EvaluationResult(
        dataset_variant=dataset_variant,
        model_name=model_name,
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, proba),
        cv_accuracy_mean=float(cv_scores["test_score"].mean()),
        cv_accuracy_std=float(cv_scores["test_score"].std()),
        train_time_sec=float(train_time),
    )


def save_comparison_plot(results_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="model_name", y="f1", hue="dataset_variant")
    plt.title("F1 Score Comparison Across Models and Dataset Variants")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "f1_comparison.png", dpi=160)
    plt.close()


def save_confusion_matrix(best_model, X_test, y_test, output_dir: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix ({name})")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / f"cm_{name}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    datasets = load_datasets()
    models = build_baseline_models()

    all_results: list[EvaluationResult] = []

    for variant, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
        )

        best_model_name = ""
        best_model_instance = None
        best_f1 = -1.0

        for model_name, model in models.items():
            result = evaluate_model(model, X_train, X_test, y_train, y_test, variant, model_name)
            all_results.append(result)

            if result.f1 > best_f1:
                best_f1 = result.f1
                best_model_name = model_name
                best_model_instance = model

        save_confusion_matrix(best_model_instance, X_test, y_test, output_dir, f"{variant}_{best_model_name}")

    results_df = pd.DataFrame([r.__dict__ for r in all_results]).sort_values(
        by=["dataset_variant", "f1"], ascending=[True, False]
    )
    results_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_comparison_plot(results_df, output_dir)

    print("Saved outputs:")
    print("- outputs/metrics_summary.csv")
    print("- outputs/figures/f1_comparison.png")
    print("- outputs/figures/cm_*.png")


if __name__ == "__main__":
    main()
