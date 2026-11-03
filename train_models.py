"""
ML Assignment 2 - Training script
Dataset : Online Shoppers Purchasing Intention (UCI #468)
Task    : Binary classification - will the session end in a purchase (Revenue)?

Trains 5 classifiers on the SAME train split, tunes hyperparameters with
5-fold stratified CV on the training data only, evaluates each tuned model
once on the held-out test split and saves everything the Streamlit app needs.

Run:  python train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import FEATURES, TARGET, build_preprocessor, clean_dataframe

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

# display name -> (file stem, estimator, param grid for the "clf" step)
MODEL_SPECS = {
    "Logistic Regression": (
        "logistic_regression",
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        {
            "clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "Decision Tree": (
        "decision_tree",
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
            "clf__min_samples_leaf": [1, 5, 10, 20, 50, 100, 200, 300, 500],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "KNN": (
        "knn",
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    ),
    "Naive Bayes": (
        "naive_bayes",
        GaussianNB(),
        {"clf__var_smoothing": np.logspace(-12, 1, 14).tolist()},
    ),
    "Support Vector Machine": (
        "svm",
        # probability=False: AUC is computed from decision_function scores,
        # avoiding the extra internal CV of Platt scaling.
        SVC(kernel="rbf", random_state=RANDOM_STATE),
        {
            "clf__C": [0.1, 0.3, 1, 3, 10],
            "clf__gamma": ["scale", 0.03, 0.01, 0.003, 0.001, 0.0003],
            "clf__class_weight": [None, "balanced"],
        },
    ),
}


def positive_scores(model, X):
    """Continuous score for the positive class (needed for AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # ---------------- 1. Load & clean ----------------
    df = pd.read_csv(RAW_CSV)
    print(f"Raw shape: {df.shape}")
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {n_dup} exact duplicate rows -> {df.shape}")
    df = clean_dataframe(df)

    X = df[FEATURES]
    y = df[TARGET]
    print(f"Class balance (Revenue=1): {y.mean():.4f}")

    # ---------------- 2. Stratified train/test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved held-out test data -> {TEST_CSV.name}")

    # ---------------- 3. Tune + fit each model on train only ----------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, best_params, conf_mats, fitted = [], {}, {}, {}

    for name, (stem, estimator, grid) in MODEL_SPECS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        model = search.best_estimator_
        metrics, y_pred = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **metrics})
        best_params[name] = {
            "params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1": round(float(search.best_score_), 4),
            "search_seconds": round(elapsed, 1),
        }
        conf_mats[name] = confusion_matrix(y_test, y_pred)
        fitted[name] = model

        joblib.dump(model, MODEL_DIR / f"{stem}.pkl")
        print(
            f"[{name}] best={best_params[name]['params']} cv_f1={search.best_score_:.4f} "
            f"({elapsed:.1f}s) | test: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    # ---------------- 4. Save artifacts ----------------
    results = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    results.to_csv(MODEL_DIR / "metrics.csv")
    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    with open(MODEL_DIR / "feature_schema.json", "w") as f:
        json.dump(
            {
                "target": TARGET,
                "features": FEATURES,
                "model_files": {n: f"{s}.pkl" for n, (s, _, _) in MODEL_SPECS.items()},
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "sklearn_version": sklearn.__version__,
            },
            f,
            indent=2,
        )

    # Confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    for ax, (name, cm) in zip(axes, conf_mats.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No (0)", "Yes (1)"], yticklabels=["No (0)", "Yes (1)"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, model in fitted.items():
        RocCurveDisplay.from_predictions(y_test, positive_scores(model, X_test), name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC curves - held-out test set")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)

    print("\n=== Test-set comparison ===")
    print(results.to_string())


if __name__ == "__main__":
    main()
