"""
Streamlit app - ML Assignment 2
Online Shoppers Purchasing Intention: evaluate 5 pre-trained classifiers
on an uploaded test CSV (no retraining - models are loaded from model/*.pkl).

Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocessing import FEATURES, TARGET, clean_dataframe

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
      .hero {padding: 1.1rem 1.4rem; border-radius: 12px;
             background: linear-gradient(90deg, #0f4c75 0%, #3282b8 100%); color: white;}
      .hero h2 {margin: 0; color: white;}
      .hero p {margin: .3rem 0 0 0; opacity: .9;}
      div[data-testid="stMetric"] {background: #f4f8fb; border: 1px solid #dbe7f0;
             border-radius: 10px; padding: .6rem .8rem;}
    </style>
    <div class="hero">
      <h2>🛒 Online Shopper Purchase Intention - Model Evaluation</h2>
      <p>Upload the held-out test CSV, pick a classifier and inspect how the
      <b>pre-trained</b> model performs. Models are never retrained here.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading saved models ...")
def load_models():
    with open(MODEL_DIR / "feature_schema.json") as f:
        schema = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in schema["model_files"].items()}
    return models, schema


@st.cache_data
def load_training_artifacts():
    train_metrics = pd.read_csv(MODEL_DIR / "metrics.csv", index_col=0)
    with open(MODEL_DIR / "best_params.json") as f:
        params = json.load(f)
    return train_metrics, params


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:  # only one class present in the uploaded labels
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner="Scoring all models ...")
def score_all(_models, df_clean: pd.DataFrame):
    X, y = df_clean[FEATURES], df_clean[TARGET].to_numpy()
    out = {}
    for name, model in _models.items():
        y_pred = model.predict(X)
        y_score = positive_scores(model, X)
        out[name] = {"y_pred": y_pred, "y_score": y_score, "metrics": compute_metrics(y, y_pred, y_score)}
    return out


models, schema = load_models()
train_metrics, best_params = load_training_artifacts()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV (features + 'Revenue')", type=["csv"])
    use_sample = st.toggle("No file? Use bundled test_data.csv", value=False, disabled=uploaded is not None)
    with open(SAMPLE_CSV, "rb") as f:
        st.download_button("⬇️ Download sample test_data.csv", f, file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))

    st.caption(
        f"Required columns: {len(FEATURES)} features + `{TARGET}` (0/1 or True/False). "
        "Extra columns are ignored."
    )

# ---------------------------------------------------------------- data
if uploaded is not None:
    raw_df, source = pd.read_csv(uploaded), f"uploaded file `{uploaded.name}`"
elif use_sample:
    raw_df, source = pd.read_csv(SAMPLE_CSV), "bundled `test_data.csv`"
else:
    st.info("Upload a test CSV from the sidebar (or enable the bundled sample) to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in raw_df.columns]
if missing:
    st.error(f"The CSV is missing required column(s): {', '.join(missing)}")
    st.stop()

try:
    df = clean_dataframe(raw_df[FEATURES + [TARGET]])
except (ValueError, TypeError) as exc:
    st.error(f"Could not parse the CSV - check the column types. Details: {exc}")
    st.stop()

if df.isna().any().any():
    st.error("The CSV contains missing values; please supply complete rows.")
    st.stop()

results = score_all(models, df)
y_true = df[TARGET].to_numpy()

c1, c2, c3 = st.columns(3)
c1.metric("Rows evaluated", f"{len(df):,}")
c2.metric("Purchases (Revenue = 1)", f"{int(y_true.sum()):,}")
c3.metric("Positive rate", f"{y_true.mean():.1%}")
st.caption(f"Data source: {source}")

tab_model, tab_compare, tab_pred, tab_info = st.tabs(
    ["📊 Selected model", "🏆 Compare all models", "🔎 Predictions", "ℹ️ Training details"]
)

# ---------------------------------------------------------------- tab 1
with tab_model:
    res = results[model_name]
    st.subheader(f"{model_name} - evaluation metrics")
    cols = st.columns(6)
    for col, m in zip(cols, METRIC_NAMES):
        col.metric(m, "n/a" if np.isnan(res["metrics"][m]) else f"{res['metrics'][m]:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, res["y_pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"], yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    with right:
        st.markdown("**ROC curve**")
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, res["y_score"])
            fig, ax = plt.subplots(figsize=(4.6, 3.8))
            ax.plot(fpr, tpr, color="#3282b8", lw=2, label=f"AUC = {res['metrics']['AUC']:.4f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("ROC/AUC needs both classes in the uploaded labels.")

    st.markdown("**Classification report**")
    report = classification_report(
        y_true, res["y_pred"], labels=[0, 1], target_names=["No purchase (0)", "Purchase (1)"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader("All five models on this test data")
    comp = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T[METRIC_NAMES]
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"), width="stretch")

    best = comp["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

    long = comp.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="Blues_d", ax=ax)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- tab 3
with tab_pred:
    st.subheader(f"Row-level predictions - {model_name}")
    pred_df = raw_df[FEATURES].copy()
    pred_df["Actual_Revenue"] = y_true
    pred_df["Predicted_Revenue"] = results[model_name]["y_pred"]
    pred_df["Score"] = np.round(results[model_name]["y_score"], 4)
    pred_df["Correct"] = pred_df["Actual_Revenue"] == pred_df["Predicted_Revenue"]
    only_wrong = st.checkbox("Show misclassified rows only")
    view = pred_df[~pred_df["Correct"]] if only_wrong else pred_df
    st.dataframe(view, width="stretch", height=380)
    st.download_button(
        "⬇️ Download predictions CSV",
        pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------- tab 4
with tab_info:
    st.subheader("How the models were built")
    st.markdown(
        f"""
- **Dataset:** UCI Online Shoppers Purchasing Intention (#468) - 12,330 sessions, 17 features; 125 duplicates removed.
- **Split:** stratified 80/20 (random_state = {schema['random_state']}); the 20% hold-out is `test_data.csv`.
- **Preprocessing (inside each saved pipeline):** log1p + standardisation on skewed counts/durations/PageValues,
  standardisation on BounceRates/ExitRates/SpecialDay, one-hot encoding of Month, VisitorType, OperatingSystems,
  Browser, Region, TrafficType; Weekend as 0/1.
- **Tuning:** GridSearchCV, 5-fold stratified CV on the training set only, scoring = F1.
- **scikit-learn version used for training:** {schema['sklearn_version']}
"""
    )
    st.markdown("**Selected hyperparameters**")
    st.dataframe(
        pd.DataFrame(
            {n: {"Best parameters": json.dumps(p["params"]), "CV F1 (train)": p["cv_f1"]} for n, p in best_params.items()}
        ).T,
        width="stretch",
    )
    st.markdown("**Reference metrics on the original hold-out set (from training run)**")
    st.dataframe(train_metrics.style.format("{:.4f}"), width="stretch")
