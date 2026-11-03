"""
Sanity check: reload every saved model from model/*.pkl and re-score it on
test_data.csv WITHOUT fitting. The metrics must equal model/metrics.csv.

Run:  python verify_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from preprocessing import FEATURES, TARGET, clean_dataframe
from train_models import evaluate

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

schema = json.loads((MODEL_DIR / "feature_schema.json").read_text())
saved = pd.read_csv(MODEL_DIR / "metrics.csv", index_col=0)
test = clean_dataframe(pd.read_csv(ROOT / "test_data.csv"))
X, y = test[FEATURES], test[TARGET]

rows = {}
for name, fname in schema["model_files"].items():
    model = joblib.load(MODEL_DIR / fname)
    metrics, y_pred = evaluate(model, X, y)
    rows[name] = metrics
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print(f"{name:<24} TN={tn:<5} FP={fp:<4} FN={fn:<4} TP={tp}")

recomputed = pd.DataFrame(rows).T.round(4)
diff = (recomputed - saved.loc[recomputed.index, recomputed.columns]).abs().to_numpy().max()
print(f"\nMax |recomputed - saved| = {diff:.6f}")
assert diff < 1e-9, "Saved models do not reproduce the training-run metrics!"
print("OK - all 5 saved models reproduce metrics.csv exactly.")

# Feature insight for the README observations
lr = joblib.load(MODEL_DIR / "logistic_regression.pkl")
names = lr.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=names).sort_values()
print("\nLogistic Regression - most negative / positive coefficients:")
print(pd.concat([coefs.head(5), coefs.tail(5)]).round(3).to_string())

dt = joblib.load(MODEL_DIR / "decision_tree.pkl")
imp = pd.Series(dt.named_steps["clf"].feature_importances_,
                index=dt.named_steps["prep"].get_feature_names_out())
print("\nDecision Tree - top feature importances:")
print(imp[imp > 0].sort_values(ascending=False).round(4).to_string())
print(f"\nBaseline (always predict 0) accuracy = {1 - y.mean():.4f}; positives in test = {int(y.sum())}")
print(f"Transformed feature count = {len(names)}")
