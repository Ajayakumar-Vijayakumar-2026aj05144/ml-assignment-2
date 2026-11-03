"""
Shared feature definitions and preprocessing for the Online Shoppers
Purchasing Intention dataset (UCI #468).

Used by both train_models.py (training) and app.py (inference) so that the
uploaded test CSV is cleaned exactly the same way as the training data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Revenue"

# Heavy right-skewed, non-negative counts / durations / page value -> log1p + scale
SKEWED_NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "PageValues",
]

# Bounded ratios / proximity score -> scale only
BOUNDED_NUMERIC = ["BounceRates", "ExitRates", "SpecialDay"]

# Nominal features (the integer codes in OS/Browser/Region/TrafficType are
# category IDs, not magnitudes, so they are one-hot encoded)
CATEGORICAL = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

BINARY = ["Weekend"]

FEATURES = SKEWED_NUMERIC + BOUNDED_NUMERIC + CATEGORICAL + BINARY

_TRUE_STRINGS = {"true", "1", "yes", "t", "y"}


def _to_binary(series: pd.Series) -> pd.Series:
    """Map bool / 0-1 / 'TRUE'-'FALSE' style values to int 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().isin(_TRUE_STRINGS).astype(int)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast raw columns so training data and uploaded CSVs are identical."""
    df = df.copy()
    for col in BINARY + ([TARGET] if TARGET in df.columns else []):
        df[col] = _to_binary(df[col])
    for col in SKEWED_NUMERIC + BOUNDED_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    for col in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    for col in ["Month", "VisitorType"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_preprocessor() -> ColumnTransformer:
    skewed_pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("skewed", skewed_pipe, SKEWED_NUMERIC),
            ("bounded", StandardScaler(), BOUNDED_NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("binary", "passthrough", BINARY),
        ],
        remainder="drop",
    )
