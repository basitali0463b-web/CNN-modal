"""Shared cleaning + input-building helpers (no Streamlit dependency, so they are unit-testable)."""
import numpy as np
import pandas as pd

TARGET = "Calories"
PCT_COLS = ["Vitamin A (% DV)", "Vitamin C (% DV)", "Calcium (% DV)", "Iron (% DV)"]
NUM_COLS = [
    "Total Fat (g)", "Trans Fat (g)", "Saturated Fat (g)", "Sodium (mg)",
    "Total Carbohydrates (g)", "Cholesterol (mg)", "Dietary Fibre (g)",
    "Sugars (g)", "Protein (g)", *PCT_COLS, "Caffeine (mg)",
]
CAT_COLS = ["Beverage_category", "Beverage", "Beverage_prep"]


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()                      # ' Total Fat (g)' -> 'Total Fat (g)'
    df["Total Fat (g)"] = df["Total Fat (g)"].replace("3 2", "3.2")  # typo in source data
    for c in PCT_COLS:                                       # '10%' / '10.00%' -> 10.0
        df[c] = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False), errors="coerce")
    for c in set(NUM_COLS) - set(PCT_COLS):                  # 'Varies' -> NaN (imputed in pipeline)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        df[c] = df[c].str.strip()
    return df


def build_input(**values) -> pd.DataFrame:
    """One-row DataFrame with exactly the columns the pipeline was trained on."""
    missing = [c for c in NUM_COLS + CAT_COLS if c not in values]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    return pd.DataFrame([{c: values[c] for c in NUM_COLS + CAT_COLS}])
