import joblib, numpy as np, pandas as pd
from data_utils import load_clean, build_input, NUM_COLS, CAT_COLS, TARGET

model = joblib.load("starbucks_model.joblib")
df = load_clean("starbucks.csv")

def test_cleaning():
    assert all(df[c].dtype.kind in "fi" for c in NUM_COLS)
    assert df["Total Fat (g)"].notna().all()
    assert list(df.columns) == list(df.columns.str.strip())

def test_every_row_predicts():
    p = model.predict(df[NUM_COLS + CAT_COLS])
    assert len(p) == len(df) and np.isfinite(p).all()

def test_in_sample_error_reasonable():
    p = model.predict(df[NUM_COLS + CAT_COLS])
    assert np.abs(p - df[TARGET]).mean() < 10

def test_unknown_category_does_not_crash():
    r = df.iloc[0][NUM_COLS + CAT_COLS].to_dict(); r["Beverage"] = "Brand New Drink"
    assert np.isfinite(model.predict(build_input(**r))[0])

def test_missing_input_raises():
    try:
        build_input(Beverage="x")
    except ValueError:
        return
    raise AssertionError("expected ValueError")

def test_old_app_example_input():
    r = df[(df.Beverage == "Caffè Latte") & (df.Beverage_prep == "Short Nonfat Milk")].iloc[0]
    p = model.predict(build_input(**r[NUM_COLS + CAT_COLS].to_dict()))[0]
    print("latte pred", p, "actual", r[TARGET]); assert abs(p - r[TARGET]) < 25


if __name__ == "__main__":  # works with or without pytest
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
