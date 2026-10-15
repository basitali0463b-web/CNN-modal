import joblib
import pandas as pd
import streamlit as st

from data_utils import CAT_COLS, build_input, load_clean

st.set_page_config(page_title="Starbucks Calorie Predictor", page_icon="☕")


@st.cache_resource
def load_model():
    return joblib.load("starbucks_model.joblib")


@st.cache_data
def load_reference():
    return load_clean("starbucks.csv")


model = load_model()
ref = load_reference()

st.title("☕ Starbucks Calorie Predictor")
st.caption("Pick a drink and adjust its nutrition values to estimate calories.")

# --- categorical inputs (options come from the data, so they always match training) ---
cat = st.selectbox("Beverage category", sorted(ref["Beverage_category"].unique()))
bev_opts = sorted(ref.loc[ref["Beverage_category"] == cat, "Beverage"].unique())
bev = st.selectbox("Beverage", bev_opts)
prep_opts = sorted(ref.loc[ref["Beverage"] == bev, "Beverage_prep"].unique())
prep = st.selectbox("Size / preparation", prep_opts)

# --- defaults: pre-fill numeric widgets from the selected drink's real row ---
row = ref[(ref["Beverage"] == bev) & (ref["Beverage_prep"] == prep)].iloc[0]
key = f"{bev}|{prep}"  # changing the drink resets the widgets to its defaults


def num(label, col, step=1.0, pct=False):
    default = float(row[col]) if pd.notna(row[col]) else 0.0
    return st.number_input(label, min_value=0.0, value=default, step=step, key=f"{col}|{key}")


c1, c2 = st.columns(2)
with c1:
    fat = num("Total Fat (g)", "Total Fat (g)", 0.1)
    trans = num("Trans Fat (g)", "Trans Fat (g)", 0.1)
    sat = num("Saturated Fat (g)", "Saturated Fat (g)", 0.1)
    sodium = num("Sodium (mg)", "Sodium (mg)")
    carbs = num("Total Carbohydrates (g)", "Total Carbohydrates (g)")
    chol = num("Cholesterol (mg)", "Cholesterol (mg)")
    fibre = num("Dietary Fibre (g)", "Dietary Fibre (g)")
with c2:
    sugars = num("Sugars (g)", "Sugars (g)")
    protein = num("Protein (g)", "Protein (g)", 0.1)
    va = num("Vitamin A (% DV)", "Vitamin A (% DV)")
    vc = num("Vitamin C (% DV)", "Vitamin C (% DV)")
    ca = num("Calcium (% DV)", "Calcium (% DV)")
    fe = num("Iron (% DV)", "Iron (% DV)")
    caf = num("Caffeine (mg)", "Caffeine (mg)")

if st.button("Predict", type="primary"):
    X = build_input(**{
        "Beverage_category": cat, "Beverage": bev, "Beverage_prep": prep,
        "Total Fat (g)": fat, "Trans Fat (g)": trans, "Saturated Fat (g)": sat,
        "Sodium (mg)": sodium, "Total Carbohydrates (g)": carbs, "Cholesterol (mg)": chol,
        "Dietary Fibre (g)": fibre, "Sugars (g)": sugars, "Protein (g)": protein,
        "Vitamin A (% DV)": va, "Vitamin C (% DV)": vc, "Calcium (% DV)": ca,
        "Iron (% DV)": fe, "Caffeine (mg)": caf,
    })
    pred = float(model.predict(X)[0])
    st.success(f"Predicted calories: **{pred:.0f} kcal**")
    st.caption(f"Menu value for this drink: {int(row['Calories'])} kcal")
