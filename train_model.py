"""Clean starbucks.csv, compare models with group-aware CV, save the best Pipeline."""
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from data_utils import load_clean, NUM_COLS, CAT_COLS, TARGET

df = load_clean("starbucks.csv")
X, y, groups = df[NUM_COLS + CAT_COLS], df[TARGET], df["Beverage"]

def make(est):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("prep", pre), ("model", est)])

candidates = {
    "DecisionTree(depth5) [old]": DecisionTreeRegressor(max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}
# GroupKFold by Beverage => test on drinks the model has never seen (honest estimate)
cv = GroupKFold(n_splits=5)
scores = {}
for name, est in candidates.items():
    pred = cross_val_predict(make(est), X, y, groups=groups, cv=cv)
    scores[name] = (mean_absolute_error(y, pred), r2_score(y, pred))
    print(f"{name:28s} MAE={scores[name][0]:6.2f} kcal  R2={scores[name][1]:.3f}")

best = min(scores, key=lambda k: scores[k][0])
print("Best:", best)
final = make(candidates[best]).fit(X, y)
joblib.dump(final, "starbucks_model.joblib")
print("Saved starbucks_model.joblib")
