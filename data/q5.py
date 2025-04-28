import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df = pd.read_csv("data/airline_2m.csv", parse_dates=["FL_DATE"])
df["delay15"] = (df["ARR_DELAY"] > 15).astype(int)

X = df.drop(columns=["FL_DATE","ARR_DELAY","delay15"])
y = df["delay15"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

num_cols = X.select_dtypes(["int64","float64"]).columns
cat_cols = X.select_dtypes(["object","category"]).columns

pre = ColumnTransformer([
    ("num", SimpleImputer("median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "RandomForest": Pipeline([("pre",pre),("sc",StandardScaler(with_mean=False)),
                              ("clf",RandomForestClassifier(n_estimators=100, random_state=42))]),
    "XGBoost":      Pipeline([("pre",pre),("sc",StandardScaler(with_mean=False)),
                              ("clf",XGBClassifier(
                                  use_label_encoder=False,
                                  eval_metric="logloss",
                                  random_state=42
                              ))])
}

for name, pipe in models.items():
    f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    print(f"{name} delay>15 min F1-score: {f1:.3f}")
