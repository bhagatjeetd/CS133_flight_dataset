import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 1. Load & target
df = pd.read_csv("data/airline_2m.csv", parse_dates=["FL_DATE"])
df["cancelled"] = df["CANCELLED"].astype(int)

X = df.drop(columns=["FL_DATE","CANCELLED","cancelled"])
y = df["cancelled"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Preprocessor
num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 4. Pipelines
models = {
    "LogisticRegression": Pipeline([
        ("pre", preprocessor),
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ("pre", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=10, random_state=42))
    ])
}

# 5. CV & results
for name, pipe in models.items():
    f1 = cross_val_score(pipe, X_train, y_train,
                         cv=5, scoring="f1").mean()
    print(f"{name} cancellation F1-score (5-fold): {f1:.3f}")