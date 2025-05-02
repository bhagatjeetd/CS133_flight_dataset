import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Paths & outputs
base = Path(__file__).parent
out = base / "outputs"
out.mkdir(exist_ok=True)

# Load & parse FlightDate
df = pd.read_csv(
    base / "airline_2m.csv",
    encoding="latin1",
    parse_dates=["FlightDate"],
    low_memory=False
).rename(columns={"FlightDate": "FL_DATE"})

# Drop diversion cols and empty cols
df = df.drop(columns=[c for c in df.columns if c.startswith("Div")])
empty = [c for c in df.columns if df[c].notna().sum() == 0]
if empty:
    df = df.drop(columns=empty)

# Infer cancellations
df["cancelled"] = df["ArrTime"].isna().astype(int)

# Build flight_key & count occurrences
df["flight_key"] = (
        df["Reporting_Airline"] + "_" +
        df["Origin"] + "_" +
        df["Dest"] + "_" +
        df["CRSDepTime"].astype(str)
)
df["flight_count"] = df.groupby("flight_key")["flight_key"].transform("count")

# Drop raw cols
df = df.drop(columns=[
    "ArrTime","DepTime",
    "ArrDelay","DepDelay",
    "ArrDelayMinutes","DepDelayMinutes",
    "CRSArrTime","CRSDepTime",
    "flight_key"
])

# Features & target
X = df.drop(columns=["FL_DATE","cancelled"])
y = df["cancelled"]

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Model pipelines: Linear SVM vs. Decision Tree
models = {
    "LinearSVM": Pipeline([
        ("pre", preprocessor),
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LinearSVC(max_iter=5000, random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ("pre", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=10, random_state=42))
    ])
}

# Train, predict, evaluate
scores = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    scores[name] = f1_score(y_test, preds)
    print(f"{name} test F1-score: {scores[name]:.3f}")

#  Plot comparison
plt.figure(figsize=(6,4))
plt.bar(scores.keys(), scores.values(), color=["#4c72b0","#55a868"])
plt.ylim(0,1)
plt.ylabel("Test F1-score")
plt.title("Cancellation Prediction: SVM vs. Decision Tree")
for i,(n,v) in enumerate(scores.items()):
    plt.text(i, v+0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig(out / "q3_test_comparison.png")
print("Saved plot to", out / "q3_test_comparison.png")
