import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Paths & output folder
base = Path(__file__).parent
out = base / "outputs"
out.mkdir(exist_ok=True)

# Load & parse date, suppress low_memory warning
df = pd.read_csv(
    base / "airline_2m.csv",
    encoding="latin1",
    parse_dates=["FlightDate"],
    low_memory=False
).rename(columns={"FlightDate": "FL_DATE"})

# Infer cancellations: missing ArrTime → cancelled
df["cancelled"] = df["ArrTime"].isna().astype(int)

# Drop columns with zero non-NA values
empty = [c for c in df.columns if df[c].notna().sum() == 0]
if empty:
    print(f"Dropping {len(empty)} empty columns: {empty}")
    df = df.drop(columns=empty)

# Build flight_key to group recurring schedules
df["flight_key"] = (
        df["Reporting_Airline"] + "_" +
        df["Origin"] + "_" +
        df["Dest"] + "_" +
        df["CRSDepTime"].astype(str)
)

# Aggregate historical cancel rate & mean delays per flight_key
agg = df.groupby("flight_key").agg(
    cancel_rate    = ("cancelled",       "mean"),
    mean_arr_delay = ("ArrDelayMinutes", "mean"),
    mean_dep_delay = ("DepDelayMinutes", "mean")
).reset_index()

df = df.merge(agg, on="flight_key", how="left")

# Drop raw columns
df = df.drop(columns=[
    "ArrTime","ArrDelay","DepTime","DepDelay",
    "CRSArrTime","DivDistance","DivArrDelay"
])

# Prepare features & label
X = df.drop(columns=["FL_DATE", "cancelled", "flight_key"])
y = df["cancelled"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Define pipelines
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

# Evaluate with 5-fold CV & collect F1-scores
scores = {}
for name, pipe in models.items():
    f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    scores[name] = f1
    print(f"{name} cancellation F1-score: {f1:.3f}")

# Plot comparison
plt.figure(figsize=(6,4))
plt.bar(scores.keys(), scores.values(), color=["#4c72b0","#55a868"])
plt.ylim(0,1)
plt.ylabel("CV F1-score")
plt.title("Cancellation Prediction: Model Comparison")
for i,(name,val) in enumerate(scores.items()):
    plt.text(i, val+0.02, f"{val:.2f}", ha="center")
plt.tight_layout()
plt.savefig(out / "q3_model_comparison.png")
print("Saved plot to outputs/q3_model_comparison.png")
