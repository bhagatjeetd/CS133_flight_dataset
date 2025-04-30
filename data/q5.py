import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Paths & ensure outputs folder
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

# Drop any entirely empty columns
empty = [c for c in df.columns if df[c].notna().sum() == 0]
if empty:
    print(f"Dropping {len(empty)} empty columns: {empty}")
    df = df.drop(columns=empty)

# Create flight_key to group recurring schedules
df["flight_key"] = (
    df["Reporting_Airline"] + "_" +
    df["Origin"] + "_" +
    df["Dest"] + "_" +
    df["CRSDepTime"].astype(str)
)

# Aggregate historical statistics per flight_key
agg = df.groupby("flight_key").agg(
    hist_cancel_rate    = ("ArrTime", lambda x: x.isna().mean()),
    hist_mean_arr_delay = ("ArrDelayMinutes", "mean"),
    hist_mean_dep_delay = ("DepDelayMinutes", "mean")
).reset_index()
df = df.merge(agg, on="flight_key", how="left")

# Create the delay-flag and drop raw delay columns
df["delay15"] = (df["ArrDelayMinutes"] > 15).astype(int)
df = df.drop(columns=[
    "ArrTime","DepTime","ArrDelay","DepDelay",
    "ArrDelayMinutes","DepDelayMinutes",
    "CRSArrTime","DivDistance","DivArrDelay"
])

# Prepare features & label
X = df.drop(columns=["FL_DATE","delay15","flight_key"])
y = df["delay15"]

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing pipeline
num_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","category"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Define model pipelines (RandomForest vs. LogisticRegression)
models = {
    "RandomForest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "LogisticRegression": Pipeline([
        ("pre", preprocessor),
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
}

# 5-fold CV and collect F1-scores
scores = {}
for name, pipe in models.items():
    f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    scores[name] = f1
    print(f"{name} delay>15 min F1-score: {f1:.3f}")

# Plot comparison
plt.figure(figsize=(6,4))
plt.bar(scores.keys(), scores.values(), color=["#4c72b0","#dd8452"])
plt.ylim(0,1)
plt.ylabel("CV F1-score")
plt.title("Delay>15 min Classification: Model Comparison")
for i,(name,val) in enumerate(scores.items()):
    plt.text(i, val + 0.02, f"{val:.2f}", ha="center")
plt.tight_layout()
plt.savefig(out / "q5_model_comparison.png")
print("Saved plot to outputs/q5_model_comparison.png")

for name, pipe in models.items():
    f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    print(f"{name} delay>15 min F1-score: {f1:.3f}")
