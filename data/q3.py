import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Paths & output folder
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

# Create cancellation flag
df["cancelled"] = df["ArrTime"].isna().astype(int)

# Drop Div* columns
df = df.drop(columns=[c for c in df.columns if c.startswith("Div")])

# Drop columns with zero non-NA
empty = [c for c in df.columns if df[c].notna().sum() == 0]
if empty:
    df = df.drop(columns=empty)

# Build flight_key & count occurrences
df["flight_key"] = (
        df["Reporting_Airline"] + "_" +
        df["Origin"] + "_" +
        df["Dest"] + "_" +
        df["CRSDepTime"].astype(str)
)
df["flight_count"] = df.groupby("flight_key")["flight_key"].transform("count")

# Drop raw columns no longer needed
df = df.drop(columns=[
    "ArrTime","DepTime",
    "ArrDelay","DepDelay",
    "ArrDelayMinutes","DepDelayMinutes",
    "CRSArrTime","CRSDepTime",
    "flight_key"
])

# Train/test split (80/20)
X = df.drop(columns=["FL_DATE","cancelled"])
y = df["cancelled"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# Model pipelines
models = {
    "LogisticRegression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            solver="liblinear", random_state=42
        ))
    ]),
    "DecisionTree": Pipeline([
        ("pre", preprocessor),
        ("clf", DecisionTreeClassifier(
            max_depth=10, random_state=42
        ))
    ])
}

# Train, predict, and evaluate on test set
scores = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1 = f1_score(y_test, preds)
    scores[name] = f1
    print(f"{name} test F1-score: {f1:.3f}")

# Plot comparison
plt.figure(figsize=(6,4))
plt.bar(scores.keys(), scores.values(), color=["#4c72b0","#55a868"])
plt.ylim(0,1)
plt.ylabel("Test F1-score")
plt.title("Cancellation Prediction on Full Data\nLR vs. Decision Tree")
for i,(name,val) in enumerate(scores.items()):
    plt.text(i, val + 0.02, f"{val:.2f}", ha="center")
plt.tight_layout()
plt.savefig(out / "q3_test_comparison.png")
print("Saved plot to", out / "q3_test_comparison.png")
