import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix

#  Paths & output directory
base = Path(__file__).parent
out = base / "outputs"
out.mkdir(exist_ok=True)

#  Load & initial EDA
df = pd.read_csv(
    base / "airline_2m.csv",
    encoding="latin1",
    parse_dates=["FlightDate"],
    low_memory=False
)
plt.figure(figsize=(8,4))
plt.hist(df["ArrDelayMinutes"].dropna(), bins=100)
plt.title("Distribution of Arrival Delay (minutes)")
plt.xlabel("Delay (min)")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig(out / "arr_delay_hist.png")
plt.close()

# Feature engineering
#   Drop diversion cols & empty cols
df.drop(columns=[c for c in df if c.startswith("Div")], inplace=True)
empty = [c for c in df if df[c].notna().sum() == 0]
df.drop(columns=empty, inplace=True)

#   flight_key & flight_count
df["flight_key"] = (
        df["Reporting_Airline"] + "_" +
        df["Origin"] + "_" +
        df["Dest"] + "_" +
        df["CRSDepTime"].astype(str)
)
df["flight_count"] = df.groupby("flight_key")["flight_key"].transform("count")

#   Target flag
df["delay15"] = (df["ArrDelayMinutes"] > 15).astype(int)

#   Drop raw columns
df.drop(columns=[
    "FlightDate","ArrTime","DepTime",
    "ArrDelay","DepDelay",
    "ArrDelayMinutes","DepDelayMinutes",
    "CRSArrTime","CRSDepTime",
    "flight_key"
], inplace=True)

#  Train/Test split (80/20, stratified)
X = df.drop(columns=["delay15"])
y = df["delay15"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
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

# 5-fold CV comparison
means, stds = {}, {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring="f1", n_jobs=1)
    means[name] = scores.mean()
    stds[name] = scores.std()
    print(f"{name} CV F1: {means[name]:.3f} ± {stds[name]:.3f}")

# Plot CV results
names = list(models.keys())
x = np.arange(len(names))
y_vals = [means[n] for n in names]
y_err = [stds[n] for n in names]

plt.figure(figsize=(6,4))
plt.bar(x, y_vals, yerr=y_err, capsize=5)
plt.xticks(x, names)
plt.ylim(0,1)
plt.ylabel("CV F1-score")
plt.title("Q5: Delay > 15 min Classification\n5-Fold CV Comparison")
for i, v in enumerate(y_vals):
    plt.text(i, v+0.02, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.savefig(out / "q5_test_comparison.png")
plt.close()

# Fit best model (DecisionTree) on full train set
best = models["DecisionTree"]
best.fit(X_train, y_train)
y_pred = best.predict(X_test)
test_f1 = f1_score(y_test, y_pred)
print(f"Decision Tree test F1: {test_f1:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["≤15 min"," >15 min"],
            yticklabels=["≤15 min"," >15 min"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Decision Tree on Test Set")
plt.tight_layout()
plt.savefig(out / "q5_confusion_matrix.png")
plt.close()