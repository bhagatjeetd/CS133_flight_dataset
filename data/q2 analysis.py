import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Starting Q2 flight delay analysis...")

# STEP 1: Load and clean the dataset
print("📂 Loading CSV...")
df = pd.read_csv("airline_2m.csv", encoding="ISO-8859-1", parse_dates=["FlightDate"])
print("✅ CSV loaded.")

# Keep only relevant columns
df = df[["FlightDate", "Origin", "DepDelay"]]

# Drop rows with missing delay info
df = df.dropna(subset=["DepDelay"])

# Optional: remove extreme delay values
df = df[(df["DepDelay"] > -60) & (df["DepDelay"] < 1000)]

# Clean airport codes
df["Origin"] = df["Origin"].str.strip().str.upper()

# Extract year from flight date
df["year"] = df["FlightDate"].dt.year
print("🧹 Data cleaned and prepared.")

# ---------------------------------------------
# STEP 2: Average delay across all airports by year
print("📊 Plotting average delay per year...")
avg_delay_by_year = df.groupby("year")["DepDelay"].mean().reset_index()

# Plot overall trend
plt.figure(figsize=(8, 5))
sns.lineplot(data=avg_delay_by_year, x="year", y="DepDelay", marker="o")
plt.title("Average Departure Delay Over the Years")
plt.ylabel("Avg Departure Delay (minutes)")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
plt.show()
print("✅ Yearly delay trend plotted.")

# ---------------------------------------------
# STEP 3: Average delay per airport per year
print("📦 Calculating airport-level yearly delays...")
airport_yearly = df.groupby(["Origin", "year"])["DepDelay"].mean().reset_index()
print("📈 Plotting airport-level delay trends...")

# Plotting trends for top 5 busiest airports
top_airports = df["Origin"].value_counts().head(5).index.tolist()
plt.figure(figsize=(10, 6))

for airport in top_airports:
    data = airport_yearly[airport_yearly["Origin"] == airport]
    sns.lineplot(data=data, x="year", y="DepDelay", label=airport, marker="o")

plt.title("Average Departure Delay Over the Years (Top 5 Airports)")
plt.xlabel("Year")
plt.ylabel("Avg Departure Delay (minutes)")
plt.grid(True)
plt.legend(title="Airport")
plt.tight_layout()
plt.show()
print("✅ Airport-level delay plot displayed.")

# ---------------------------------------------
# ---------------------------------------------
# STEP 4: Identify improvement or decline per airport
print("📈 Analyzing improvement/decline in performance...")

# Pivot to get year columns side by side
pivoted = airport_yearly.pivot(index="Origin", columns="year", values="DepDelay")

# Only keep airports with data in the first and last year
first_year = pivoted.columns.min()
last_year = pivoted.columns.max()
pivoted = pivoted.dropna(subset=[first_year, last_year])

# Calculate change in delay (last year - first year)
pivoted["change"] = pivoted[last_year] - pivoted[first_year]

# Sort to find top improvements and declines
most_improved = pivoted.sort_values("change").head(6)  # top 6 to exclude SCK
most_improved = most_improved[~most_improved.index.isin(["SCK"])].head(5)  # drop SCK
most_declined = pivoted.sort_values("change", ascending=False).head(5)     # top 5 worst

# Print results
print("\n🟢 Top 5 Most Improved Airports (delays reduced):")
print(most_improved[["change"]])

print("\n🔴 Top 5 Most Declined Airports (delays increased):")
print(most_declined[["change"]])
print("✅ Improvement/decline analysis done.")

# ---------------------------------------------
# STEP 5: Plot updated trends for 5 airports
print("📊 Plotting updated relative improvement for top 5 improved airports...")

# Recalculate DelayChange with same logic
baseline = airport_yearly.groupby("Origin")["year"].min().reset_index()
baseline_delays = airport_yearly.merge(baseline, on=["Origin", "year"], suffixes=("", "_baseline"))
baseline_delays = baseline_delays[["Origin", "DepDelay"]].rename(columns={"DepDelay": "BaselineDelay"})
airport_yearly = airport_yearly.merge(baseline_delays, on="Origin")
airport_yearly["DelayChange"] = airport_yearly["BaselineDelay"] - airport_yearly["DepDelay"]

# Plot for improved airports
improved_airports = most_improved.index.tolist()
improved_data = airport_yearly[airport_yearly["Origin"].isin(improved_airports)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=improved_data, x="year", y="DelayChange", hue="Origin", marker="o")
plt.title("Improvement in Departure Delay (Higher = Better)")
plt.ylabel("Improvement from Baseline Delay (minutes)")
plt.xlabel("Year")
plt.axhline(0, color="black", linestyle="--")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot for declined airports
print("📉 Now plotting updated relative worsening for top 5 declined airports...")
declined_airports = most_declined.index.tolist()
declined_data = airport_yearly[airport_yearly["Origin"].isin(declined_airports)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=declined_data, x="year", y="DelayChange", hue="Origin", marker="o")
plt.title("Worsening in Departure Delay (Lower = Worse)")
plt.ylabel("Change from Baseline Delay (minutes)")
plt.xlabel("Year")
plt.axhline(0, color="black", linestyle="--")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
