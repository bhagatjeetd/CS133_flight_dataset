
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

# OPTIONAL: Save this table
airport_yearly.to_csv("airport_delay_trends.csv", index=False)
print("📁 Airport delay trends saved to CSV.")

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
most_improved = pivoted.sort_values("change").head(10)
most_declined = pivoted.sort_values("change", ascending=False).head(10)

# Print results
print("\n🟢 Top 10 Most Improved Airports (delays reduced):")
print(most_improved[["change"]])

print("\n🔴 Top 10 Most Declined Airports (delays increased):")
print(most_declined[["change"]])
print("✅ Improvement/decline analysis done.")

# ---------------------------------------------
# STEP 5: Plot example airport trends
print("📊 Plotting trends for top changing airports...")
top_airports = most_improved.index.tolist() + most_declined.index.tolist()
filtered = airport_yearly[airport_yearly["Origin"].isin(top_airports)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered, x="year", y="DepDelay", hue="Origin", marker="o")
plt.title("Delay Trends at Top Changing Airports")
plt.ylabel("Avg Departure Delay (minutes)")
plt.xlabel("Year")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
print("✅ Airport trend plot complete.")
