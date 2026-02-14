import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/crimes_clean.parquet"
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"

df = pd.read_parquet(DATA_PATH)

# ---- helper ----
def savefig(name: str):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}", dpi=200)
    plt.close()

# Yearly trend
year_counts = df.groupby("Year").size().sort_index()
year_counts.to_csv(f"{TAB_DIR}/year_counts.csv", header=["count"])

plt.figure()
year_counts.plot(kind="line")
plt.title("Crime Incidents by Year (2014–2025)")
plt.xlabel("Year")
plt.ylabel("Number of Incidents")
savefig("temporal_year_trend.png")

# Monthly seasonality (overall)
month_counts = df.groupby("Month").size().sort_index()
month_counts.to_csv(f"{TAB_DIR}/month_counts.csv", header=["count"])

plt.figure()
month_counts.plot(kind="bar")
plt.title("Crime Incidents by Month (All Years)")
plt.xlabel("Month")
plt.ylabel("Number of Incidents")
savefig("temporal_month_seasonality.png")

# Hour of day distribution
hour_counts = df.groupby("Hour").size().sort_index()
hour_counts.to_csv(f"{TAB_DIR}/hour_counts.csv", header=["count"])

plt.figure()
hour_counts.plot(kind="line")
plt.title("Crime Incidents by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Number of Incidents")
savefig("temporal_hour_distribution.png")

# Weekday distribution (ordered)
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_counts = df["Weekday"].value_counts().reindex(weekday_order)
weekday_counts.to_csv(f"{TAB_DIR}/weekday_counts.csv", header=["count"])

plt.figure()
weekday_counts.plot(kind="bar")
plt.title("Crime Incidents by Day of Week")
plt.xlabel("Weekday")
plt.ylabel("Number of Incidents")
savefig("temporal_weekday_distribution.png")

# eatmap table: Weekday x Hour (counts)
heat = pd.crosstab(df["Weekday"], df["Hour"]).reindex(weekday_order)
heat.to_csv(f"{TAB_DIR}/heatmap_weekday_hour_counts.csv")

# Simple heatmap plot (matplotlib only)
plt.figure(figsize=(10,4))
plt.imshow(heat.values, aspect="auto")
plt.title("Heatmap: Incidents by Weekday and Hour (Counts)")
plt.xlabel("Hour")
plt.ylabel("Weekday")
plt.yticks(range(len(weekday_order)), weekday_order)
plt.colorbar(label="Count")
savefig("temporal_heatmap_weekday_hour.png")

print("Saved figures to outputs/figures and tables to outputs/tables.")
