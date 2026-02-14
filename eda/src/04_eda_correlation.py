import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/crimes_clean.parquet"
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"

df = pd.read_parquet(DATA_PATH)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}", dpi=200)
    plt.close()

# Top 15 crime types
type_counts = df["Primary Type"].value_counts()
type_counts.head(15).to_csv(f"{TAB_DIR}/top15_primary_type_counts.csv", header=["count"])

plt.figure()
type_counts.head(15).plot(kind="bar")
plt.title("Top 15 Crime Types (Primary Type)")
plt.xlabel("Primary Type")
plt.ylabel("Number of Incidents")
savefig("corr_top15_crime_types.png")

# Crime type vs hour (top 8 types)
top_types = type_counts.head(8).index
sub = df[df["Primary Type"].isin(top_types)]

ct_hour = pd.crosstab(sub["Hour"], sub["Primary Type"]).sort_index()
ct_hour.to_csv(f"{TAB_DIR}/crosstab_hour_by_type_top8.csv")

plt.figure(figsize=(10,4))
plt.imshow(ct_hour.values, aspect="auto")
plt.title("Heatmap: Hour vs Crime Type (Top 8 Types)")
plt.xlabel("Crime Type")
plt.ylabel("Hour")
plt.xticks(range(len(ct_hour.columns)), ct_hour.columns, rotation=30, ha="right")
plt.yticks(range(0,24,2), range(0,24,2))
plt.colorbar(label="Count")
savefig("corr_heatmap_hour_vs_type_top8.png")

# Arrest rate by crime type (top 15)
arrest_rate = df.groupby("Primary Type")["Arrest"].mean().sort_values(ascending=False)
arrest_rate.head(15).to_csv(f"{TAB_DIR}/top15_arrest_rate_by_type.csv", header=["arrest_rate"])

plt.figure()
arrest_rate.head(15).plot(kind="bar")
plt.title("Top 15 Crime Types by Arrest Rate")
plt.xlabel("Primary Type")
plt.ylabel("Arrest Rate")
savefig("corr_top15_arrest_rate_by_type.png")

# Domestic rate by crime type (top 15)
dom_rate = df.groupby("Primary Type")["Domestic"].mean().sort_values(ascending=False)
dom_rate.head(15).to_csv(f"{TAB_DIR}/top15_domestic_rate_by_type.csv", header=["domestic_rate"])

plt.figure()
dom_rate.head(15).plot(kind="bar")
plt.title("Top 15 Crime Types by Domestic Rate")
plt.xlabel("Primary Type")
plt.ylabel("Domestic Rate")
savefig("corr_top15_domestic_rate_by_type.png")

print("Saved correlation figures and tables.")
