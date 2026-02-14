# cd /Users/wusilin/Desktop/nus/sem2/IT5006/eda
# source .venv/bin/activate

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Crimes.csv")
OUT_PATH = Path("data/processed/crimes_clean.parquet")

print("Loading:", RAW_PATH.resolve())

# 1) load
df = pd.read_csv(RAW_PATH, low_memory=False)
df.columns = [c.strip() for c in df.columns]
print("Raw rows:", len(df))

# 2) parse dates
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).copy()

if "Updated On" in df.columns:
    df["Updated On"] = pd.to_datetime(df["Updated On"], errors="coerce")

# 3) dedup by ID
if "ID" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["ID"])
    print(f"Dedup by ID: {before} -> {len(df)}")

# 4) fix mixed-type columns for parquet
string_cols = [
    "IUCR", "FBI Code", "Case Number", "Block",
    "Location", "Location Description", "Description", "Primary Type"
]
for c in string_cols:
    if c in df.columns:
        df[c] = df[c].astype("string")

# 5) numeric columns
for c in ["Latitude", "Longitude", "District", "Ward", "Community Area", "Beat"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 6) time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Hour"] = df["Date"].dt.hour
df["Weekday"] = df["Date"].dt.day_name()

# 7) optional: coarse geo filter (keep NaNs)
if "Latitude" in df.columns and "Longitude" in df.columns:
    mask_ok = df["Latitude"].between(41.0, 42.5, inclusive="both") & df["Longitude"].between(-88.5, -87.0, inclusive="both")
    df = df[mask_ok | df["Latitude"].isna() | df["Longitude"].isna()].copy()

print("Clean rows:", len(df))
print("Date range:", df["Date"].min(), "->", df["Date"].max())
print("Year counts (tail):")
print(df["Year"].value_counts().sort_index().tail(15))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
