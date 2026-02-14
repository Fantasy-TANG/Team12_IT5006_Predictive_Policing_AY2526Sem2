import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

DATA_PATH = "data/processed/crimes_clean.parquet"
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"
MAP_DIR = "outputs/maps"

df = pd.read_parquet(DATA_PATH)

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}", dpi=200)
    plt.close()

# District ranking
district_counts = df["District"].fillna(-1).astype("int64").value_counts().sort_values(ascending=False)
district_counts.head(15).to_csv(f"{TAB_DIR}/top15_district_counts.csv", header=["count"])

plt.figure()
district_counts.head(15).plot(kind="bar")
plt.title("Top 15 Districts by Crime Count")
plt.xlabel("District (Unknown=-1)")
plt.ylabel("Number of Incidents")
savefig("spatial_top15_districts.png")

# Community Area ranking
if "Community Area" in df.columns:
    ca_counts = df["Community Area"].fillna(-1).astype("int64").value_counts().sort_values(ascending=False)
    ca_counts.head(15).to_csv(f"{TAB_DIR}/top15_community_area_counts.csv", header=["count"])

    plt.figure()
    ca_counts.head(15).plot(kind="bar")
    plt.title("Top 15 Community Areas by Crime Count")
    plt.xlabel("Community Area (Unknown=-1)")
    plt.ylabel("Number of Incidents")
    savefig("spatial_top15_community_areas.png")

# Folium heatmap (sample to avoid huge file)
geo = df[["Latitude","Longitude"]].dropna()
# sampling to keep HTML size reasonable
geo_sample = geo.sample(n=min(200000, len(geo)), random_state=42)

center = [geo_sample["Latitude"].mean(), geo_sample["Longitude"].mean()]
m = folium.Map(location=center, zoom_start=11, control_scale=True)
HeatMap(geo_sample.values.tolist(), radius=6, blur=8).add_to(m)

out_html = f"{MAP_DIR}/crime_heatmap_chicago.html"
m.save(out_html)

print("Saved spatial figures to outputs/figures, tables to outputs/tables, map to outputs/maps/crime_heatmap_chicago.html")

