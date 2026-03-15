# Feature Engineering

## Overview
This module documents the complete feature engineering pipeline applied 
to the Chicago Crime Dataset (2014–2025), producing the final encoded 
dataset used for all downstream model training and evaluation.

---

## Files

| File | Description |
|------|-------------|
| `feature engineering.ipynb` | Full feature engineering pipeline |
| `/figure` | Plots |

> Raw data source: [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

---

## Pipeline Steps

### Column Removal
The following columns were removed prior to modelling: `ID`, `Case Number`, `Date`, `Updated On`, `Block`, `Location`, `X Coordinate`, `Y Coordinate`, `Latitude`, `Longitude`, `IUCR`, `Description`, `FBI Code`

### Low-Frequency Category Merging

* Primary Type (33 categories → retained all 33)  
All categories were retained as arrest rates vary dramatically across 
crime types (4% to 99%). Merging would destroy critical predictive signal.

* Location Description (173 categories → 31)  
Top 30 categories by frequency were retained (94.2% coverage); 
remaining 143 categories were merged into `OTHER`.  
The `OTHER` bucket arrest rate (17.9%) is consistent with the 
overall dataset arrest rate (18.7%), confirming that no high-signal 
categories were incorrectly merged.

### Binary Encoding
| Feature | Encoding |
|---------|----------|
| Arrest | False → 0, True → 1 (target variable) |
| Domestic | False → 0, True → 1 |

### Derived Features
| Feature | Definition |
|---------|------------|
| is_weekend | 1 if Saturday or Sunday, else 0 |

### Cyclical Encoding
`Hour`, `Month`, and `Weekday` were encoded using sine/cosine transformation 
to preserve their cyclical continuity (e.g., Hour 23 and Hour 0 are adjacent):

### One-Hot Encoding
| Feature | Categories | Columns Generated |
|---------|------------|-------------------|
| Primary Type | 33 | 33 |
| Location Description | 31 (Top 30 + OTHER) | 31 |
| District | 23 | 23 |

### Multicollinearity Check (VIF)
VIF was computed on all numeric features. All remaining features returned VIF < 5, 
indicating no multicollinearity concern.

| Feature | VIF |
|---------|-----|
| is_night | 4.98 |
| Year | 4.78 |
| is_weekend | 3.97 |
| hour_cos | 3.17 |
| Others | < 3.0 |

### Train/Test Split

| Split | Condition | Size | Arrest Rate |
|-------|-----------|------|-------------|
| Train | Year < 2025 | 2,795,351 | 18.70% |
| Test | Year ≥ 2025 | 236,114 | 15.90% |

The 2.8% difference in arrest rates between train and test sets 
reflects the long-term declining trend in citywide arrest rates, 
not a sampling artifact.
