# Exploratory Data Analysis (EDA)

This folder contains the Exploratory Data Analysis (EDA) code developed for the Team12 Predictive Policing project (IT5006, AY25/26 Semester 2).
The goal of this EDA module is to analyze temporal patterns, spatial distributions, and crime-related correlations in the Chicago crime dataset
(2014–2025), providing insights to support later predictive modeling tasks.

---

## 1. Dataset Description

- Dataset: Chicago Crimes (2014–2025)
- Source: Chicago Data Portal
- Each row represents a reported crime incident, including:
  - Date and time of occurrence
  - Crime category (Primary Type)
  - Location information (District, Community Area, Latitude, Longitude)
  - Arrest indicator
  - Domestic indicator

Due to file size limitations, raw data files are **not included** in this repository.

Expected local data file:
```
eda/data/Crimes.csv
```

---

## 2. Directory Structure

```
eda/
├── src/
│   ├── 01_load_clean.py           # Data loading and cleaning
│   ├── 02_temporal_analysis.py    # Temporal pattern analysis
│   ├── 03_spatial_analysis.py     # Spatial distribution analysis
│   └── 04_correlation_analysis.py # Crime-type correlation analysis
├── outputs/
│   └── *.png                      # Generated figures
├── data/
│   └── (local only, not committed)
└── README.md
```

---

## 3. Environment Setup

Recommended Python version: **Python 3.10+**

Create and activate a virtual environment from the project root directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(If the team maintains a unified dependency file, please follow the team-level configuration.)

---

## 4. Execution Workflow

All scripts should be executed from the **project root directory**.

### Step 1: Data Cleaning and Feature Engineering

```bash
python eda/src/01_load_clean.py
```

This script:
- Loads the raw CSV file
- Parses date and time fields
- Removes duplicate records
- Generates temporal features (Year, Month, Hour, Weekday)
- Outputs a cleaned dataset in parquet format for fast downstream access

---

### Step 2: Temporal Analysis

```bash
python eda/src/02_temporal_analysis.py
```

Analyzes crime patterns across:
- Yearly trends
- Monthly seasonality
- Hour-of-day distributions
- Day-of-week distributions
- Weekday–hour heatmaps

---

### Step 3: Spatial Analysis

```bash
python eda/src/03_spatial_analysis.py
```

Explores spatial crime distributions, including:
- Top districts by crime count
- Top community areas by crime count
- Identification of spatial hotspots

---

### Step 4: Crime Type and Correlation Analysis

```bash
python eda/src/04_correlation_analysis.py
```

Focuses on:
- Distribution of major crime types
- Hour vs. crime type heatmaps
- Arrest rates by crime category
- Domestic crime rates by crime category

---

## 5. Key Analysis Dimensions

- **Temporal Analysis**
  - Long-term yearly trends
  - Seasonal (monthly) variations
  - Daily and hourly crime cycles

- **Spatial Analysis**
  - District-level and community-area-level crime concentration
  - Identification of high-risk regions

- **Correlation Analysis**
  - Relationships between crime types and time
  - Comparison of arrest likelihood across crime categories
  - Analysis of domestic vs. non-domestic crimes

---

## 6. Notes and Limitations

- Raw data files (.csv, .parquet) are excluded via .gitignore
- This EDA is descriptive and exploratory, not predictive
- Results are intended to inform feature engineering and modeling decisions in later project stages

---

## 7. Author

EDA module implemented as part of the Team12 IT5006 Predictive Policing project.
