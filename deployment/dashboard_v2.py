import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Crime Arrest Prediction Dashboard", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

MODEL_PATH = os.path.join(project_root, "model", "model_xgb.pkl")
TRAIN_COLUMNS_PATH = os.path.join(project_root, "model", "train_columns.pkl")
RESULTS_DIR = os.path.join(project_root, "results")

TYPE_DISTRICT_RATE_PATH = os.path.join(RESULTS_DIR, "arrest_rate_type_district.csv")
HOUR_RATE_PATH = os.path.join(RESULTS_DIR, "arrest_rate_by_hour.csv")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
ROC_ALL_MODELS_PATH = os.path.join(RESULTS_DIR, "roc_all_models.png")

PRIMARY_TYPE_OPTIONS = [
    "ARSON", "ASSAULT", "BATTERY", "BURGLARY", "CONCEALED CARRY LICENSE VIOLATION",
    "CRIM SEXUAL ASSAULT", "CRIMINAL DAMAGE", "CRIMINAL SEXUAL ASSAULT",
    "CRIMINAL TRESPASS", "DECEPTIVE PRACTICE", "GAMBLING", "HOMICIDE",
    "HUMAN TRAFFICKING", "INTERFERENCE WITH PUBLIC OFFICER", "INTIMIDATION",
    "KIDNAPPING", "LIQUOR LAW VIOLATION", "MOTOR VEHICLE THEFT", "NARCOTICS",
    "NON-CRIMINAL", "OBSCENITY", "OFFENSE INVOLVING CHILDREN",
    "OTHER NARCOTIC VIOLATION", "OTHER OFFENSE", "PROSTITUTION",
    "PUBLIC INDECENCY", "PUBLIC PEACE VIOLATION", "RITUALISM", "ROBBERY",
    "SEX OFFENSE", "STALKING", "THEFT", "WEAPONS VIOLATION",
]

LOCATION_OPTIONS = [
    "ALLEY", "APARTMENT", "BANK", "BAR OR TAVERN", "COMMERCIAL / BUSINESS OFFICE",
    "CONVENIENCE STORE", "CTA BUS", "CTA TRAIN", "DEPARTMENT STORE", "DRUG STORE",
    "GAS STATION", "GROCERY FOOD STORE", "HOSPITAL BUILDING / GROUNDS",
    "HOTEL / MOTEL", "OTHER", "OTHER (SPECIFY)", "PARK PROPERTY",
    "PARKING LOT / GARAGE (NON RESIDENTIAL)",
    "POLICE FACILITY / VEHICLE PARKING LOT", "RESIDENCE",
    "RESIDENCE - GARAGE", "RESIDENCE - PORCH / HALLWAY",
    "RESIDENCE - YARD (FRONT / BACK)", "RESTAURANT",
    "SCHOOL PUBLIC BUILDING", "SCHOOL PUBLIC GROUNDS", "SIDEWALK",
    "SMALL RETAIL STORE", "STREET", "VEHICLE NON-COMMERCIAL",
]

DISTRICT_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 31]

MODEL_OPTIONS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]

MODEL_FILE_KEYS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
}


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_train_columns(columns_path: str):
    columns = joblib.load(columns_path)
    if isinstance(columns, pd.Index):
        return columns.tolist()
    return list(columns)


@st.cache_data
def load_csv(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


@st.cache_data
def load_classification_report(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path, index_col=0)


def clean_token(value) -> str:
    text = str(value)
    cleaned = (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def set_first_existing_column(feature_row: pd.Series, candidates, value):
    for col in candidates:
        if col in feature_row.index:
            feature_row[col] = value
            break


def activate_one_hot(feature_row: pd.Series, prefixes, value):
    raw = str(value)
    cleaned = clean_token(value)

    candidates = []
    for prefix in prefixes:
        candidates.extend(
            [
                f"{prefix}_{raw}",
                f"{prefix}_{cleaned}",
                f"{clean_token(prefix)}_{raw}",
                f"{clean_token(prefix)}_{cleaned}",
            ]
        )

    set_first_existing_column(feature_row, candidates, 1)


def build_feature_vector(
    train_columns,
    primary_type,
    location_description,
    district,
    year,
    month,
    day,
    hour,
    domestic,
):
    event_dt = datetime(year, month, day)
    weekday = event_dt.weekday()

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    is_weekend = 1 if weekday >= 5 else 0

    feature_row = pd.Series(0.0, index=train_columns, dtype="float64")

    numeric_candidates = {
        "Year": year,
        "Month": month,
        "Day": day,
        "Hour": hour,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "Domestic": int(domestic),
    }

    for col_name, value in numeric_candidates.items():
        if col_name in feature_row.index:
            feature_row[col_name] = value

    activate_one_hot(feature_row, ["Primary Type", "Primary_Type"], primary_type)
    activate_one_hot(feature_row, ["Location Description", "Location_Description"], location_description)
    activate_one_hot(feature_row, ["District"], district)
    activate_one_hot(feature_row, ["Year"], year)
    activate_one_hot(feature_row, ["Month"], month)
    activate_one_hot(feature_row, ["Day"], day)
    activate_one_hot(feature_row, ["Hour"], hour)
    activate_one_hot(feature_row, ["Domestic"], int(domestic))
    activate_one_hot(feature_row, ["is_weekend"], is_weekend)

    feature_df = pd.DataFrame([feature_row])
    feature_df = feature_df.reindex(columns=train_columns, fill_value=0)
    return feature_df


def render_prediction_page():
    st.title("Arrest Prediction Tool")
    st.write("Enter a case profile and click Predict to estimate arrest likelihood.")

    col1, col2 = st.columns(2)
    with col1:
        primary_type = st.selectbox("Primary Type", PRIMARY_TYPE_OPTIONS)
        location_description = st.selectbox("Location Description", LOCATION_OPTIONS)
        district = st.selectbox("District", DISTRICT_OPTIONS)

    with col2:
        row_year, row_month, row_day = st.columns(3)
        with row_year:
            year = st.selectbox("Year", list(range(2015, 2027)))
        with row_month:
            month = st.selectbox("Month", list(range(1, 13)))
        max_day = int(pd.Timestamp(year=year, month=month, day=1).days_in_month)
        with row_day:
            day = st.selectbox("Day", list(range(1, max_day + 1)))
        hour = st.slider("Hour", min_value=0, max_value=23, value=12)
        st.write("")
        st.write("")
        domestic = st.checkbox("Domestic")

    if st.button("Predict", type="primary"):
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model not found at: {MODEL_PATH}")
            return
        if not os.path.exists(TRAIN_COLUMNS_PATH):
            st.error(f"Train columns file not found at: {TRAIN_COLUMNS_PATH}")
            return

        try:
            train_columns = load_train_columns(TRAIN_COLUMNS_PATH)
            model = load_model(MODEL_PATH)
            input_df = build_feature_vector(
                train_columns=train_columns,
                primary_type=primary_type,
                location_description=location_description,
                district=district,
                year=year,
                month=month,
                day=day,
                hour=hour,
                domestic=domestic,
            )
        except ValueError as err:
            st.error(f"Invalid date selected: {err}")
            return
        except Exception as err:
            st.error(f"Failed during preprocessing: {err}")
            return

        try:
            pred_label = int(model.predict(input_df)[0])
            pred_prob = float(model.predict_proba(input_df)[0, 1])
        except Exception as err:
            st.error(f"Prediction failed: {err}")
            return

        st.subheader("Prediction Result")
        arrest_text = "Yes" if pred_label == 1 else "No"

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Arrest Predicted", arrest_text)
        with result_col2:
            st.metric("Arrest Probability", f"{pred_prob:.2%}")
        st.progress(min(max(int(round(pred_prob * 100)), 0), 100))

        if pred_prob > 0.6:
            st.success("High likelihood of arrest")
        elif pred_prob > 0.3:
            st.warning("Moderate likelihood")
        else:
            st.error("Low likelihood of arrest")

        st.subheader("Historical Reference")
        type_district_df = load_csv(TYPE_DISTRICT_RATE_PATH)
        hour_df = load_csv(HOUR_RATE_PATH)

        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            st.markdown("**Same Primary Type & District**")
            if type_district_df is None:
                st.info("Historical file not found: arrest_rate_type_district.csv")
            else:
                mask = (
                    (type_district_df["Primary Type"] == primary_type)
                    & (type_district_df["District"] == district)
                )
                match = type_district_df.loc[mask]
                if match.empty:
                    st.info("No historical data for this Primary Type & District.")
                else:
                    row = match.iloc[0]
                    st.metric("Historical Arrest Rate", f"{float(row['arrest_rate']):.2%}")
                    st.caption(f"Total historical cases: {int(row['total_cases']):,}")

        with hist_col2:
            st.markdown("**Same Hour**")
            if hour_df is None:
                st.info("Historical file not found: arrest_rate_by_hour.csv")
            else:
                match = hour_df.loc[hour_df["Hour"] == hour]
                if match.empty:
                    st.info("No historical data for this hour.")
                else:
                    row = match.iloc[0]
                    st.metric("Historical Arrest Rate", f"{float(row['arrest_rate']):.2%}")
                    st.caption(f"Total historical cases: {int(row['total_cases']):,}")


def render_model_comparison_page():
    st.title("Model Comparison")

    st.subheader("Overall Model Performance")
    model_df = load_csv(MODEL_COMPARISON_PATH)
    if model_df is None:
        st.warning("Model comparison file not found: model_comparison.csv")
    else:
        preferred_cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
        available_cols = [col for col in preferred_cols if col in model_df.columns]
        display_df = model_df[available_cols].copy()
        metric_cols = [
            col
            for col in display_df.columns
            if col != "Model" and pd.api.types.is_numeric_dtype(display_df[col])
        ]
        if metric_cols:
            display_df[metric_cols] = display_df[metric_cols].round(4)
        styled_df = display_df.style.hide(axis="index")
        if metric_cols:
            styled_df = styled_df.highlight_max(subset=metric_cols, color="#c6f6d5")
            styled_df = styled_df.format({col: "{:.4f}" for col in metric_cols})
        st.dataframe(styled_df, use_container_width=True)

    overview_col1, overview_col2 = st.columns(2)
    with overview_col1:
        st.markdown("**Features Used**")
        st.markdown(
            "\n".join(
                [
                    "- Primary Type",
                    "- Location Description",
                    "- District",
                    "- Year",
                    "- Month",
                    "- Day",
                    "- Hour",
                    "- Domestic",
                ]
            )
        )

    with overview_col2:
        st.markdown("**ROC Curve**")
        if os.path.exists(ROC_ALL_MODELS_PATH):
            st.image(ROC_ALL_MODELS_PATH, caption="ROC Curve Comparison", use_container_width=True)
        else:
            st.info("ROC comparison image not found: roc_all_models.png")

    st.divider()
    st.subheader("Model Details")

    selected_model = st.selectbox("Choose a model", MODEL_OPTIONS)
    model_key = MODEL_FILE_KEYS[selected_model]

    conf_mat_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_key}.png")
    report_path = os.path.join(RESULTS_DIR, f"classification_report_{model_key}.csv")
    feat_imp_path = os.path.join(RESULTS_DIR, f"feature_importance_{model_key}.png")

    report_df = load_classification_report(report_path)
    if report_df is None:
        st.info(f"Classification report not available for {selected_model}.")
    else:
        st.dataframe(report_df.round(2), use_container_width=True)

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        if os.path.exists(conf_mat_path):
            st.image(conf_mat_path, caption=f"Confusion Matrix - {selected_model}", use_container_width=True)
        else:
            st.info(f"Confusion matrix not available for {selected_model}.")

    with detail_col2:
        if os.path.exists(feat_imp_path):
            st.image(feat_imp_path, caption=f"Feature Importance - {selected_model}", use_container_width=True)
        else:
            st.info(f"Feature importance chart not available for {selected_model}.")


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Page 1: Arrest Prediction Tool", "Page 2: Model Comparison"],
)

st.sidebar.subheader("About")
st.sidebar.write(
    "This dashboard predicts Chicago crime arrest likelihood, compares model "
    "performance, to support transparent data-informed "
    "analysis and operational decisions citywide."
)

st.sidebar.subheader("Contributors")
st.sidebar.markdown("**IT5006       Group 12**")
contrib_col1, contrib_col2 = st.sidebar.columns(2)
with contrib_col1:
    st.write("LI Mingyue")
    st.write("TANG Yun")
with contrib_col2:
    st.write("LI Sitong")
    st.write("WU Silin")

if page == "Page 1: Arrest Prediction Tool":
    render_prediction_page()
else:
    render_model_comparison_page()
