from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "final_model.joblib"
DATA_PATH = ROOT / "AmesHousing_cleaned.csv"


st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    artifact = joblib.load(MODEL_PATH)
    reference = pd.read_csv(DATA_PATH)
    return artifact, artifact["preprocessing"], reference


def typical_values(reference):
    values = {}
    for column in reference.columns:
        if column == "SalePrice":
            continue
        series = reference[column].dropna()
        if pd.api.types.is_numeric_dtype(series):
            values[column] = float(series.median())
        else:
            values[column] = series.mode().iloc[0]
    return values


def options(reference, column):
    return sorted(reference[column].dropna().unique().tolist(), key=str)


def preprocess_input(values, preprocessing, expected_columns):
    frame = pd.DataFrame([values]).copy()

    frame = frame.drop(columns=preprocessing["identifier_features"], errors="ignore")

    for column in preprocessing["log_features"]:
        frame[column] = np.log1p(pd.to_numeric(frame[column], errors="raise"))

    frame["House Age"] = frame["Yr Sold"] - frame["Year Built"]
    frame["Remod Age"] = frame["Yr Sold"] - frame["Year Remod/Add"]
    frame["Garage Age"] = frame["Yr Sold"] - frame["Garage Yr Blt"]
    frame["Total SF"] = (
        np.expm1(frame["Total Bsmt SF"])
        + np.expm1(frame["1st Flr SF"])
        + frame["2nd Flr SF"]
    )
    for column in ["Pool Area", "Misc Val", "3Ssn Porch", "Screen Porch", "Low Qual Fin SF"]:
        frame[f"Has {column}"] = (frame[column] > 0).astype(int)
    frame = frame.drop(columns=["Year Built", "Year Remod/Add", "Garage Yr Blt"])

    for column, order in preprocessing["ordinal_rank_order"].items():
        frame[column] = frame[column].map({level: rank for rank, level in enumerate(order)})
        if frame[column].isna().any():
            raise ValueError(f"Unsupported value for {column}.")

    frame["Neighborhood"] = frame["Neighborhood"].where(
        frame["Neighborhood"].isin(preprocessing["common_neighborhoods"]), "Other"
    )
    frame = pd.get_dummies(frame, columns=preprocessing["nominal_features"], drop_first=True)
    bool_columns = frame.select_dtypes(include="bool").columns
    frame[bool_columns] = frame[bool_columns].astype(int)

    frame[preprocessing["scale_columns"]] = preprocessing["scaler"].transform(
        frame[preprocessing["scale_columns"]]
    )
    frame = frame.drop(columns=preprocessing["redundant_features_dropped"])
    frame = frame.reindex(columns=expected_columns, fill_value=0)
    missing_columns = frame.columns[frame.isna().any()].tolist()
    if missing_columns:
        raise ValueError(f"The submitted inputs produced missing features: {', '.join(missing_columns[:5])}.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("The submitted inputs produced a non-finite feature vector.")
    return frame


def number_input(label, defaults, column, **kwargs):
    value = defaults[column]
    uses_float = any(isinstance(kwargs.get(name), float) for name in ("min_value", "max_value", "step"))
    value = float(value) if uses_float else int(value)
    return st.number_input(label, value=value, key=f"input_{column}", **kwargs)


def select_input(label, reference, defaults, column):
    values = options(reference, column)
    default = defaults[column]
    index = values.index(default) if default in values else 0
    return st.selectbox(label, values, index=index, key=f"input_{column}")


try:
    artifact, preprocessing, reference = load_artifacts()
    defaults = typical_values(reference)
except Exception as error:
    st.error(f"The prediction assets could not be loaded: {error}")
    st.stop()


st.title("Ames House Price Predictor")
st.write(
    "Estimate the sale price of an Ames, Iowa home from its size, quality, rooms, "
    "location, and amenities. This demo uses the project's saved ensemble model "
    "and preprocessing pipeline; it does not retrain anything."
)

with st.form("prediction_form"):
    st.subheader("Property and location")
    property_col_1, property_col_2, property_col_3 = st.columns(3)
    with property_col_1:
        neighborhood = select_input("Neighborhood", reference, defaults, "Neighborhood")
        ms_zoning = select_input("Zoning", reference, defaults, "MS Zoning")
        lot_area = number_input("Lot area (sq ft)", defaults, "Lot Area", min_value=1.0, step=100.0)
    with property_col_2:
        overall_qual = number_input("Overall quality (1-10)", defaults, "Overall Qual", min_value=1, max_value=10, step=1)
        overall_cond = number_input("Overall condition (1-9)", defaults, "Overall Cond", min_value=1, max_value=9, step=1)
        lot_frontage = number_input("Lot frontage (ft)", defaults, "Lot Frontage", min_value=0.0, step=1.0)
    with property_col_3:
        year_built = number_input("Year built", defaults, "Year Built", min_value=1800, max_value=2026, step=1)
        year_remod = number_input("Year remodeled", defaults, "Year Remod/Add", min_value=1800, max_value=2026, step=1)
        central_air = select_input("Central air", reference, defaults, "Central Air")

    st.subheader("Size and rooms")
    size_col_1, size_col_2, size_col_3 = st.columns(3)
    with size_col_1:
        gr_liv_area = number_input("Above-ground living area (sq ft)", defaults, "Gr Liv Area", min_value=1.0, step=50.0)
        first_floor = number_input("First-floor area (sq ft)", defaults, "1st Flr SF", min_value=0.0, step=50.0)
        second_floor = number_input("Second-floor area (sq ft)", defaults, "2nd Flr SF", min_value=0.0, step=50.0)
    with size_col_2:
        basement = number_input("Basement area (sq ft)", defaults, "Total Bsmt SF", min_value=0.0, step=50.0)
        basement_finished = number_input("Finished basement area (sq ft)", defaults, "BsmtFin SF 1", min_value=0.0, step=50.0)
        full_bath = number_input("Full bathrooms", defaults, "Full Bath", min_value=0, max_value=10, step=1)
    with size_col_3:
        bedrooms = number_input("Bedrooms", defaults, "Bedroom AbvGr", min_value=0, max_value=15, step=1)
        half_bath = number_input("Half bathrooms", defaults, "Half Bath", min_value=0, max_value=10, step=1)
        kitchen_qual = select_input("Kitchen quality", reference, defaults, "Kitchen Qual")

    st.subheader("Condition, garage, and amenities")
    detail_col_1, detail_col_2, detail_col_3 = st.columns(3)
    with detail_col_1:
        exter_qual = select_input("Exterior quality", reference, defaults, "Exter Qual")
        heating_qc = select_input("Heating quality", reference, defaults, "Heating QC")
        functional = select_input("Home functionality", reference, defaults, "Functional")
    with detail_col_2:
        garage_type = select_input("Garage type", reference, defaults, "Garage Type")
        garage_cars = number_input("Garage capacity (cars)", defaults, "Garage Cars", min_value=0.0, max_value=6.0, step=1.0)
        garage_area = number_input("Garage area (sq ft)", defaults, "Garage Area", min_value=0.0, step=25.0)
    with detail_col_3:
        garage_finish = select_input("Garage finish", reference, defaults, "Garage Finish")
        fireplaces = number_input("Fireplaces", defaults, "Fireplaces", min_value=0, max_value=10, step=1)
        pool_area = number_input("Pool area (sq ft)", defaults, "Pool Area", min_value=0.0, step=25.0)

    submitted = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

if submitted:
    values = defaults.copy()
    values.update(
        {
            "Neighborhood": neighborhood,
            "MS Zoning": ms_zoning,
            "Lot Area": lot_area,
            "Overall Qual": overall_qual,
            "Overall Cond": overall_cond,
            "Lot Frontage": lot_frontage,
            "Year Built": year_built,
            "Year Remod/Add": year_remod,
            "Central Air": central_air,
            "Gr Liv Area": gr_liv_area,
            "1st Flr SF": first_floor,
            "2nd Flr SF": second_floor,
            "Total Bsmt SF": basement,
            "BsmtFin SF 1": basement_finished,
            "Full Bath": full_bath,
            "Bedroom AbvGr": bedrooms,
            "Half Bath": half_bath,
            "Kitchen Qual": kitchen_qual,
            "Exter Qual": exter_qual,
            "Heating QC": heating_qc,
            "Functional": functional,
            "Garage Type": garage_type,
            "Garage Cars": garage_cars,
            "Garage Area": garage_area,
            "Garage Finish": garage_finish,
            "Fireplaces": fireplaces,
            "Pool Area": pool_area,
        }
    )
    try:
        model_frame = preprocess_input(values, preprocessing, artifact["feature_columns"])
        prediction_log = artifact["model"].predict(model_frame)[0]
        prediction = max(0.0, float(np.expm1(prediction_log)))
        st.success(f"Estimated sale price: ${prediction:,.0f}")
        st.caption("This is a model estimate, not an appraisal. Actual prices depend on details not represented here.")
    except (TypeError, ValueError, KeyError) as error:
        st.error(f"Please review the property details: {error}")


with st.expander("Model and project details"):
    metrics = artifact.get("metrics", {})
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Model", artifact.get("model_name", "Saved ensemble"))
    metric_col_2.metric("Test R2 (log price)", f"{metrics.get('test_r2_log', 0):.3f}")
    metric_col_3.metric("Test MAE", f"${metrics.get('test_mae_dollars', 0):,.0f}")
    st.write(
        "The final model is a VotingRegressor combining tuned Ridge, Lasso, and "
        "Gradient Boosting models. It predicts log-transformed sale prices and "
        "converts them back to dollars for display."
    )