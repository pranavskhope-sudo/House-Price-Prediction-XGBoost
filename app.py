
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fa;
    color: #172b4d;
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: #172b4d;
}

p, label {
    color: #52627a;
}

.eyebrow {
    color: #287d79;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e4e9f0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(23,43,77,0.04);
}

div[data-testid="stMetricLabel"] p {
    color: #64748b;
    font-size: 13px;
}

div[data-testid="stMetricValue"] {
    color: #173b67;
    font-weight: 700;
}

div.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 9px;
    border: 1px solid #173b67;
    background-color: #173b67;
    color: white;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #24588d;
    color: white;
    border-color: #24588d;
}

div[data-testid="stForm"] {
    border: none;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border-color: #e4e9f0;
    border-radius: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TRAIN XGBOOST MODEL
# --------------------------------------------------
@st.cache_resource
def train_model():

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, X, y_test, y_pred, mae, rmse, r2


with st.spinner("Training XGBoost model..."):
    model, X, y_test, y_pred, mae, rmse, r2 = train_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="eyebrow">Machine Learning • Regression</div>',
    unsafe_allow_html=True
)

st.title("House Price Prediction")

st.markdown(
    '<p class="subtitle">Estimate residential property value '
    'using an XGBoost regression model.</p>',
    unsafe_allow_html=True
)

st.write("")

# --------------------------------------------------
# PROPERTY DETAILS
# --------------------------------------------------
st.subheader("Property Valuation")

st.caption(
    "Enter the property details below to generate an estimated value."
)

input_col, result_col = st.columns(
    [1.45, 1],
    gap="large"
)

with input_col:

    with st.container(border=True):

        st.markdown("#### Property Details")

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                income = st.number_input(
                    "Median Income",
                    min_value=0.5,
                    max_value=15.0,
                    value=5.0,
                    step=0.1,
                    help="Median income in the area, in units of $10,000."
                )

                age = st.number_input(
                    "House Age (Years)",
                    min_value=1.0,
                    max_value=52.0,
                    value=30.0,
                    step=1.0
                )

                rooms = st.number_input(
                    "Average Rooms",
                    min_value=1.0,
                    max_value=15.0,
                    value=5.0,
                    step=0.1
                )

                bedrooms = st.number_input(
                    "Average Bedrooms",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.1
                )

            with col2:

                population = st.number_input(
                    "Population",
                    min_value=3.0,
                    max_value=35000.0,
                    value=1000.0,
                    step=100.0
                )

                occupancy = st.number_input(
                    "Average Occupancy",
                    min_value=1.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1
                )

                latitude = st.number_input(
                    "Latitude",
                    min_value=32.0,
                    max_value=42.0,
                    value=34.0,
                    step=0.1
                )

                longitude = st.number_input(
                    "Longitude",
                    min_value=-124.0,
                    max_value=-114.0,
                    value=-118.0,
                    step=0.1
                )

            submitted = st.form_submit_button(
                "Predict House Price",
                type="primary"
            )

# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------
with result_col:

    with st.container(border=True):

        st.markdown("#### Estimated Property Value")

        if submitted:

            new_house = pd.DataFrame([{
                "MedInc": income,
                "HouseAge": age,
                "AveRooms": rooms,
                "AveBedrms": bedrooms,
                "Population": population,
                "AveOccup": occupancy,
                "Latitude": latitude,
                "Longitude": longitude
            }], columns=X.columns)

            prediction = float(model.predict(new_house)[0])

            price = prediction * 100000

            st.markdown(
                f"<h2 style='color:#173b67; font-size:36px;'>"
                f"${price:,.0f}</h2>",
                unsafe_allow_html=True
            )

            st.caption("XGBoost model estimate • USD")

            st.success("Prediction generated successfully.")

        else:

            st.markdown(
                "<h2 style='color:#94a3b8; font-size:36px;'>—</h2>",
                unsafe_allow_html=True
            )

            st.caption(
                "Enter property details and click Predict House Price."
            )

        st.divider()

        st.caption("Dataset: California Housing")
        st.caption("Model: XGBoost Regression")
        st.caption("Prediction Unit: USD")

        st.info(
            "This is an educational estimate, "
            "not a formal real-estate appraisal."
        )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.write("")

st.subheader("Model Performance")

st.caption(
    "Evaluation on the held-out 20% test dataset. "
    "R² is a regression metric, not classification accuracy."
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Mean Absolute Error",
    f"{mae:.4f}"
)

m2.metric(
    "Root Mean Squared Error",
    f"{rmse:.4f}"
)

m3.metric(
    "R² Score",
    f"{r2:.4f}"
)

# --------------------------------------------------
# MODEL ANALYSIS
# --------------------------------------------------
st.write("")

st.subheader("Model Analysis")

chart1, chart2 = st.columns(2, gap="large")

# Actual vs Predicted Graph
with chart1:

    with st.container(border=True):

        st.markdown("#### Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(6, 4.5))

        actual = np.asarray(y_test) * 100000
        predicted = np.asarray(y_pred) * 100000

        ax.scatter(
            actual,
            predicted,
            alpha=0.3,
            s=12
        )

        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())

        ax.plot(
            [low, high],
            [low, high],
            linestyle="--",
            linewidth=1.5
        )

        ax.set_xlabel("Actual Price (USD)")
        ax.set_ylabel("Predicted Price (USD)")
        ax.grid(alpha=0.18)
        ax.ticklabel_format(
            style="plain",
            axis="both"
        )

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.caption(
            "Points closer to the dashed line indicate "
            "predictions closer to actual values."
        )

# Feature Importance Graph
with chart2:

    with st.container(border=True):

        st.markdown("#### Feature Importance")

        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values()

        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        importance.plot(
            kind="barh",
            ax=ax2
        )

        ax2.set_xlabel("Importance")
        ax2.set_ylabel("")
        ax2.grid(
            axis="x",
            alpha=0.18
        )

        fig2.tight_layout()

        st.pyplot(fig2, use_container_width=True)

        plt.close(fig2)

        st.caption(
            "Shows the relative importance of each input "
            "feature in the trained model."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <p style='text-align:center; color:#94a3b8; font-size:13px;'>
        House Price Prediction using XGBoost<br>
        Fundamental of Machine Learning • California Housing Dataset
    </p>
    """,
    unsafe_allow_html=True
)