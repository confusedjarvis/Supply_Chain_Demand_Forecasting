import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Page config
st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    layout="wide"
)

st.title("📦 Supply Chain Demand Forecasting")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "supply_chain_data.csv"
MODEL_PATH = BASE_DIR / "demand_forecasting_model.pkl"

# Load model & features

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
model_features = joblib.load(BASE_DIR / "model_features.pkl")

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📊 Dataset Overview", "📈 Analysis", "🔮 Forecast Demand"]
)

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Key Business Metrics")
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Products", df["SKU"].nunique())
    c2.metric("Avg Price", round(df["Price"].mean(), 2))
    c3.metric("Total Units Sold", int(df["Number of products sold"].sum()))

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Demand Distribution")
    st.bar_chart(df["Number of products sold"])

    st.subheader("Price vs Demand")
    st.scatter_chart(
        df,
        x="Price",
        y="Number of products sold"
    )

    # ---------- FEATURE IMPORTANCE ----------
    st.subheader("🔍 Feature Importance")

    importances = model.feature_importances_
    features = model_features

    fi_df = (
        pd.DataFrame({"Feature": features, "Importance": importances})
          .sort_values("Importance", ascending=False)
          .head(10)
    )

    fig, ax = plt.subplots()
    ax.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features Driving Demand")

    st.pyplot(fig)

    st.caption(
        "Feature importance indicates how strongly each variable influences the demand prediction."
    )

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Forecast Product Demand")

    price = st.slider("Price", 0.0, 500.0, 50.0)
    availability = st.slider("Availability", 0, 1000, 100)

    if st.button("Predict Demand"):
        # Create empty input with all features
        input_df = pd.DataFrame(0, index=[0], columns=model_features)

        # Fill known values
        if "Price" in input_df.columns:
            input_df.loc[0, "Price"] = price
        if "Availability" in input_df.columns:
            input_df.loc[0, "Availability"] = availability

        prediction = model.predict(input_df)[0]
        st.success(f"📦 Predicted Demand: **{int(prediction)} units**")