
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained models and dataset
regression_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")
df = pd.read_csv("access_logs.csv")

# Page configuration
st.set_page_config(
    page_title="API Response Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ API Response-Time Intelligence System")
st.write(
    "Machine Learning based API latency prediction "
    "and performance classification."
)

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total API Records", f"{len(df):,}")
col2.metric("ML Models", "2")
col3.metric("Prediction Engine", "Active")

st.divider()

st.subheader("API Request Configuration")
st.write("Select the API request details below.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    server_id = st.selectbox(
        "Server ID",
        sorted(df["server_id"].dropna().unique().tolist())
    )

    method = st.selectbox(
        "HTTP Method",
        sorted(df["method"].dropna().unique().tolist())
    )

    path = st.selectbox(
        "API Endpoint",
        sorted(df["path"].dropna().unique().tolist())
    )

    user_agent = st.selectbox(
        "User Agent",
        df["user_agent"].dropna().unique().tolist()
    )

with col2:
    referrer = st.selectbox(
        "Referrer",
        df["referrer"].dropna().unique().tolist()
    )

    hour = st.slider("Request Hour", 0, 23, 12)

    day_of_week = st.selectbox(
        "Day of Week",
        options=list(range(7)),
        format_func=lambda x: [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ][x]
    )

st.divider()

# Prediction button
if st.button("Predict API Performance", type="primary"):

    input_data = pd.DataFrame([{
        "server_id": server_id,
        "method": method,
        "path": path,
        "user_agent": user_agent,
        "referrer": referrer,
        "hour": hour,
        "day_of_week": day_of_week
    }])

    # Regression prediction
    predicted_time = regression_model.predict(input_data)[0]

    # Classification prediction
    predicted_category = classification_model.predict(
        input_data
    )[0]

    st.subheader("Prediction Results")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Predicted Response Time",
            f"{predicted_time:.2f} ms"
        )

    with result2:
        st.metric(
            "Performance Category",
            predicted_category
        )

    if predicted_category == "Fast":
        st.success("API performance category: Fast")
    elif predicted_category == "Normal":
        st.info("API performance category: Normal")
    elif predicted_category == "Slow":
        st.warning("API performance category: Slow")
    else:
        st.error("API performance category: Critical")

    st.subheader("Request Details")
    st.dataframe(input_data, use_container_width=True)

    st.caption(
        "Predictions are based on historical training data "
        "and are not guaranteed production measurements."
    )

st.divider()

st.subheader("Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("Regression MAE", "174.49 ms")
m2.metric("Regression RMSE", "284.77 ms")
m3.metric("Classification Accuracy", "64%")

st.caption(
    "Model evaluation values are from the current test dataset."
)


# Advanced Analytics Section
st.divider()

st.header("Advanced API Analytics")

# Response time distribution
st.subheader("API Response-Time Distribution")

st.bar_chart(
    df["response_time_ms"].value_counts().sort_index().head(30)
)

# HTTP method analysis
st.subheader("HTTP Method Distribution")

method_counts = df["method"].value_counts()
st.bar_chart(method_counts)

# Server request analysis
st.subheader("Server Request Distribution")

server_counts = df["server_id"].value_counts()
st.bar_chart(server_counts)

# Response time statistics
st.subheader("Response-Time Statistics")

st.dataframe(
    df["response_time_ms"].describe().to_frame(),
    use_container_width=True
)
