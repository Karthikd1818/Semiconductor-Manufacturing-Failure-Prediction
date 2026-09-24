import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Semiconductor Failure Prediction",
    page_icon="🔬",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔬 Semiconductor Manufacturing Failure Prediction")

st.write(
    "Upload semiconductor sensor data to predict manufacturing quality."
)


# =========================================================
# LOAD MODEL INFORMATION
# =========================================================

MODEL_PATH = "secom_quality_model.joblib"

try:

    model_bundle = joblib.load(MODEL_PATH)

    feature_names = model_bundle["feature_names"]
    threshold = model_bundle["threshold"]

except Exception as e:

    st.error(
        f"Unable to load model information: {e}"
    )

    st.stop()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Sensor CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # READ CSV
        # -------------------------------------------------

        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully ✅")

        # -------------------------------------------------
        # DATASET INFO
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        st.subheader("📊 Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # -------------------------------------------------
        # CHECK REQUIRED FEATURES
        # -------------------------------------------------

        missing_features = [
            col
            for col in feature_names
            if col not in df.columns
        ]

        if missing_features:

            st.error(
                f"❌ {len(missing_features)} required "
                f"features are missing from the CSV."
            )

            st.write(
                "Missing features:",
                missing_features
            )

            st.stop()

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        if st.button(
            "🔮 Predict",
            type="primary"
        ):

            # ------------------------------------------------
            # IMPORTANT:
            # Select EXACT training features
            # in EXACT training order
            # ------------------------------------------------

            input_df = df[
                feature_names
            ].copy()

            # ------------------------------------------------
            # Convert to numeric
            # ------------------------------------------------

            input_df = input_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            # ------------------------------------------------
            # DO NOT fill missing values.
            #
            # Pipeline handles them using:
            # SimpleImputer(strategy="median")
            # ------------------------------------------------

            sensor_values = []

            for row in input_df.itertuples(
                index=False,
                name=None
            ):

                values = []

                for value in row:

                    if pd.isna(value):

                        values.append(None)

                    else:

                        values.append(
                            float(value)
                        )

                sensor_values.append(values)

            # ------------------------------------------------
            # Verify shape
            # ------------------------------------------------

            st.write(
                "Input shape:",
                input_df.shape
            )

            st.write(
                "Model expects:",
                len(feature_names),
                "features"
            )

            # ------------------------------------------------
            # API PAYLOAD
            # ------------------------------------------------

            payload = {
                "sensor_values": sensor_values
            }

            API_URL = (
                "http://127.0.0.1:8000/predict"
            )

            # ------------------------------------------------
            # SEND REQUEST
            # ------------------------------------------------

            try:

                with st.spinner(
                    "Running prediction..."
                ):

                    response = requests.post(
                        API_URL,
                        json=payload,
                        timeout=120
                    )

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "Prediction completed successfully! ✅"
                    )

                    results = result["results"]

                    results_df = pd.DataFrame(
                        results
                    )

                    # ------------------------------------------------
                    # Format probability
                    # ------------------------------------------------

                    results_df[
                        "fail_probability"
                    ] = (
                        results_df[
                            "fail_probability"
                        ] * 100
                    ).round(2)

                    results_df = results_df.rename(
                        columns={
                            "sample": "Sample",
                            "prediction": "Prediction",
                            "fail_probability":
                                "Fail Probability (%)"
                        }
                    )

                    # ------------------------------------------------
                    # RESULTS
                    # ------------------------------------------------

                    st.subheader(
                        "📋 Prediction Results"
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # SUMMARY
                    # ------------------------------------------------

                    total = len(results_df)

                    fail_count = int(
                        (
                            results_df["Prediction"]
                            == "FAIL"
                        ).sum()
                    )

                    pass_count = (
                        total - fail_count
                    )

                    st.subheader(
                        "📊 Prediction Summary"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Total Samples",
                            total
                        )

                    with col2:

                        st.metric(
                            "PASS",
                            pass_count
                        )

                    with col3:

                        st.metric(
                            "FAIL",
                            fail_count
                        )

                    st.info(
                        f"Decision threshold: {threshold}"
                    )

                # ------------------------------------------------
                # API ERROR
                # ------------------------------------------------

                else:

                    st.error(
                        f"FastAPI returned "
                        f"status {response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except:

                        st.write(
                            response.text
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI."
                )

                st.info(
                    "Start FastAPI with: "
                    "uvicorn main:app --reload"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "❌ API request timed out."
                )

    except Exception as e:

        st.error(
            f"❌ Error processing CSV: {e}"
        )