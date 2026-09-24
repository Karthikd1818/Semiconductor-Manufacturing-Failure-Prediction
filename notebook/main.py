from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI(
    title="Semiconductor Manufacturing Failure Prediction API",
    version="1.0"
)


# =========================================================
# LOAD PRODUCTION MODEL
# =========================================================

MODEL_PATH = "secom_quality_model.joblib"

try:
    model_bundle = joblib.load(MODEL_PATH)

    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]
    feature_names = model_bundle["feature_names"]

    print("✅ Production model loaded successfully")
    print("Pipeline:", type(pipeline))
    print("Threshold:", threshold)
    print("Features:", len(feature_names))

except Exception as e:
    pipeline = None
    threshold = None
    feature_names = None

    print("❌ Model loading failed:", e)


# =========================================================
# REQUEST MODEL
# IMPORTANT: None is allowed because original CSV
# contains missing values and Pipeline handles them
# using SimpleImputer(strategy="median")
# =========================================================

class PredictionRequest(BaseModel):
    sensor_values: list[list[float | None]]


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Semiconductor Manufacturing Failure Prediction API is running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    if pipeline is None:

        return {
            "status": "unhealthy",
            "model": "not loaded"
        }

    return {
        "status": "healthy",
        "model": "loaded",
        "features": len(feature_names),
        "threshold": threshold
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if pipeline is None:

        raise HTTPException(
            status_code=500,
            detail="Model is not loaded"
        )

    try:

        # -------------------------------------------------
        # Convert request data to numpy
        # None becomes np.nan
        # -------------------------------------------------

        X = np.array(
            request.sensor_values,
            dtype=np.float64
        )

        # -------------------------------------------------
        # Check 2D
        # -------------------------------------------------

        if X.ndim != 2:

            raise HTTPException(
                status_code=400,
                detail="sensor_values must be a 2D list"
            )

        # -------------------------------------------------
        # Check feature count
        # -------------------------------------------------

        expected_features = len(feature_names)
        received_features = X.shape[1]

        if received_features != expected_features:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected {expected_features} features, "
                    f"but received {received_features}."
                )
            )

        # -------------------------------------------------
        # IMPORTANT
        #
        # DO NOT fill NaN here.
        #
        # Production pipeline already contains:
        #
        # SimpleImputer(strategy="median")
        #
        # -------------------------------------------------

        # -------------------------------------------------
        # Get probability from production pipeline
        # -------------------------------------------------

        probabilities = pipeline.predict_proba(X)

        # Class 1 = Fail
        fail_probability = probabilities[:, 1]

        # -------------------------------------------------
        # Apply locked threshold from training
        # threshold = 0.7
        # -------------------------------------------------

        predictions = (
            fail_probability >= threshold
        ).astype(int)

        # -------------------------------------------------
        # Create results
        # -------------------------------------------------

        results = []

        for i in range(len(predictions)):

            if predictions[i] == 1:

                prediction_text = "FAIL"

            else:

                prediction_text = "PASS"

            results.append(
                {
                    "sample": i + 1,
                    "prediction": prediction_text,
                    "fail_probability": float(
                        fail_probability[i]
                    )
                }
            )

        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return {
            "status": "success",
            "threshold": float(threshold),
            "results": results
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    