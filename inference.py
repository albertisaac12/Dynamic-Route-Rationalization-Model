import json
import joblib
import numpy as np
import pandas as pd
import os

print("\n=== Loading Inference Engine ===")

# -------------------------------------------------------
# FILE PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "ModelArtifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model_xgb.joblib")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoders.joblib")
FEATURE_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
ROUTE_MAP_PATH = os.path.join(ARTIFACT_DIR, "route_id_map.json")

# -------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------
print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading encoders...")
encoders = joblib.load(ENCODER_PATH)

print("Loading feature columns...")
feature_columns = joblib.load(FEATURE_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

# Load route ID → actual (source,destination) mapping
with open(ROUTE_MAP_PATH, "r") as f:
    route_id_map = json.load(f)

print("Artifacts loaded successfully!\n")


# -------------------------------------------------------
# INPUT PREPARATION
# -------------------------------------------------------
def prepare_input(input_data):
    df = pd.DataFrame([input_data])

    # Encode ONLY: source, destination
    for col in ["source", "destination"]:
        if col in df.columns:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col])
            except:
                val = df[col].iloc[0]
                print(f"⚠ Warning: unseen category '{val}' in {col}. Using fallback.")
                df[col] = le.transform([le.classes_[0]])

    # Make sure all columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # Scale numerical features
    df_scaled = scaler.transform(df)
    return df_scaled


# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------
def predict_route(input_data):
    x = prepare_input(input_data)
    pred_class = int(model.predict(x)[0])

    # Map class back to real route
    if str(pred_class) in route_id_map:
        return {
            "predicted_route_id": pred_class,
            "source": route_id_map[str(pred_class)]["source"],
            "destination": route_id_map[str(pred_class)]["destination"]
        }
    else:
        return {
            "predicted_route_id": pred_class,
            "source": "UNKNOWN",
            "destination": "UNKNOWN"
        }


# -------------------------------------------------------
# QUICK TEST
# -------------------------------------------------------
if __name__ == "__main__":
    print("Running sample prediction...\n")

    sample = {
        "source": "Hyderabad",
        "destination": "Gachibowli",
        "temp": 28,
        "tempmax": 30,
        "tempmin": 22,
        "humidity": 55,
        "windspeed": 3,
        "conditions": 0,       # Already numeric in training
        "traffic_distance_m": 9400,
        "traffic_duration_min": 22,
        "traffic_pressure": 3.2,
        "traffic_efficiency": 510,
        "is_weekend": 0,
        "day_of_week": 3,
        "hour_of_day": 18
    }

    result = predict_route(sample)
    print("\nPREDICTION RESULT:")
    print(json.dumps(result, indent=4))
