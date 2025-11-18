import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================================
# LOAD DATA SAFELY
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.normpath(os.path.join(BASE_DIR, "..", "Data", "weather_traffic_weekday_realistic.csv"))

df = pd.read_csv(INPUT)
print("Dataset loaded:", df.shape)

# ======================================================
# STEP 1 — CLUSTER CONDITIONS INTO 4 CATEGORIES (Option B)
# ======================================================

vals = df["conditions"].values.reshape(-1, 1)

kmeans = KMeans(n_clusters=4, random_state=42)
df["condition_group"] = kmeans.fit_predict(vals)

# Sort clusters by average severity value
cluster_order = (
    df.groupby("condition_group")["conditions"]
      .mean()
      .sort_values()
      .index.tolist()
)

severity_map = {cluster: rank for rank, cluster in enumerate(cluster_order)}
df["condition_group"] = df["condition_group"].map(severity_map)

weather_labels = {
    0: "Clear",     # Lowest severity
    1: "Cloudy",
    2: "Rainy",
    3: "Storm"      # Highest severity
}

df["condition_label"] = df["condition_group"].map(weather_labels)

# Replace original conditions with severity code (0–3)
df["conditions"] = df["condition_group"]

df.drop(columns=["condition_group", "condition_label"], inplace=True)

print("Weather condition classes created successfully!")


# ======================================================
# STEP 2 — PREPROCESSING
# ======================================================

# Encode ONLY source & destination (conditions is already numeric)
cat_cols = ["source", "destination"]
encoders = {}

for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

# Features & Target
X = df.drop(columns=["route_id"])
y = df["route_id"].astype(int)

# Store column order (very important)
feature_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale numeric columns
num_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ======================================================
# STEP 3 — MODEL TRAINING
# ======================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel training completed!")

# ======================================================
# STEP 4 — EVALUATION
# ======================================================

y_pred = model.predict(X_test)

print("\n=== METRICS ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, y_pred))

# ======================================================
# CONFUSION MATRIX
# ======================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=fi_df)
plt.title("Feature Importance")
plt.show()

# ======================================================
# CROSS VALIDATION
# ======================================================

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("\nCross-validation accuracy:", cv_scores)
print("Mean:", cv_scores.mean())


# ======================================================
# STEP 5 — FIXED PREDICTION FUNCTION (WORKS 100%)
# ======================================================

def predict_best_route(sample):
    """Predict the best route for a given feature input."""

    # Convert dict → DataFrame
    sample_df = pd.DataFrame([sample])

    # Validate that all columns exist
    for col in feature_columns:
        if col not in sample_df.columns:
            raise ValueError(f"Missing column in sample_input: {col}")

    # Apply correct column order
    sample_df = sample_df[feature_columns]

    # Encode categorical (source/destination)
    for col in cat_cols:
        sample_df[col] = encoders[col].transform(sample_df[col])

    # Scale numeric columns
    sample_df[num_cols] = scaler.transform(sample_df[num_cols])

    # Predict route
    pred = model.predict(sample_df)[0]
    return int(pred)

print("\nReady for inference!")


# ======================================================
# STEP 6 — TEST SAMPLE PREDICTION (VALID)
# ======================================================

# You now MUST use numeric condition codes:
# 0 = Clear, 1 = Cloudy, 2 = Rainy, 3 = Storm

sample_input = {
    "source": "Hyderabad",
    "destination": "Gachibowli",
    "traffic_distance_m": 20000,
    "traffic_duration_min": 50,
    "temp": 28.5,
    "tempmin": 22.0,
    "tempmax": 34.0,
    "humidity": 55,
    "windspeed": 12,
    "conditions": 0,     # Clear
    "is_weekend": 0,
    "traffic_pressure": 18.2,
    "traffic_efficiency": 0.019,
    "day_of_week": 1
}

result = predict_best_route(sample_input)

print("\n============================")
print("🔥 Best Route Prediction 🔥")
print("============================")
print("Optimal Route ID:", result)
print("============================")
