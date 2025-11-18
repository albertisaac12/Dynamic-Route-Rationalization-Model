import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# LOAD DATA
# ============================================
INPUT = os.path.join(os.pardir, "Data", "weather_traffic_weekday_realistic.csv")
df = pd.read_csv(INPUT)

print("Dataset loaded:", df.shape)

# ============================================
# PREPROCESSING
# ============================================

# Encode categorical variables
cat_cols = ["source", "destination", "conditions"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Features and Target
X = df.drop(columns=["route_id"])
y = df["route_id"].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Normalize numeric columns
scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=[np.number]).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ============================================
# MODEL TRAINING
# ============================================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
print("Model training completed!")

# ============================================
# EVALUATION
# ============================================
y_pred = model.predict(X_test)

print("\n=== METRICS ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, y_pred))

# ============================================
# CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================
# FEATURE IMPORTANCE
# ============================================
importances = model.feature_importances_
fi_df = pd.DataFrame({"feature": X.columns, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=fi_df)
plt.title("Feature Importance")
plt.show()

# ============================================
# CROSS VALIDATION
# ============================================
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("\nCross-validation accuracy:", cv_scores)
print("Mean:", cv_scores.mean())

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_best_route(sample):
    """
    sample = {
        "source": "Hyderabad",
        "destination": "Gachibowli",
        "day_of_week": 1,
        "temp": 30,
        "humidity": 60,
        "windspeed": 10,
        ...
    }
    """
    sample_df = pd.DataFrame([sample])

    # Encode categorical
    for col in cat_cols:
        sample_df[col] = le.fit(df[col]).transform(sample_df[col])

    # Scale numeric
    sample_df[num_cols] = scaler.transform(sample_df[num_cols])

    pred = model.predict(sample_df)[0]
    return int(pred)

print("\nReady for inference!")

# ============================
# TEST PREDICTION EXAMPLE
# ============================

# Example input (you can modify this)
sample_input = {
    "source": "Hyderabad",
    "destination": "Gachibowli",
    "day_of_week": 1,            # Monday
    "temp": 28.5,
    "tempmin": 22.0,
    "tempmax": 34.0,
    "humidity": 55,
    "windspeed": 12,
    "conditions": "Clear",       # or Cloudy, Rain, etc.
    "traffic_distance_m": 20000,
    "traffic_duration_min": 50,
    "traffic_pressure": 18.2,
    "traffic_efficiency": 0.019,
    "is_weekend": 0
}

# Call the prediction function
best_route = predict_best_route(sample_input)

print("\n============================")
print("🔥 Best Route Prediction 🔥")
print("============================")
print(f"Optimal Route ID: {best_route}")
print("============================")

