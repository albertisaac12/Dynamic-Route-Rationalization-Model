import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
import xgboost as xgb


# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR.joinpath("..", "Data", "weather_traffic_weekday_realistic.csv")
ARTIFACT_DIR = BASE_DIR.joinpath("..", "ModelArtifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset from:", INPUT)
df = pd.read_csv(INPUT)
print("Dataset loaded:", df.shape)


# ======================================================
# STEP 1 — CLUSTER CONDITIONS (Clear/Cloudy/Rainy/Storm)
# ======================================================
vals = df["conditions"].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=4, random_state=42)
grp = kmeans.fit_predict(vals)

cluster_means = {g: df["conditions"][grp==g].mean() for g in np.unique(grp)}
ordered_clusters = sorted(cluster_means, key=lambda x: cluster_means[x])
remap = {cluster: rank for rank, cluster in enumerate(ordered_clusters)}

df["conditions"] = [remap[int(c)] for c in grp]
print("Clustered 'conditions' into:", remap)


# ======================================================
# FEATURE ENGINEERING
# ======================================================
def engineer(df):
    df = df.copy()

    if "day_of_week" not in df:
        df["day_of_week"] = np.random.randint(1, 8, len(df))
    if "hour_of_day" not in df:
        df["hour_of_day"] = np.random.randint(0, 24, len(df))

    df["is_peak"] = df["hour_of_day"].apply(
        lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 20) else 0
    )

    df["speed_kmph"] = (df["traffic_distance_m"] / df["traffic_duration_min"]) * 60 / 1000
    df["speed_kmph"].replace([np.inf, -np.inf], np.nan, inplace=True)
    df["speed_kmph"].fillna(df["speed_kmph"].median(), inplace=True)

    df["distance_km"] = df["traffic_distance_m"] / 1000
    df["distance_bin"] = pd.qcut(
        df["distance_km"].rank(method="first"), q=4, labels=False
    ).astype(int)

    def part(h):
        if 5 <= h < 12: return 0
        if 12 <= h < 17: return 1
        if 17 <= h < 21: return 2
        return 3

    df["day_part"] = df["hour_of_day"].apply(part)

    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d in [6,7] else 0)

    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())

    return df


df = engineer(df)
print("Feature engineering complete:", df.shape)


# ======================================================
# PREPARE TRAIN/TEST
# ======================================================

target = "route_id"
X = df.drop(columns=["route_id"])
y = df["route_id"].astype(int)

# -------- FIX: Convert classes {1,2,3} → {0,1,2} --------
y = y - y.min()


feature_columns = X.columns.tolist()

# Encode source/destination
encoders = {}
for col in ["source", "destination"]:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# Scale numeric columns
scaler = StandardScaler()
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


# ======================================================
# OPTUNA + MODERN XGBOOST (2.0.3)
# ======================================================
def objective(trial):
    params = {
        "tree_method": "hist",
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
    }

    model = xgb.XGBClassifier(**params)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        Xtr, Xv = X.iloc[train_idx], X.iloc[valid_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(
            Xtr, ytr,
            eval_set=[(Xv, yv)],
            early_stopping_rounds=50,
            verbose=False
        )

        preds = model.predict(Xv)
        scores.append(f1_score(yv, preds, average="weighted"))

    return np.mean(scores)


print("\nStarting Optuna tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nBest parameters:", study.best_trial.params)
best_params = study.best_trial.params

best_params.update({
    "tree_method": "hist",
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "eval_metric": "mlogloss",
})


# ======================================================
# TRAIN FINAL XGBOOST MODEL
# ======================================================
best_model = xgb.XGBClassifier(**best_params)

best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=60,
    verbose=True
)


# ======================================================
# EVALUATION
# ======================================================
y_pred = best_model.predict(X_test)

print("\n=== FINAL METRICS ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.show()


# ======================================================
# SAVE MODEL + ARTIFACTS
# ======================================================
joblib.dump(best_model, ARTIFACT_DIR.joinpath("best_model_xgb.joblib"))
joblib.dump(scaler, ARTIFACT_DIR.joinpath("scaler.joblib"))
joblib.dump(encoders, ARTIFACT_DIR.joinpath("encoders.joblib"))
joblib.dump(feature_columns, ARTIFACT_DIR.joinpath("feature_columns.joblib"))

print("\nArtifacts saved to:", ARTIFACT_DIR)


# ======================================================
# PREDICTION FUNCTION (returns original route_id 1,2,3)
# ======================================================
def predict_best_route(sample: dict):
    model = joblib.load(ARTIFACT_DIR.joinpath("best_model_xgb.joblib"))
    scaler = joblib.load(ARTIFACT_DIR.joinpath("scaler.joblib"))
    encoders = joblib.load(ARTIFACT_DIR.joinpath("encoders.joblib"))
    feature_columns = joblib.load(ARTIFACT_DIR.joinpath("feature_columns.joblib"))

    df_s = pd.DataFrame([sample])

    for col, enc in encoders.items():
        df_s[col] = enc.transform(df_s[col])

    num_cols = df_s.select_dtypes(include=[np.number]).columns
    df_s[num_cols] = scaler.transform(df_s[num_cols])

    df_s = df_s[feature_columns]

    pred = model.predict(df_s)[0]

    # convert ML label 0,1,2 back to original 1,2,3
    return int(pred + 1)


# ======================================================
# Example prediction
# ======================================================
example = {}
for col in feature_columns:
    example[col] = X[col].median()

example["conditions"] = 0         # Clear
example["day_of_week"] = 1
example["hour_of_day"] = 9
example["is_peak"] = 1

print("\n Predicting example...")
print("Predicted route: ", predict_best_route(example))
