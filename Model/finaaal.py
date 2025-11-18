# train_realistic_hd_routes_xgb.py
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Try import xgboost and optuna
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost not installed. Install with `pip install xgboost==2.0.3`") from e

try:
    import optuna
except Exception as e:
    raise ImportError("optuna not installed. Install with `pip install optuna`") from e

# -------------------------
# Paths
# -------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE.joinpath("..", "Data", "realistic_hd_routes_B_dataset.csv").resolve()
MAP = BASE.joinpath("..", "Data", "realistic_hd_routes_B_route_id_map.csv").resolve()
ART = BASE.joinpath("..", "ModelArtifacts")
ART.mkdir(parents=True, exist_ok=True)

print("Loading Data: ")
# --- PATCH: Load CSV ---
df = pd.read_csv(DATA)
print("Rows:", df.shape[0], "Unique route_ids:", df['route_id'].nunique())

# --- PATCH: FIX route_id indexing for XGBoost ---
# route_id was 1..132 → convert to 0..131
df['route_id'] = df['route_id'] - 1


# -------------------------
# Preprocess
# -------------------------
# Drop 'variant' (route_id encodes variant)
X = df.drop(columns=["route_id", "variant"])
y = df["route_id"].astype(int)

# Encode categorical source/destination
encoders = {}
for col in ["source", "destination"]:
    if col in X.columns and X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# Feature list and numeric scaling
feature_cols = X.columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Train/test split:", X_train.shape, X_test.shape)

# -------------------------
# Optuna objective
# -------------------------
def objective(trial):
    params = {
        "tree_method": "hist",
        "objective": "multi:softprob",
        "num_class": len(np.unique(y)),
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        "random_state": 42,
        "verbosity": 0,
    }

    clf = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = []
    for train_idx, valid_idx in skf.split(X, y):
        Xtr, Xv = X.iloc[train_idx], X.iloc[valid_idx]
        ytr, yv = y.iloc[train_idx], y.iloc[valid_idx]
        clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], early_stopping_rounds=40, verbose=False)
        preds = clf.predict(Xv)
        scores.append(f1_score(yv, preds, average="weighted"))
    return np.mean(scores)

# -------------------------
# Run small Optuna study (30 trials)
# -------------------------
print("Starting Optuna tuning (30 trials). This may take some minutes...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best trial params:", study.best_trial.params)
best_params = study.best_trial.params
best_params.update({
    "tree_method": "hist",
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "eval_metric": "mlogloss",
    "random_state": 42,
    "verbosity": 0
})

# -------------------------
# Train final model on train split with early stopping on test
# -------------------------
print("Training final model with best params...")
final_clf = XGBClassifier(**best_params)
final_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=80, verbose=True)

# -------------------------
# Evaluate
# -------------------------
y_pred = final_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nFINAL METRICS")
print("Accuracy:", acc)
print("Precision (weighted):", prec)
print("Recall (weighted):", rec)
print("F1 (weighted):", f1)
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (final model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature importance (top 20)
try:
    fi = pd.DataFrame({"feature": feature_cols, "importance": final_clf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False).head(20)
    print("\nTop features:\n", fi)
    plt.figure(figsize=(8,6))
    sns.barplot(x="importance", y="feature", data=fi)
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not compute feature importances:", e)

# -------------------------
# Save artifacts
# -------------------------
joblib.dump(final_clf, ART.joinpath("best_model_xgb.joblib"))
joblib.dump(scaler, ART.joinpath("scaler.joblib"))
joblib.dump(encoders, ART.joinpath("encoders.joblib"))
joblib.dump(feature_cols, ART.joinpath("feature_columns.joblib"))

print("\nSaved artifacts in:", ART.resolve())
print("Model file:", ART.joinpath("best_model_xgb.joblib").resolve())

# -------------------------
# Quick inference helper printout (example)
# -------------------------
def predict_route_from_example(example_dict):
    # example_dict must contain the same feature columns used for training (source/destination numeric or strings)
    ex = example_dict.copy()
    # encode if necessary
    for c, enc in encoders.items():
        if isinstance(ex.get(c, None), str):
            ex[c] = enc.transform([ex[c]])[0]
    # build df
    df_ex = pd.DataFrame([ex])[feature_cols]
    df_ex[num_cols] = scaler.transform(df_ex[num_cols])
    pred = final_clf.predict(df_ex)[0]
    return int(pred)

print("\nDone. Use the saved model in ModelArtifacts/ for inference.")
