# XGBoost Model Training Explanation (`finaaal.py`)

This document provides a detailed explanation of the `Model/finaaal.py` script, which is responsible for training the core XGBoost model used in the Dynamic Route Rationalization system.

## 1. Overview
The script performs an end-to-end machine learning pipeline:
1.  **Loads** the route dataset.
2.  **Preprocesses** the data (encoding, scaling).
3.  **Optimizes** hyperparameters using Optuna.
4.  **Trains** the final XGBoost classifier.
5.  **Evaluates** performance.
6.  **Saves** the trained model and artifacts for the backend to use.

## 2. Code Breakdown

### Imports and Setup
The script starts by importing necessary libraries:
-   **Pandas/Numpy**: For data manipulation.
-   **Scikit-Learn**: For preprocessing (`StandardScaler`, `LabelEncoder`), splitting data, and evaluation metrics.
-   **XGBoost**: The gradient boosting library used for the model.
-   **Optuna**: An automatic hyperparameter optimization framework.

It also defines paths to the dataset (`Data/realistic_hd_routes_B_dataset.csv`) and the output directory (`ModelArtifacts/`).

### Data Loading & Preprocessing
```python
# Load Data
df = pd.read_csv(DATA)

# Fix Route ID Indexing
df['route_id'] = df['route_id'] - 1
```
-   **Route ID Adjustment**: XGBoost expects class labels to start from 0. Since the dataset likely has IDs starting from 1, we subtract 1.

```python
# Drop unnecessary columns
X = df.drop(columns=["route_id", "variant"])
y = df["route_id"].astype(int)
```
-   **Feature Selection**: We separate the target (`route_id`) from the features. `variant` is dropped as it's likely a duplicate or non-predictive identifier.

### Encoding and Scaling
```python
# Encode Categorical Features
encoders = {}
for col in ["source", "destination"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Scale Numeric Features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
```
-   **Label Encoding**: Converts text locations (e.g., "Hyderabad", "Gachibowli") into numbers. These encoders are saved so the backend can encode user input exactly the same way.
-   **Standard Scaling**: Normalizes numeric features (like temperature, distance) to have a mean of 0 and variance of 1, which helps the model converge faster.

### Hyperparameter Optimization (Optuna)
```python
def objective(trial):
    params = { ... } # Search space definition
    clf = XGBClassifier(**params)
    # Cross-validation loop...
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```
-   **Objective Function**: Defines what we want to optimize (F1-score).
-   **Search Space**: Optuna tries different combinations of:
    -   `max_depth`: How deep the trees can grow.
    -   `learning_rate`: Step size shrinkage.
    -   `n_estimators`: Number of trees.
    -   `subsample` / `colsample_bytree`: Fraction of data/columns to use per tree (prevents overfitting).
-   **Cross-Validation**: Uses `StratifiedKFold` to ensure the model is robust and doesn't just memorize one specific train/test split.

### Final Model Training
```python
best_params = study.best_trial.params
final_clf = XGBClassifier(**best_params)
final_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=80)
```
-   **Retraining**: Once the best parameters are found, a new model is initialized with them.
-   **Early Stopping**: If the model's performance on the test set stops improving for 80 rounds, training stops early to prevent overfitting.

### Evaluation
The script calculates and prints key metrics:
-   **Accuracy**: Overall correctness.
-   **Precision/Recall/F1**: Weighted averages to handle class imbalances.
-   **Confusion Matrix**: Visualizes where the model is making mistakes.
-   **Feature Importance**: Shows which factors (e.g., "Traffic Pressure", "Hour of Day") drive the decisions most.

### Artifact Saving
```python
joblib.dump(final_clf, ART.joinpath("best_model_xgb.joblib"))
joblib.dump(scaler, ART.joinpath("scaler.joblib"))
# ... saves encoders and feature columns
```
-   **Serialization**: The trained model and all preprocessing objects are saved as `.joblib` files.
-   **Importance**: The Backend (`app.py`) loads these exact files to make predictions that match the training logic.

## 3. How to Run
To retrain the model (e.g., after adding new data):
```bash
python Model/finaaal.py
```
This will overwrite the files in `ModelArtifacts/`, automatically updating the backend's logic.
