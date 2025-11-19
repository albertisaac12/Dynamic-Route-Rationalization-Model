**Project Overview**:
- **Name**: Dynamic Route Rationalization Model
- **Purpose**: Predict the best route (route ID) between a source and destination using combined weather and live traffic signals and synthetic/augmented training data.
- **Key idea**: Collect weather + traffic features per route, clean and augment the data, then train a multi-class classifier (XGBoost) to map features to route IDs. Inference maps model labels back to human-readable source/destination via `route_id_map.json`.

**Quick Start**:
- **Install dependencies** (use a virtualenv):
```
python -m venv venv
venv\Scripts\Activate.ps1; pip install -r requirements.txt; pip install xgboost optuna joblib matplotlib seaborn numpy
```
- **Prepare data**: Many scripts expect CSVs in `Data/` (see Project Layout). Several pre-built files are included (realistic datasets, route mapping, cleaned/augmented CSVs).
- **Run feature collection (optional)**:
  - `Services/feature_extraction.py` collects Visual Crossing weather and TomTom traffic for routes in `Data/raw_routes.csv` and saves `Data/weather_traffic_aggregated.csv`.
- **Clean & augment**:
  - `Services/clean_and_augment.py` reads `Data/weather_traffic_by_route.csv` (or aggregated outputs) and writes `Data/weather_traffic_clean_augmented.csv`.
- **Train model (example)**:
  - Use `Model/finaaal.py` or `Model/model_xgboost.py` to run Optuna tuning and train a final XGBoost classifier. These scripts save artifacts into `ModelArtifacts/`.
- **Run inference**:
  - `inference.py` loads artifacts from `ModelArtifacts/` and defines `predict_route(input_data)` which returns predicted route id and human-readable source/destination.
  - Example: `python inference.py` runs a sample prediction in `__main__`.

**Project Layout**:
- `Data/` — CSV datasets used for training, mapping, and cleaned/augmented outputs.
  - Important files referenced in scripts:
    - `realistic_hd_routes_B_dataset.csv` (training dataset)
    - `realistic_hd_routes_B_route_id_map.csv` (route → source/destination mapping)
    - `raw_routes.csv` (list of routes used by collector)
    - `weather_traffic_by_route.csv`, `weather_traffic_aggregated.csv`, `weather_traffic_clean_augmented.csv`, `weather_traffic_weekday_realistic.csv`
- `Model/` — training scripts and helper models
  - `finaaal.py` — Optuna + XGBoost training (saves `best_model_xgb.joblib`, encoders, scaler, feature list into `ModelArtifacts/`).
  - `model_xgboost.py` — alternative training pipeline with clustering for `conditions` and feature engineering.
  - `model.py` — RandomForest training example and helper functions.
- `ModelArtifacts/` — saved model artifacts (binary joblib files and `route_id_map.json`)
  - Present artifacts:
    - `best_model_xgb.joblib` — trained model
    - `encoders.joblib` — dict of LabelEncoders (source/destination)
    - `feature_columns.joblib` — list of feature columns used during training
    - `scaler.joblib` and `scaler_initial.joblib` — StandardScaler state
    - `route_id_map.json` — mapping from model label → {source, destination}
- `Services/` — ETL and preprocessing utilities
  - `feature_extraction.py` — collects Visual Crossing weather and TomTom traffic and builds combined dataset
  - `clean_and_augment.py` — data cleaning, augmentation, and feature engineering
  - `feature_extraction_2.py`, `normalize.py`, etc. — auxiliary helpers (see folder for details)
- `Backend/` — API server / serving layer (currently empty stubs)
  - `app.py`, `model_loader.py`, `route_predictor.py` are present but empty — placeholders for a Flask API.
- Root scripts:
  - `inference.py` — light-weight inference engine that loads artifacts from `ModelArtifacts/` and exposes `predict_route()` for single-input predictions.
  - `jjson.py` — converts `Data/realistic_hd_routes_B_route_id_map.csv` into `ModelArtifacts/route_id_map.json`.
  - `test.py` — safe API tests for Visual Crossing (weather) and TomTom (traffic) endpoints.

**Model & Artifacts**:
- Trained model: XGBoost multi-class (`best_model_xgb.joblib`). Labels are integer route IDs (often shifted to 0-based for ML training). Scripts generally subtract/add 1 to map between 1-based route IDs present in datasets and 0-based classes used by the ML model.
- Encoders: `encoders.joblib` contains LabelEncoders for `source` and `destination` (transform during inference). Inference falls back to a default class if an unseen category appears.
- Feature columns: `feature_columns.joblib` is used to align input features before scaling and prediction.
- Scaler: `scaler.joblib` (StandardScaler) used to scale numeric columns during both training and inference.
- Route mapping: `route_id_map.json` maps model label (string) → {source, destination} for human-readable results.

**Data Schema (typical row)**:
- route_id, source, destination, date, temp, tempmin, tempmax, humidity, windspeed, precip, conditions, traffic_distance_m, traffic_duration_min, no_traffic_duration_min, is_weekend, day_of_week, hour_of_day, traffic_pressure, traffic_efficiency, month, day
- Feature engineering scripts compute: `is_peak`, `distance_km`, `distance_bin`, `day_part`, `traffic_pressure`, `traffic_efficiency`.

**How Inference Works**:
- `inference.py` loads `best_model_xgb.joblib`, `encoders.joblib`, `feature_columns.joblib`, `scaler.joblib`, and `route_id_map.json` from `ModelArtifacts/`.
- Call `predict_route(input_data)` where `input_data` is a dict with keys matching at least `source`, `destination`, and usual numeric/weather/traffic fields.
- The pipeline:
  1. Build a DataFrame for the single input
  2. Encode `source`/`destination` using saved encoders (with a fallback for unseen values)
  3. Ensure all feature columns exist, fill missing with 0
  4. Scale numeric columns using `scaler` and pass to model
  5. Model predicts class label → map via `route_id_map.json` to source/destination

**Run Commands / Examples**:
- Sample inference run:
```
python inference.py
```
- Train (example):
```
python Model/finaaal.py
# or
python Model/model_xgboost.py
```
- Generate route map JSON:
```
python jjson.py
# creates ModelArtifacts/route_id_map.json from Data/realistic_hd_routes_B_route_id_map.csv
```
- Collect weather+traffic (requires API keys in script):
```
python Services/feature_extraction.py
```
- Clean & augment collected data:
```
python Services/clean_and_augment.py
```

**Dependencies**:
- See `requirements.txt` for base packages. Additional packages required by training scripts:
  - `xgboost`, `optuna`, `joblib`, `matplotlib`, `seaborn`, `numpy`
- Example install (PowerShell):
```
venv\Scripts\Activate.ps1; pip install -r requirements.txt; pip install xgboost optuna joblib matplotlib seaborn numpy
```

**API Keys / Secrets**:
- `Services/feature_extraction.py` and `test.py` contain placeholder Visual Crossing and TomTom API keys. Replace with your own keys before running the collector.

**Notes & Caveats**:
- `Backend/` contains empty placeholder files — if you plan to serve the model as an API, implement `app.py` (Flask) and use `inference.py` or `Backend/model_loader.py` to load artifacts.
- Many CSVs are present in `Data/` — training scripts expect specific filenames. If you rename a dataset, update the path variables inside the scripts.
- Some training scripts assume route IDs start at 1 and translate to 0-based labels internally — be careful when mapping outputs.
- `ModelArtifacts/` contains binary joblib files — these are required for `inference.py` to function.

**Contributing / Next Steps**:
- Implement a Flask API in `Backend/app.py` that exposes a `/predict` endpoint and uses `inference.predict_route()`.
- Add argument parsing to training and inference scripts to make them less hard-coded.
- Add unit tests and CI for data validation and model inference.

**Files Created/Modified by This Documentation Task**:
- `DOCUMENTATION.md` — this file (created at project root) summarizing the repo and usage.

---
If you want, I can now:
- Commit `DOCUMENTATION.md` and create a small `Backend/app.py` example to serve the model, or
- Run a quick local inference using the existing `inference.py` sample call (requires `ModelArtifacts/` present).  Which would you prefer?