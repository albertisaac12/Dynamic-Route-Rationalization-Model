# Dynamic Route Rationalization Model

## Project Overview
The Dynamic Route Rationalization Model is a comprehensive AI-powered system designed to predict optimal travel routes by analyzing real-time traffic data, weather conditions, and historical route efficiency. Unlike traditional navigation systems that rely solely on distance or current traffic, this model integrates environmental factors such as temperature, humidity, and wind speed, along with traffic pressure and efficiency metrics, to recommend the most rational route.

The system consists of a Python-based Flask backend that serves a pre-trained XGBoost classifier and a modern, interactive React frontend that visualizes the route predictions in real-time.

## Features

- **AI-Driven Route Prediction**: Utilizes an XGBoost classifier trained on extensive historical and synthetic data to predict the optimal route ID.
- **Multi-Factor Analysis**: Considers a wide range of variables including:
    - **Weather**: Temperature, Humidity, Wind Speed, Conditions.
    - **Traffic**: Distance, Duration, Traffic Pressure, Traffic Efficiency.
    - **Time**: Hour of day, Day of week, Weekend status.
- **Interactive Visualization**: A dynamic map simulation that visualizes the source, the predicted intermediary "via" point, and the final destination.
- **Real-Time Simulation**: Animated route tracing to demonstrate the predicted path.
- **Modern User Interface**: A responsive, dark-mode dashboard built with React and Tailwind CSS.

## Technology Stack

### Frontend
- **Framework**: React (Vite)
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **State Management**: React Hooks (useState, useEffect)

### Backend
- **Server**: Flask (Python)
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Serialization**: Joblib

### Data Pipeline
- **Weather API**: Visual Crossing
- **Traffic API**: TomTom
- **ETL**: Custom Python scripts for extraction, transformation, and loading.

## Project Structure

```
Dynamic-Route-Rationalization-Model/
├── Backend/
│   └── app.py              # Flask API entry point serving the prediction endpoint
├── Frontend/
│   ├── src/                # React source code
│   ├── public/             # Static assets
│   ├── index.html          # HTML entry point
│   └── package.json        # Frontend dependencies and scripts
├── Model/
│   ├── finaaal.py          # Primary XGBoost training pipeline with Optuna optimization
│   └── model_xgboost.py    # Alternative training script with clustering logic
├── ModelArtifacts/
│   ├── best_model_xgb.joblib  # Trained XGBoost model
│   ├── route_id_map.json      # Mapping of Route IDs to Source/Destination pairs
│   ├── encoders.joblib        # Label encoders for categorical features
│   ├── scaler.joblib          # Standard scaler for numeric features
│   └── feature_columns.joblib # List of features used in training
├── Services/
│   ├── feature_extraction.py  # Script to fetch live weather and traffic data
│   └── clean_and_augment.py   # Data cleaning and augmentation pipeline
├── Data/                      # Directory for raw and processed datasets
├── inference.py               # Core inference logic used by the backend
└── README.md                  # Project documentation
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- npm (Node Package Manager)

### 1. Backend Setup

1.  Navigate to the project root directory.
2.  (Optional) Create and activate a virtual environment.
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
3.  Install the required Python packages.
    ```bash
    pip install flask flask-cors xgboost scikit-learn pandas numpy joblib
    ```
4.  Start the Flask server.
    ```bash
    python Backend/app.py
    ```
    The server will start running at `http://localhost:5000`.

### 2. Frontend Setup

1.  Open a new terminal window and navigate to the `Frontend` directory.
    ```bash
    cd Frontend
    ```
2.  Install the Node.js dependencies.
    ```bash
    npm install
    ```
3.  Start the development server.
    ```bash
    npm run dev
    ```
    The application will be accessible at `http://localhost:5173`.

## Usage Guide

1.  Ensure both the Backend and Frontend servers are running.
2.  Open your web browser and navigate to `http://localhost:5173`.
3.  **Input Parameters**:
    -   **Source & Destination**: Select your starting point and destination from the dropdown menus.
    -   **Weather Conditions**: Modify temperature, humidity, and other weather metrics if needed (defaults are provided).
    -   **Traffic & Time**: Adjust traffic metrics and time settings to simulate different scenarios.
4.  **Prediction**: Click the "Predict Optimal Route" button.
5.  **Results**:
    -   The dashboard will display the **Optimal Route ID**.
    -   The **Live Route Simulation** map will animate the path from your Source, through the predicted "Via" point (Intermediary), to the Destination.

## Model Architecture

The core prediction engine is built using **XGBoost**, a gradient boosting framework known for its performance and efficiency.

-   **Training**: The model is trained on a dataset combining historical route data with synthetic augmentations to cover edge cases.
-   **Optimization**: Hyperparameters are tuned using **Optuna** to maximize accuracy.
-   **Inference Flow**:
    1.  Input data is received via the API.
    2.  Categorical variables (Source, Destination) are encoded using saved LabelEncoders.
    3.  Numeric variables are scaled using a saved StandardScaler.
    4.  The XGBoost model predicts the class (Route ID).
    5.  The Route ID is mapped back to human-readable locations using `route_id_map.json`.

## License

This project is open-source and available under the MIT License.
