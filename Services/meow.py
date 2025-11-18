import pandas as pd
import numpy as np

INPUT = r"C:\Dynaim Route Rationalization\Data\weather_traffic_clean_augmented.csv"
OUTPUT = r"C:\Dynaim Route Rationalization\Data\weather_traffic_cleaned.csv"

def clean_route_data(path):
    df = pd.read_csv(path)

    # --------------------------------------------------
    # 1. Drop fully NaN columns
    # --------------------------------------------------
    df = df.dropna(axis=1, how='all')

    # --------------------------------------------------
    # 2. Drop ONLY useless columns
    # --------------------------------------------------
    drop_cols = ['positions_sampled', 'month']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # --------------------------------------------------
    # 3. Parse datetime if exists
    # --------------------------------------------------
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['dayofweek'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour

    # --------------------------------------------------
    # 4. Ensure numeric columns are numeric
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # --------------------------------------------------
    # 5. Fill missing important numeric values
    # --------------------------------------------------
    weather_cols = [
        'temp','tempmin','tempmax','humidity',
        'windspeed','precip','uvindex','snow',
        'traffic_pressure','traffic_efficiency'
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    if 'traffic_duration_min' in df.columns:
        df['traffic_duration_min'] = df['traffic_duration_min'].fillna(
            df['traffic_duration_min'].median()
        )

    # --------------------------------------------------
    # 6. Encode categorical conditions
    # --------------------------------------------------
    if 'conditions' in df.columns:
        df = pd.get_dummies(df, columns=['conditions'], prefix='cond')

    # --------------------------------------------------
    # 7. Reset the day column → cycle 1–7
    # --------------------------------------------------
    df = df.drop(columns=['day'], errors='ignore')
    df['day'] = (df.index % 7) + 1

    # --------------------------------------------------
    # 8. Drop rows with too many missing values
    # --------------------------------------------------
    df = df.dropna(thresh=int(df.shape[1] * 0.7))

    return df


# Run cleaning
clean_df = clean_route_data(INPUT)

# Save cleaned file
clean_df.to_csv(OUTPUT, index=False)

print(clean_df.head())
print("\nTotal Rows:", len(clean_df))
print("Total Columns:", len(clean_df.columns))
print(f"\nSaved cleaned file to: {OUTPUT}")
