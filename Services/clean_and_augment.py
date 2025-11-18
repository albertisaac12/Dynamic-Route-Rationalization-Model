"""
Clean + Augment Weather/Traffic Dataset
---------------------------------------
Fixes:
- missing weekday/is_weekend/traffic_pressure (date parsing issue)
- removes useless rows
- adds small synthetic variation for ML
- engineering new features

Output:
    Data/weather_traffic_clean_augmented.csv
"""

import os
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_by_route.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_augmented.csv")


# ---------------- MAIN ----------------
def main():
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT)

    # Ensure floats
    num_cols = [
        "traffic_distance_m", "traffic_duration_min", "no_traffic_duration_min",
        "temp", "tempmin", "tempmax", "humidity", "windspeed", "precip"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # 1. DROP USELESS ROWS
    # -------------------------
    print("🧹 Cleaning invalid rows...")

    before = len(df)

    # Remove rows without traffic values
    df = df.dropna(subset=["traffic_distance_m", "traffic_duration_min"])

    # Remove zero or negative values
    df = df[df["traffic_distance_m"] > 0]
    df = df[df["traffic_duration_min"] > 0]

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows where all routes of a destination are identical
    valid_groups = []
    for dest in df["destination"].unique():
        ddf = df[df["destination"] == dest]

        if (
            ddf["traffic_distance_m"].nunique() > 1 or
            ddf["traffic_duration_min"].nunique() > 1
        ):
            valid_groups.append(ddf)

    if valid_groups:
        df = pd.concat(valid_groups)

    print(f"   → Removed {before - len(df)} useless rows")


    # -------------------------
    # 2. AUGMENTATION
    # -------------------------
    print("🔧 Adding controlled synthetic noise...")

    # Traffic noise: ±1.5% on distance, ±2% on duration
    df["traffic_distance_m"] *= (1 + np.random.normal(0, 0.015, len(df)))
    df["traffic_duration_min"] *= (1 + np.random.normal(0, 0.02, len(df)))

    # Small weather noise
    df["temp"]      += np.random.normal(0, 0.2, len(df))
    df["humidity"]  += np.random.normal(0, 1.0, len(df))
    df["windspeed"] += np.random.normal(0, 0.5, len(df))

    # Clamp negative values
    df = df[df["traffic_distance_m"] > 0]
    df = df[df["traffic_duration_min"] > 0]


    # -------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------
    print("⚙️ Feature engineering...")

    # Fix date: "DD-MM-YYYY"
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

    # Weekday (0=Mon, 6=Sun)
    df["weekday"] = df["date"].dt.dayofweek

    # Weekend indicator
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Traffic pressure (simple composite index)
    df["traffic_pressure"] = (
        df["windspeed"] * 0.10 +
        df["humidity"] * 0.02 +
        df["traffic_duration_min"] * 0.30
    )

    # Efficiency: meters per minute
    df["traffic_efficiency"] = (
        df["traffic_distance_m"] / df["traffic_duration_min"]
    )

    # Month for seasonal insights
    df["month"] = df["date"].dt.month

    # Day of month
    df["day"] = df["date"].dt.day


    # -------------------------
    # 4. SAVE OUTPUT
    # -------------------------
    print("💾 Saving output...")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"✅ Done! Saved → {OUTPUT}")
    print("📊 Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
