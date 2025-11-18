import pandas as pd
import numpy as np
import os

# Paths
INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_clean_fixed.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_weekday_realistic.csv")

df = pd.read_csv(INPUT)

# Ensure day_of_week exists
if "day_of_week" not in df.columns:
    df["day_of_week"] = np.random.randint(1, 8, size=len(df))

# Weekend mapping
weekday_traffic_factor = {
    1: 1.35,  # Monday - highest traffic
    2: 1.25,  # Tuesday
    3: 1.20,  # Wednesday
    4: 1.25,  # Thursday
    5: 1.15,  # Friday
    6: 0.85,  # Saturday - lower traffic
    7: 0.75,  # Sunday - lowest traffic
}

# Weather influence weights
weather_weights = {
    "humidity": 0.05,         # 5% influence
    "windspeed": 0.02,        # 2% influence
    "temp": 0.01,             # 1% influence
    "precip": 0.10 if "precip" in df.columns else 0
}

# Apply day-based traffic realism
def apply_realism(row):
    dow = int(row["day_of_week"])
    base_factor = weekday_traffic_factor[dow]

    # Traffic duration
    new_duration = row["traffic_duration_min"] * base_factor

    # Add weather influence
    for wcol, weight in weather_weights.items():
        if wcol in row:
            new_duration += row[wcol] * weight

    # Add realistic noise
    new_duration *= np.random.uniform(0.95, 1.05)

    # Traffic pressure (scaled with duration)
    new_pressure = row["traffic_pressure"] * (base_factor + np.random.uniform(-0.05, 0.05))

    # Efficiency is inverse of duration
    new_eff = 1 / (new_duration + 1e-6)

    return pd.Series({
        "traffic_duration_min": new_duration,
        "traffic_pressure": new_pressure,
        "traffic_efficiency": new_eff
    })

# Apply realism transformations
realistic_updates = df.apply(apply_realism, axis=1)

# Update columns
df["traffic_duration_min"] = realistic_updates["traffic_duration_min"]
df["traffic_pressure"] = realistic_updates["traffic_pressure"]
df["traffic_efficiency"] = realistic_updates["traffic_efficiency"]

# Clean up numeric precision
df = df.round(6)

# Save final dataset
df.to_csv(OUTPUT, index=False)

print("🚀 Realistic weekday-based traffic applied successfully!")
print("Saved to:", OUTPUT)
