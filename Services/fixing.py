import pandas as pd
import numpy as np
import os

# File paths
INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_synthetic_augmented.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_fixed.csv")

# Load dataset
df = pd.read_csv(INPUT)

# -------------------------------
# 1. Fix day_of_week (missing → random 1–7)
# -------------------------------
if "day_of_week" in df.columns:
    missing_mask = df["day_of_week"].isna()
    df.loc[missing_mask, "day_of_week"] = np.random.randint(1, 8, size=missing_mask.sum())

    # Ensure integer 1–7
    df["day_of_week"] = df["day_of_week"].round().astype(int)

# -------------------------------
# 2. Fix is_weekend based on day_of_week
# -------------------------------
if "is_weekend" in df.columns:
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [6, 7] else 0)

# -------------------------------
# 3. Fix route_id
# -------------------------------
if "route_id" in df.columns:
    df["route_id"] = df["route_id"].astype(float)
    df["route_id"] = df["route_id"].fillna(method='ffill').fillna(method='bfill')

# -------------------------------
# 4. Fill numeric NaNs with column mean
# -------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# -------------------------------
# 5. Save final cleaned dataset
# -------------------------------
df.to_csv(OUTPUT, index=False)

print("✨ Dataset inconsistencies fixed successfully!")
print("Saved to:", OUTPUT)
print("\nSummary:")
print("- Missing day_of_week fixed")
print("- is_weekend aligned with day_of_week")
print("- route_id cleaned and type-enforced")
print("- Numeric NaNs filled with column means")
print("- Dataset now ready for training 🚀")
