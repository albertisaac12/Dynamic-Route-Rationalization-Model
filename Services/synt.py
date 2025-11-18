import pandas as pd
import numpy as np
import os

# File paths
INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_clean_final_normalized.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_synthetic_augmented.csv")

# How many synthetic rows per original row
SYN_FACTOR = 3  # adjust as needed

# Load dataset
df = pd.read_csv(INPUT)

# Columns that must remain EXACT
LOCKED_COLS = ["source", "destination", "route_id"]

# Identify numeric and categorical columns (remove locked ones)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in LOCKED_COLS]

categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in LOCKED_COLS]

synthetic_rows = []

# ------------------------------------------
# Create synthetic rows
# ------------------------------------------
for idx, row in df.iterrows():

    for _ in range(SYN_FACTOR):

        new_row = {}

        # Keep locked columns fixed
        for col in LOCKED_COLS:
            new_row[col] = row[col]

        # Add the new day_of_week column (random 1 to 7)
        new_row["day_of_week"] = np.random.randint(1, 8)

        # Synthetic numeric features (small Gaussian noise)
        for col in numeric_cols:
            mean = row[col]
            noise = np.random.normal(0, 0.05)  # 5% noise
            new_val = mean + noise
            new_val = np.clip(new_val, 0, 1)   # keep normalized range
            new_row[col] = new_val

        # Synthetic categorical features
        for col in categorical_cols:
            new_row[col] = np.random.choice(df[col].unique())

        synthetic_rows.append(new_row)

# Convert synthetic rows into a DataFrame
df_synth = pd.DataFrame(synthetic_rows)

# ------------------------------------------
# Combine original + synthetic data
# ------------------------------------------
df_final = pd.concat([df, df_synth], ignore_index=True)

# Save
df_final.to_csv(OUTPUT, index=False)

print("Synthetic augmentation complete.")
print("Locked columns unchanged:", LOCKED_COLS)
print("Added column: day_of_week (1–7)")
print("Original rows:", len(df))
print("Synthetic rows added:", len(df_synth))
print("Final dataset saved to:", OUTPUT)
print("Final size:", df_final.shape)
