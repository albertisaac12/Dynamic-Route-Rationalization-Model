import pandas as pd
import numpy as np
import os

# File paths
INPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_final.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_final.csv")

# Load dataset
df = pd.read_csv(INPUT)

# Replace is_weekend with random 0/1
df["is_weekend"] = np.random.randint(0, 2, size=len(df))

# Save cleaned dataset
df.to_csv(OUTPUT, index=False)

print("Randomized 'is_weekend' column with 0/1 values.")
print("Remaining columns:", df.columns.tolist())
