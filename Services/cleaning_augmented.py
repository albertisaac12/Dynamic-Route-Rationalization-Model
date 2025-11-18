import pandas as pd
import os

# File paths
INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_clean_augmented.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_final.csv")

# Load dataset
df = pd.read_csv(INPUT)

# 1) Drop all empty columns (columns with all NaN values)
df = df.dropna(axis=1, how='all')

# 2) Drop the positions_sampled column if it exists
if "positions_sampled" in df.columns:
    df = df.drop(columns=["positions_sampled"])

# Save cleaned dataset
df.to_csv(OUTPUT, index=False)

print("Cleaning completed successfully!")
print("Cleaned file saved to:", OUTPUT)
print("Remaining columns:", df.columns.tolist())
