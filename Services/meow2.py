import pandas as pd
import os

# File paths
INPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_final.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_final.csv")

# Load dataset
df = pd.read_csv(INPUT)

# Drop the precip column if it exists
if "precip" in df.columns:
    df = df.drop(columns=["precip"])

# Save cleaned dataset
df.to_csv(OUTPUT, index=False)

print("Dropped column: precip")
print("Remaining columns:", df.columns.tolist())
