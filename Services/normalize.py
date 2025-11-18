import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# File paths
INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_clean_final.csv")
OUTPUT = os.path.join(os.pardir, "Data", "weather_traffic_clean_final_normalized.csv")

# Load dataset
df = pd.read_csv(INPUT)

# ------------------------------
# 1) Encode 'conditions' to numeric
# ------------------------------
if "conditions" in df.columns:
    le = LabelEncoder()
    df["conditions"] = le.fit_transform(df["conditions"].astype(str))
    print("Encoded 'conditions' mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ------------------------------
# 2) Convert route_id to float (but don't normalize it)
# ------------------------------
df["route_id"] = df["route_id"].astype(float)

# ------------------------------
# 3) Normalize numeric columns EXCEPT route_id
# ------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude route_id from normalization
numeric_cols.remove("route_id")

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ------------------------------
# 4) Save output
# ------------------------------
df.to_csv(OUTPUT, index=False)

print("route_id converted to float and left unnormalized.")
print("Normalized columns:", numeric_cols)
print("Saved normalized dataset to:", OUTPUT)
