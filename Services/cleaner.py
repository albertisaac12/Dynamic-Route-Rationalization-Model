"""
Duplicate Row Cleaner
----------------------
✅ Loads weather_traffic_aggregated.csv
✅ Drops duplicate rows (based on all columns)
✅ Saves cleaned dataset to Data/weather_traffic_cleaned.csv
"""

import pandas as pd
import os

# Input and output file paths
INPUT_FILE = os.path.join(os.pardir, "Data", "weather_traffic_aggregated.csv")
OUTPUT_FILE = os.path.join(os.pardir, "Data", "weather_traffic_cleaned.csv")

def main():
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    before = len(df)
    print(f"Total rows before cleaning: {before}")

    # Drop exact duplicates
    df_clean = df.drop_duplicates()

    # Optional: drop near-duplicates ignoring 'date' column
    # df_clean = df.drop_duplicates(subset=[col for col in df.columns if col != "date"])

    after = len(df_clean)
    print(f"✅ Rows after dropping duplicates: {after}")
    print(f"🧮 Removed {before - after} duplicate rows.")

    # Save cleaned data
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Cleaned dataset saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
