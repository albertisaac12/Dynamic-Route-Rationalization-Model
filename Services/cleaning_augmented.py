import pandas as pd
import os

INPUT  = os.path.join(os.pardir, "Data", "weather_traffic_clean_augmented.csv")

df = pd.read_csv(INPUT)

print(df.head(10))
