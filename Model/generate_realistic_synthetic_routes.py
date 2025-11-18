# generate_realistic_hd_routes_B.py
import numpy as np
import pandas as pd
import os
from pathlib import Path
import math
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.joinpath("..", "Data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Config
# ---------------------------
SAMPLES_PER_ROUTE = 200  # ~132 * 200 = 26,400 rows total
CONDITION_PROBS = [0.50, 0.30, 0.15, 0.05]  # probabilities for 0=Clear,1=Cloudy,2=Rainy,3=Storm

# 44 realistic Hyderabad-area O-D pairs (from previous list)
PAIRS = [
    ("Hyderabad","Gachibowli"),
    ("Hyderabad","Madhapur"),
    ("Hyderabad","Banjara Hills"),
    ("Hyderabad","Secunderabad"),
    ("Hyderabad","Begumpet"),
    ("Hyderabad","Kukatpally"),
    ("Hyderabad","Uppal"),
    ("Hyderabad","LB Nagar"),
    ("Hyderabad","Dilsukhnagar"),
    ("Hyderabad","Hitech City"),
    ("Gachibowli","Madhapur"),
    ("Gachibowli","Banjara Hills"),
    ("Gachibowli","Airport"),
    ("Gachibowli","Kondapur"),
    ("Gachibowli","Financial District"),
    ("Secunderabad","Hitech City"),
    ("Secunderabad","Kukatpally"),
    ("Secunderabad","Banjara Hills"),
    ("Secunderabad","Gachibowli"),
    ("Secunderabad","Begumpet"),
    ("Uppal","Hitech City"),
    ("Uppal","Gachibowli"),
    ("Uppal","Secunderabad"),
    ("Uppal","Financial District"),
    ("Madhapur","Jubilee Hills"),
    ("Madhapur","Airport"),
    ("Madhapur","BHEL"),
    ("Madhapur","KPHB"),
    ("KPHB","Hitech City"),
    ("KPHB","Begumpet"),
    ("KPHB","Airport"),
    ("LB Nagar","Gachibowli"),
    ("LB Nagar","Hitech City"),
    ("LB Nagar","Banjara Hills"),
    ("Dilsukhnagar","Airport"),
    ("Dilsukhnagar","Hitech City"),
    ("Dilsukhnagar","Gachibowli"),
    ("Kondapur","Financial District"),
    ("Kondapur","Jubilee Hills"),
    ("Airport","Gachibowli"),
    ("Airport","Banjara Hills"),
    ("Airport","Madhapur"),
    ("Jubilee Hills","Banjara Hills"),
    ("Jubilee Hills","Kondapur"),
]

assert len(PAIRS) == 44

# ---------------------------
# Route variant generator
# ---------------------------
# We'll create 3 variants per pair:
#  A = fastest route (shorter distance / higher avg speed)
#  B = balanced route
#  C = longer but stable (slightly longer distance, less variance)
VARIANTS = ["A", "B", "C"]

def base_distance_and_duration(pair_idx, variant):
    """
    Generate base distance_km and base_duration_min for given pair index and variant.
    We use a seed based deterministic approach so base values differ by pair and variant.
    For 'B' (mixed option), distances range ~5–35 km.
    """
    # base distance by pair_idx spread in 5..35
    min_km, max_km = 5.0, 35.0
    t = (pair_idx % len(PAIRS)) / max(1, len(PAIRS)-1)
    # vary base distance by pair index and add some jitter
    pair_base = min_km + t * (max_km - min_km)
    # variant adjustments
    if variant == "A":  # fastest, slightly shorter
        dist = pair_base * np.random.uniform(0.92, 0.98)
        speed_kmph = np.random.uniform(35, 55)  # faster average speed
    elif variant == "B":  # balanced
        dist = pair_base * np.random.uniform(0.98, 1.06)
        speed_kmph = np.random.uniform(25, 40)
    else:  # C - longer/stable
        dist = pair_base * np.random.uniform(1.06, 1.25)
        speed_kmph = np.random.uniform(18, 32)
    # ensure distance within bounds
    dist = float(np.clip(dist, min_km, max_km))
    # duration in minutes = distance_km / speed_kmph * 60
    duration_min = float(max(5.0, (dist / speed_kmph) * 60.0))
    return dist, duration_min, speed_kmph

def is_peak_hour(hour):
    return 1 if (7 <= hour <= 10 or 17 <= hour <= 20) else 0

def sample_weather():
    return np.random.choice([0,1,2,3], p=CONDITION_PROBS)

# weather sensitivity per route variant (multiplier per severity step)
# e.g., severity 0..3 -> multiplier 1.00, 1.05, 1.12, 1.30 (varies by route)
def weather_multiplier_by_variant(variant):
    if variant == "A":
        return {0:1.00, 1:1.04, 2:1.08, 3:1.15}
    elif variant == "B":
        return {0:1.00, 1:1.07, 2:1.14, 3:1.25}
    else:
        return {0:1.00, 1:1.10, 2:1.20, 3:1.40}

def peak_multiplier_by_variant(variant):
    if variant == "A":
        return 1.10
    elif variant == "B":
        return 1.25
    else:
        return 1.40

# traffic_pressure baseline factor per variant
pressure_base_by_variant = {"A": 8.0, "B": 12.0, "C": 18.0}  # arbitrary baseline that scales with congestion

# ---------------------------
# Generate dataset
# ---------------------------
rows = []
route_map = []
route_id = 0

for pair_idx, (src, dst) in enumerate(PAIRS):
    for variant in VARIANTS:
        route_id += 1
        # compute base properties
        base_km, base_duration_min, base_speed = base_distance_and_duration(pair_idx, variant)
        # small route-specific randomness for base
        base_km = round(base_km + np.random.normal(0, 0.2), 3)
        base_duration_min = float(round(base_duration_min + np.random.normal(0, 1.2), 3))
        route_map.append({
            "route_id": route_id,
            "pair_idx": pair_idx+1,
            "source": src,
            "destination": dst,
            "variant": variant,
            "base_distance_km": base_km,
            "base_duration_min": base_duration_min,
            "base_speed_kmph": round(base_speed, 2)
        })

        wmult = weather_multiplier_by_variant(variant)
        p_mult = peak_multiplier_by_variant(variant)
        pressure_base = pressure_base_by_variant[variant]

        for _ in range(SAMPLES_PER_ROUTE):
            # time features
            hour = np.random.randint(0, 24)
            day = np.random.randint(1, 8)  # 1..7
            is_peak = is_peak_hour(hour)
            is_weekend = 1 if day in (6,7) else 0

            # weather
            cond = int(sample_weather())

            # compute duration: start from base, apply multipliers
            duration = base_duration_min
            duration *= wmult[cond]                 # weather effect
            if is_peak:
                duration *= p_mult                  # peak-hour multiplier
            # weekend slightly lighter
            if is_weekend:
                duration *= 0.92
            # add stochastic noise
            duration *= np.random.normal(1.0, 0.05)  # 5% variation
            duration = max(2.0, float(duration))

            # distance: base_km + small jitter
            dist_km = max(0.5, float(np.random.normal(base_km, 0.15)))
            # convert to meters
            traffic_distance_m = dist_km * 1000.0

            # compute efficiency: km per minute
            traffic_efficiency = dist_km / duration

            # traffic_pressure: function of duration per km and condition and peak
            pressure = (duration / max(0.1, dist_km)) * (1 + 0.25*cond) * (1 + 0.2*is_peak)
            # scale pressure to a human-friendly range
            traffic_pressure = float(round(pressure * pressure_base / 5.0, 3))

            # temperatures realistic for Hyderabad
            temp = round(np.random.uniform(23, 36) + np.random.normal(0, 1.0), 2)
            tempmin = round(temp - np.random.uniform(2, 6), 2)
            tempmax = round(temp + np.random.uniform(1, 4), 2)
            humidity = round(np.random.uniform(40, 90) + np.random.normal(0, 3.0), 2)
            windspeed = round(abs(np.random.normal(6, 4)), 2)

            rows.append({
                "route_id": route_id,
                "source": src,
                "destination": dst,
                "variant": variant,
                "traffic_distance_m": round(traffic_distance_m, 2),
                "traffic_duration_min": round(duration, 3),
                "temp": temp,
                "tempmin": tempmin,
                "tempmax": tempmax,
                "humidity": humidity,
                "windspeed": windspeed,
                "conditions": cond,         # 0..3
                "is_weekend": is_weekend,
                "traffic_pressure": traffic_pressure,
                "traffic_efficiency": round(traffic_efficiency, 6),
                "day_of_week": int(day),
                "hour_of_day": int(hour),
            })

# convert to DataFrame
df_out = pd.DataFrame(rows)
df_map = pd.DataFrame(route_map)

# shuffle dataset
df_out = df_out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

# save files
main_file = OUT_DIR.joinpath("realistic_hd_routes_B_dataset.csv")
map_file  = OUT_DIR.joinpath("realistic_hd_routes_B_route_id_map.csv")

df_out.to_csv(main_file, index=False)
df_map.to_csv(map_file, index=False)

print("Saved dataset to:", main_file)
print("Saved route map to:", map_file)
print("Rows:", df_out.shape[0], "Unique route_ids:", df_out['route_id'].nunique())
print(df_out.head(6))
