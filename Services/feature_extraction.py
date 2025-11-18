"""
Weather + Traffic Data Collector
---------------------------------
✅ Fetches Visual Crossing 15-day weather forecast (daily)
✅ Fetches TomTom live traffic data (distance, travel time)
✅ Combines both for all routes from raw_routes.csv
✅ Uses real coordinates for Hyderabad & destinations
✅ Outputs -> Data/weather_traffic_aggregated.csv
"""

import os
import ast
import time
import requests
import pandas as pd

# ---------------- CONFIG ----------------
RAW_ROUTES_FILE = os.path.join(os.pardir, "Data", "raw_routes.csv")
OUTPUT_FILE = os.path.join(os.pardir, "Data", "weather_traffic_aggregated.csv")

# API KEYS
VISUAL_KEY = "Z3XKFBDX6WY8RN3MD4B8PUXPK"
TOMTOM_KEY = "jkm7zvIhnQIW5NcVRz0CSz0TA5VJqf1B"

# Hyderabad hub coordinates
HYDERABAD = (17.3850, 78.4867)

# Destination coordinates
DESTINATIONS = {
    "Gachibowli":    (17.4401, 78.3489),
    "Charminar":     (17.3616, 78.4747),
    "Kukatpally":    (17.4933, 78.4070),
    "LB Nagar":      (17.3000, 78.5600),
    "Miyapur":       (17.4957, 78.3615),
    "Uppal":         (17.3961, 78.5581),
    "Kompally":      (17.5470, 78.4984),
    "Shamshabad":    (17.2551, 78.4002),
    "HITEC City":    (17.4483, 78.3915),
    "Secunderabad":  (17.4399, 78.4983)
}

# API ENDPOINTS
VISUAL_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
TOMTOM_URL = "https://api.tomtom.com/routing/1/calculateRoute/{start}:{end}/json"


# ---------------- WEATHER ----------------
def fetch_weather(lat, lon):
    """Fetch 15-day daily weather forecast using Visual Crossing."""
    url = f"{VISUAL_URL}/{lat},{lon}"
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": VISUAL_KEY,
        "contentType": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        forecasts = []
        for day in data.get("days", []):
            forecasts.append({
                "date": day.get("datetime"),
                "temp": day.get("temp"),
                "tempmin": day.get("tempmin"),
                "tempmax": day.get("tempmax"),
                "humidity": day.get("humidity"),
                "windspeed": day.get("windspeed"),
                "precip": day.get("precip"),
                "conditions": day.get("conditions")
            })
        return forecasts
    except Exception as e:
        print(f"⚠️ Weather fetch failed for {lat},{lon}: {e}")
        return []


# ---------------- TRAFFIC ----------------
def fetch_traffic(start_lat, start_lon, end_lat, end_lon):
    """Fetch live traffic distance and travel time from TomTom."""
    start = f"{start_lat},{start_lon}"
    end = f"{end_lat},{end_lon}"
    url = TOMTOM_URL.format(start=start, end=end)
    params = {"key": TOMTOM_KEY, "traffic": "true"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        summary = data["routes"][0]["summary"]
        return {
            "traffic_distance_m": summary.get("lengthInMeters"),
            "traffic_duration_min": summary.get("travelTimeInSeconds", 0) / 60,
            "no_traffic_duration_min": summary.get("noTrafficTravelTimeInSeconds", None)
        }
    except Exception as e:
        print(f"⚠️ Traffic fetch failed for {start} → {end}: {e}")
        return {
            "traffic_distance_m": None,
            "traffic_duration_min": None,
            "no_traffic_duration_min": None
        }


# ---------------- MAIN ----------------
def main():
    print("📂 Loading route data...")
    df_routes = pd.read_csv(RAW_ROUTES_FILE)
    df_routes["node_sequence"] = df_routes["node_sequence"].apply(ast.literal_eval)

    all_records = []

    for _, row in df_routes.iterrows():
        src = row["source"]
        dest = row["destination"]
        rid = row["route_id"]

        # Coordinates for Hyderabad and destination
        start_lat, start_lon = HYDERABAD
        if dest not in DESTINATIONS:
            print(f"⚠️ Destination '{dest}' not found in DESTINATIONS — skipping.")
            continue
        end_lat, end_lon = DESTINATIONS[dest]

        print(f"\n🌤 Fetching weather + traffic for route {rid} ({src} → {dest})")

        # 1️⃣ Weather at destination
        weather_data = fetch_weather(end_lat, end_lon)
        if not weather_data:
            print("⚠️ Skipping route due to missing weather data.")
            continue

        # 2️⃣ Live traffic
        traffic_data = fetch_traffic(start_lat, start_lon, end_lat, end_lon)

        # 3️⃣ Combine
        for w in weather_data:
            record = {
                "route_id": rid,
                "source": src,
                "destination": dest,
                **w,
                **traffic_data
            }
            all_records.append(record)
            time.sleep(0.2)

    # Save final dataset
    df_out = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved {len(df_out)} combined weather+traffic records → {OUTPUT_FILE}")
    print(df_out.head())


if __name__ == "__main__":
    main()
