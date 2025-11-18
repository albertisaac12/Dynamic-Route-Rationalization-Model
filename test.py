"""
Test script for Visual Crossing + TomTom APIs
----------------------------------------------
Purpose:
- Verify API keys
- Inspect JSON response structure (no large-scale calls)
- Print only sample values
"""

import requests
from datetime import datetime

# ---- Replace with your own API keys ----
VISUAL_KEY = "Z3XKFBDX6WY8RN3MD4B8PUXPK"   # get from https://www.visualcrossing.com/sign-up
TOMTOM_KEY = "jkm7zvIhnQIW5NcVRz0CSz0TA5VJqf1B"

# ---- Sample Coordinates (Hyderabad) ----
LAT, LON = 17.3850, 78.4867

# ---- API Endpoints ----
VISUAL_URL = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LAT},{LON}"
TOMTOM_URL = "https://api.tomtom.com/routing/1/calculateRoute/{start}:{end}/json"


# ---- TEST WEATHER (Visual Crossing) ----
def test_weather():
    print("\n🌦️ Testing Visual Crossing Weather API ...")
    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": VISUAL_KEY,
        "contentType": "json"
    }

    try:
        r = requests.get(VISUAL_URL, params=params, timeout=10)
        print(f"➡️ Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print("✅ Weather API call succeeded.")
            print(f"Location: {data.get('resolvedAddress', 'Unknown')}")
            print(f"Days returned: {len(data.get('days', []))}")
            if data.get("days"):
                first = data["days"][0]
                print("\n--- Sample Day ---")
                print(f"Date: {first.get('datetime')}")
                print(f"Temp: {first.get('temp', 'N/A')} °C")
                print(f"Humidity: {first.get('humidity', 'N/A')} %")
                print(f"Wind Speed: {first.get('windspeed', 'N/A')} km/h")
                print(f"Conditions: {first.get('conditions', 'N/A')}")
        else:
            print("❌ Weather API error:", r.text)
    except Exception as e:
        print("⚠️ Weather API test failed:", e)


# ---- TEST TRAFFIC (TomTom) ----
def test_traffic():
    print("\n🚗 Testing TomTom Routing + Traffic API ...")
    start_lat, start_lon = 17.3850, 78.4867
    end_lat, end_lon = 17.4500, 78.4000
    start = f"{start_lat},{start_lon}"
    end = f"{end_lat},{end_lon}"

    url = TOMTOM_URL.format(start=start, end=end)
    params = {
        "key": TOMTOM_KEY,
        "traffic": "true"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"➡️ Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            summary = data["routes"][0]["summary"]
            print("✅ Traffic API call succeeded.")
            print("\n--- Route Summary ---")
            print(f"Distance: {summary['lengthInMeters']} meters")
            print(f"Travel Time (with traffic): {summary['travelTimeInSeconds'] / 60:.2f} min")
            if "noTrafficTravelTimeInSeconds" in summary:
                print(f"No Traffic Time: {summary['noTrafficTravelTimeInSeconds'] / 60:.2f} min")
            else:
                print("⚠️ 'noTrafficTravelTimeInSeconds' not provided in response.")
        else:
            print("❌ Traffic API error:", r.text)
    except Exception as e:
        print("⚠️ Traffic API test failed:", e)


# ---- MAIN ----
if __name__ == "__main__":
    print("🔍 Running API tests safely (no bulk requests)...")
    test_weather()
    test_traffic()
    print("\n✅ Done! If both succeed, you're ready to run the full collector.")
