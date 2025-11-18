"""
Weather + Traffic Collector (ROUTE-AWARE)
-----------------------------------------
Fixes the issue where all routes had identical traffic results.

Now:
- Loads raw_routes.csv
- Loads OSM graph to convert node_sequence → lat/lon
- Samples 8 waypoints from each route
- TomTom computes traffic for ACTUAL GEOMETRY
- Attach Visual Crossing 15-day weather
- Outputs: Data/weather_traffic_by_route.csv
"""

import os
import ast
import time
import requests
import pandas as pd
import osmnx as ox
import networkx as nx

# ---------------- CONFIG ----------------
RAW_ROUTES_FILE = os.path.join(os.pardir, "Data", "raw_routes.csv")
OUTPUT_FILE     = os.path.join(os.pardir, "Data", "weather_traffic_by_route.csv")

VISUAL_KEY = "Z3XKFBDX6WY8RN3MD4B8PUXPK"
TOMTOM_KEY = "jkm7zvIhnQIW5NcVRz0CSz0TA5VJqf1B"

HYD = (17.3850, 78.4867)  # Graph center
GRAPH_DIST = 15000        # 15 km radius like before
MAX_WAYPOINTS = 8

VC_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
TOMTOM_URL = "https://api.tomtom.com/routing/1/calculateRoute/{points}/json"


# ---------------- HELPERS ----------------
def convert_mdigraph_to_digraph(G_multi):
    """Convert MultiDiGraph to DiGraph keeping shortest edges & CRS."""
    G = nx.DiGraph()

    for n, attrs in G_multi.nodes(data=True):
        G.add_node(n, **attrs)

    for u, v, data in G_multi.edges(data=True):
        ln = data.get("length", float("inf"))
        if G.has_edge(u, v):
            if ln < G[u][v]["length"]:
                G[u][v].clear()
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)

    G.graph.update(G_multi.graph.copy())
    return G


def fetch_weather(lat, lon):
    """Visual Crossing 15-day forecast."""
    url = f"{VC_URL}/{lat},{lon}"
    params = {
        "key": VISUAL_KEY,
        "unitGroup": "metric",
        "include": "days",
        "contentType": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        days = r.json().get("days", [])

        out = []
        for d in days:
            out.append({
                "date": d.get("datetime"),
                "temp": d.get("temp"),
                "tempmin": d.get("tempmin"),
                "tempmax": d.get("tempmax"),
                "humidity": d.get("humidity"),
                "windspeed": d.get("windspeed"),
                "precip": d.get("precip"),
                "conditions": d.get("conditions")
            })

        return out

    except Exception as e:
        print(f"⚠️ Weather error @ {lat},{lon}: {e}")
        return []


def sample_waypoints(G, node_sequence):
    """Convert node_sequence to ≤8 lat/lon points."""
    nodes = [n for n in node_sequence if n in G.nodes]

    if len(nodes) <= MAX_WAYPOINTS:
        chosen = nodes
    else:
        chosen = [nodes[0]]
        step = (len(nodes) - 2) / float(MAX_WAYPOINTS - 2)
        for i in range(1, MAX_WAYPOINTS - 1):
            idx = int(round(i * step))
            chosen.append(nodes[idx])
        chosen.append(nodes[-1])

    coords = []
    for n in chosen:
        data = G.nodes[n]
        coords.append((data["y"], data["x"]))  # lat, lon

    return coords


def coords_to_tomtom_format(coords):
    """lat,lon:lat,lon:lat,lon"""
    return ":".join(f"{lat},{lon}" for lat, lon in coords)


def fetch_traffic_for_route(coords):
    """TomTom route call with waypoints."""
    if len(coords) < 2:
        return {"traffic_distance_m": None, "traffic_duration_min": None}

    point_str = coords_to_tomtom_format(coords)
    url = TOMTOM_URL.format(points=point_str)

    params = {
        "key": TOMTOM_KEY,
        "traffic": "true"
    }

    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()

        summary = r.json()["routes"][0]["summary"]

        return {
            "traffic_distance_m": summary.get("lengthInMeters"),
            "traffic_duration_min": summary.get("travelTimeInSeconds", 0) / 60,
            "no_traffic_duration_min": summary.get("noTrafficTravelTimeInSeconds")
        }

    except Exception as e:
        print(f"⚠️ Traffic error on route: {e}")
        return {
            "traffic_distance_m": None,
            "traffic_duration_min": None,
            "no_traffic_duration_min": None
        }


# ---------------- MAIN ----------------
def main():
    print("📥 Loading raw routes...")
    df = pd.read_csv(RAW_ROUTES_FILE)
    df["node_sequence"] = df["node_sequence"].apply(ast.literal_eval)

    print("🌐 Loading OSM graph...")
    G_multi = ox.graph_from_point(HYD, dist=GRAPH_DIST, network_type="drive")
    if "crs" not in G_multi.graph:
        G_multi = ox.project_graph(G_multi)

    G = convert_mdigraph_to_digraph(G_multi)

    all_rows = []

    for _, row in df.iterrows():
        src = row["source"]
        dst = row["destination"]
        rid = row["route_id"]
        seq = row["node_sequence"]

        print(f"➡️ Route {rid}: {src} → {dst}")

        # 1. Waypoints
        coords = sample_waypoints(G, seq)

        # 2. Traffic for THIS geometry
        traffic = fetch_traffic_for_route(coords)

        # 3. Weather at destination
        dest_lat, dest_lon = coords[-1]
        weather = fetch_weather(dest_lat, dest_lon)
        if not weather:
            continue

        # 4. Join all weather entries with traffic
        for w in weather:
            all_rows.append({
                "source": src,
                "destination": dst,
                "route_id": rid,
                "positions_sampled": len(coords),
                **traffic,
                **w
            })

        time.sleep(0.25)

    df_out = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Done. Saved {len(df_out)} rows → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
