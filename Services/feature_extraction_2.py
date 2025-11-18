"""
Weather + Traffic Data Collector (per-route TomTom calls)
---------------------------------------------------------
- Uses Visual Crossing for 15-day daily weather (per destination)
- Uses TomTom routing with intermediate waypoints sampled from each route's node_sequence
  so TomTom evaluates the *actual route geometry*, not just start/end pair.
- Requires: raw_routes.csv (with node_sequence of OSM node IDs), OSMnx installed and able to download the Hyderabad graph
- Outputs -> Data/weather_traffic_by_route.csv
"""

import os
import ast
import time
import requests
import pandas as pd
import osmnx as ox
import networkx as nx
from math import ceil

# ---------------- CONFIG ----------------
RAW_ROUTES_FILE = os.path.join(os.pardir, "Data", "raw_routes.csv")
OUTPUT_FILE = os.path.join(os.path.pardir, "Data", "weather_traffic_by_route.csv")

VISUAL_KEY = ""   # Visual Crossing
TOMTOM_KEY = ""  # TomTom

# Hyderabad graph center (used to download the graph to get node coords)
HYDERABAD = (17.3850, 78.4867)
DISTANCE_GRAPH_METERS = 15000  # same radius used before

# Endpoints
VISUAL_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
TOMTOM_ROUTE_URL = "https://api.tomtom.com/routing/1/calculateRoute/{positions}/json"

# Maximum number of waypoints to include in TomTom request (including start & end)
MAX_WAYPOINTS = 8  # tune as needed (TomTom allows many, but long URLs might break)


# ---------------- HELPERS ----------------
def convert_to_digraph_preserve_graph_attrs(G_multi):
    G = nx.DiGraph()
    for n, attrs in G_multi.nodes(data=True):
        G.add_node(n, **attrs)
    for u, v, data in G_multi.edges(data=True):
        length = data.get("length", float("inf"))
        if G.has_edge(u, v):
            if length < G[u][v].get("length", float("inf")):
                G[u][v].clear()
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)
    if hasattr(G_multi, "graph") and isinstance(G_multi.graph, dict):
        G.graph.update(G_multi.graph.copy())
    return G


def fetch_weather(lat, lon):
    url = f"{VISUAL_URL}/{lat},{lon}"
    params = {"unitGroup": "metric", "include": "days", "key": VISUAL_KEY, "contentType": "json"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        payload = r.json()
        days = payload.get("days", [])
        forecasts = [{
            "date": d.get("datetime"),
            "temp": d.get("temp"),
            "tempmin": d.get("tempmin"),
            "tempmax": d.get("tempmax"),
            "humidity": d.get("humidity"),
            "windspeed": d.get("windspeed"),
            "precip": d.get("precip"),
            "conditions": d.get("conditions")
        } for d in days]
        return forecasts
    except Exception as e:
        print(f"⚠️ Weather fetch failed for {lat},{lon}: {e}")
        return []


def sample_waypoints_from_node_sequence(G, node_seq, max_points=MAX_WAYPOINTS):
    """
    Convert a node_sequence (list of node IDs) to a sampled list of (lat,lon) pairs.
    Keeps first and last nodes, samples intermediate nodes uniformly to maintain route shape.
    """
    # Filter nodes that actually exist in graph
    nodes = [n for n in node_seq if n in G.nodes]
    if len(nodes) == 0:
        return []

    # If nodes count <= max_points, use them directly (but ensure unique)
    if len(nodes) <= max_points:
        chosen = nodes
    else:
        # Always include first and last
        chosen = [nodes[0]]
        # choose intermediate indices uniformly
        num_intermediate = max_points - 2
        step = (len(nodes) - 2) / float(num_intermediate + 1)
        for i in range(1, num_intermediate + 1):
            idx = int(round(i * step))
            idx = max(1, min(len(nodes) - 2, idx))
            chosen.append(nodes[idx])
        chosen.append(nodes[-1])

    # map node ids to lat,lon with correct order
    coords = []
    for n in chosen:
        node_data = G.nodes[n]
        lat = node_data.get("y")
        lon = node_data.get("x")
        if lat is None or lon is None:
            # skip if no coords
            continue
        coords.append((lat, lon))
    # deduplicate consecutive duplicates
    dedup = []
    prev = None
    for c in coords:
        if c != prev:
            dedup.append(c)
        prev = c
    return dedup


def tomtom_route_positions_string(coords):
    """
    Given [(lat,lon), ...] produce the positions path parameter string for TomTom:
    pos1:pos2:pos3 where each pos is "lat,lon" (TomTom accepts lat,lon).
    """
    # Ensure at least start and end exist
    if len(coords) < 2:
        return None
    parts = [f"{lat},{lon}" for lat, lon in coords]
    return ":".join(parts)


def fetch_traffic_for_route_positions(positions_str):
    """
    Call TomTom with positions_str built from waypoints. Returns summary dict.
    """
    if not positions_str:
        return {"traffic_distance_m": None, "traffic_duration_min": None, "no_traffic_duration_min": None}
    url = TOMTOM_ROUTE_URL.format(positions=positions_str)
    params = {"key": TOMTOM_KEY, "traffic": "true"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        summary = data.get("routes", [])[0].get("summary", {})
        return {
            "traffic_distance_m": summary.get("lengthInMeters"),
            "traffic_duration_min": summary.get("travelTimeInSeconds", 0) / 60.0,
            "no_traffic_duration_min": summary.get("noTrafficTravelTimeInSeconds", None)
        }
    except Exception as e:
        # print small message and return Nones so we can continue collecting other routes
        print(f"⚠️ TomTom request failed for positions [{positions_str[:200]}...]: {e}")
        return {"traffic_distance_m": None, "traffic_duration_min": None, "no_traffic_duration_min": None}


# ---------------- MAIN ----------------
def main():
    print("📂 Loading raw routes...")
    df = pd.read_csv(RAW_ROUTES_FILE)
    # node_sequence should be list-like strings, convert
    df["node_sequence"] = df["node_sequence"].apply(ast.literal_eval)

    print("📡 Downloading local OSMnx graph (this is used only to map node IDs -> lat/lon)...")
    G_multi = ox.graph_from_point(HYDERABAD, dist=DISTANCE_GRAPH_METERS, network_type="drive")
    if "crs" not in G_multi.graph:
        # project_graph sets crs and geometry fields; OSMnx functions need 'crs' present
        G_multi = ox.project_graph(G_multi)
    G = convert_to_digraph_preserve_graph_attrs(G_multi)
    print("  Graph loaded. Nodes:", len(G.nodes))

    all_records = []
    for idx, row in df.iterrows():
        src = row.get("source", "Hyderabad")
        dest = row["destination"]
        rid = row["route_id"]
        node_seq = row["node_sequence"]

        # get sampled waypoints for the route
        coords = sample_waypoints_from_node_sequence(G, node_seq, max_points=MAX_WAYPOINTS)
        if len(coords) < 2:
            print(f"⚠️ route {rid} {src}->{dest} has insufficient coords; skipping.")
            continue

        positions_str = tomtom_route_positions_string(coords)
        # fetch TomTom for that exact route geometry
        traffic = fetch_traffic_for_route_positions(positions_str)

        # choose weather location (destination)
        dest_lat, dest_lon = coords[-1]  # end node lat,lon
        weather = fetch_weather(dest_lat, dest_lon)
        if not weather:
            # still keep traffic but no weather rows
            print(f"⚠️ No weather for {dest}, skipping weather join.")
            continue

        # Combine: for every weather day, attach the traffic metrics for this route
        for w in weather:
            rec = {
                "route_id": rid,
                "source": src,
                "destination": dest,
                "positions_sampled": len(coords),
                "traffic_distance_m": traffic["traffic_distance_m"],
                "traffic_duration_min": traffic["traffic_duration_min"],
                "no_traffic_duration_min": traffic["no_traffic_duration_min"],
                **w
            }
            all_records.append(rec)

        # polite throttle
        time.sleep(0.3)

    df_out = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(df_out)} rows → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
