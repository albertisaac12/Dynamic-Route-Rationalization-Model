"""
Day 1: Route Collection Script (fixed)
Generates up to k routes (with intermediaries) from Hyderabad hub to 10 destinations
and writes raw route-node sequences + distance to CSV.

Fix: when converting MultiDiGraph -> DiGraph we preserve G.graph attributes
(e.g., 'crs') so OSMnx functions like nearest_nodes() work.
"""

import os
import osmnx as ox
import networkx as nx
import pandas as pd

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
OUTPUT_FILE = os.path.join(os.pardir, "Data", "raw_routes.csv")
HYDERABAD = (17.3850, 78.4867)  # (lat, lon)
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
K = 3  # alternate routes per pair

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def convert_to_digraph_preserve_graph_attrs(G_multi):
    """
    Convert a MultiDiGraph to a DiGraph, keep the shortest edge between u->v,
    preserve node/edge attributes and copy top-level graph attributes (including 'crs').
    """
    G = nx.DiGraph()
    # copy nodes with attributes
    for n, attrs in G_multi.nodes(data=True):
        G.add_node(n, **attrs)

    # add edges keeping the shortest 'length' edge where multiple exist
    for u, v, data in G_multi.edges(data=True):
        length = data.get("length", float("inf"))
        if G.has_edge(u, v):
            # if existing edge is longer, replace attributes with this shorter one
            if length < G[u][v].get("length", float("inf")):
                # overwrite attributes with current (shorter) edge's attributes
                G[u][v].clear()
                G[u][v].update(data)
        else:
            G.add_edge(u, v, **data)

    # Preserve graph-level attributes (critical: 'crs' must exist)
    if hasattr(G_multi, "graph") and isinstance(G_multi.graph, dict):
        # shallow copy to avoid accidental mutation
        G.graph.update(G_multi.graph.copy())

    return G

def get_routes(G, source_coords, dest_coords, k=3):
    """
    Return up to k shortest simple paths (node sequences + distance in meters).
    nearest_nodes expects x=lon, y=lat in the same CRS as the graph.
    """
    # nearest_nodes signature: nearest_nodes(G, X, Y) where X is longitude, Y latitude
    src_node = ox.distance.nearest_nodes(G, source_coords[1], source_coords[0])
    dst_node = ox.distance.nearest_nodes(G, dest_coords[1], dest_coords[0])

    try:
        path_gen = nx.shortest_simple_paths(G, src_node, dst_node, weight="length")
        routes = []
        for i, path in enumerate(path_gen):
            if i >= k:
                break
            total_dist = 0.0
            for u, v in zip(path[:-1], path[1:]):
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    # edge_data is a dict of attributes (since DiGraph)
                    total_dist += edge_data.get("length", 0)
            routes.append({
                "route_id": i + 1,
                "node_sequence": list(path),
                "distance_m": round(total_dist, 2)
            })
        return routes
    except nx.NetworkXNoPath:
        print(f"⚠️ No path found: {source_coords} → {dest_coords}")
        return []

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    print("📡 Downloading Hyderabad road network (may take a few minutes)...")
    # get graph (MultiDiGraph)
    G_multi = ox.graph_from_point(HYDERABAD, dist=15000, network_type="drive")

    # Ensure CRS exists: if not present, project the graph (OSMnx will set appropriate CRS)
    if "crs" not in G_multi.graph:
        G_multi = ox.projection.project_graph(G_multi)

    print("  original graph type:", type(G_multi))
    # convert to DiGraph while preserving G.graph attributes (crs etc.)
    G = convert_to_digraph_preserve_graph_attrs(G_multi)
    print("  converted graph type:", type(G))

    records = []
    print("🛣️ Collecting route data...")
    for dest_name, dest_coords in DESTINATIONS.items():
        routes = get_routes(G, HYDERABAD, dest_coords, k=K)
        for r in routes:
            records.append({
                "source": "Hyderabad",
                "destination": dest_name,
                "route_id": r["route_id"],
                "node_sequence": r["node_sequence"],
                "distance_m": r["distance_m"]
            })

    # write CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved {len(records)} routes to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
