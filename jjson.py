import pandas as pd
import json
import os

# ---------------------------
# Paths (auto detects correct dirs)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Data", "realistic_hd_routes_B_route_id_map.csv")
OUT_DIR = os.path.join(BASE_DIR, "ModelArtifacts")
OUT_PATH = os.path.join(OUT_DIR, "route_id_map.json")

print("Loading:", CSV_PATH)

df = pd.read_csv(CSV_PATH)

route_map = {}

# Build route map dictionary
for _, row in df.iterrows():
    rid = str(int(row["route_id"]))
    route_map[rid] = {
        "source": row["source"],
        "destination": row["destination"]
    }

# Save JSON
with open(OUT_PATH, "w") as f:
    json.dump(route_map, f, indent=4)

print("\nSaved route_id_map.json to:")
print(OUT_PATH)
