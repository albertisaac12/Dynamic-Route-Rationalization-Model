# 🚦 SUMO Basics on Windows

## 🔹 Running SUMO on Windows

### 1. Run a Simulation
- **With GUI (recommended at first):**
  ```powershell
  sumo-gui -c myconfig.sumocfg
  ```
  You’ll see cars moving on your network.

- **Without GUI (headless mode):**
  ```powershell
  sumo -c myconfig.sumocfg
  ```

---

### 2. Convert OSM → SUMO Network
If you downloaded a map from **OpenStreetMap** (`.osm` file):
```powershell
netconvert --osm-files mymap.osm -o mymap.net.xml
```
- `--osm-files` → input map  
- `-o` → output SUMO network file  

---

### 3. Generate Random Routes/Trips
SUMO has a Python tool called **randomTrips.py**.  
Example (inside PowerShell):
```powershell
python "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n mymap.net.xml -o routes.rou.xml -e 3600
```
- `-n mymap.net.xml` → use your network file  
- `-o routes.rou.xml` → generate routes here  
- `-e 3600` → run for 3600 seconds (1 hour)  

⚠️ You got the `Error: unknown url type '200'` earlier because you wrote `-n 200`.  
In SUMO, `-n` expects a **file path**, not a number.  

---

## 🔹 Understanding the XML Files

### 1. **Network (`.net.xml`)**
- The **map** of your simulation.  
- Contains roads, intersections, lanes, speed limits, etc.  

Example:
```xml
<edge id="E1" from="J1" to="J2" numLanes="2" speed="13.9"/>
```
➡️ A road (`edge`) from junction `J1` to `J2`, 2 lanes, max speed ~50 km/h.

---

### 2. **Routes (`.rou.xml`)**
- Defines **vehicles and their trips**.  
- Each vehicle has:
  - ID  
  - Departure time  
  - Path (edges to follow)  

Example:
```xml
<vehicle id="car1" depart="0">
  <route edges="E1 E2 E3"/>
</vehicle>
```
➡️ Car1 starts at time `0s` and drives along edges E1 → E2 → E3.

---

### 3. **Configuration (`.sumocfg`)**
- The **master file** that tells SUMO which network and routes to load.  

Example:
```xml
<configuration>
  <input>
    <net-file value="mymap.net.xml"/>
    <route-files value="routes.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
</configuration>
```
➡️ Runs `mymap.net.xml` with `routes.rou.xml` for 3600s.

---

### 4. **Additional Files**
- **`additional.xml`** → traffic lights, bus stops, detectors.  
- **`poly.xml`** → polygons (buildings, rivers, regions).  
- **`trips.trips.xml`** → raw trips (converted later into routes).  

---

## ✅ Summary
- `mymap.net.xml` → the roads (map).  
- `routes.rou.xml` → the vehicles and their trips.  
- `myconfig.sumocfg` → ties everything together.  
