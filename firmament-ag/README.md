# Firmament — Agricultural Branch
## Edge-Native Autonomous Crop Scouting Platform

### Branch Difference: Security → Agriculture

| Feature | Security Branch (main) | Agricultural Branch (ag) |
|---------|----------------------|--------------------------|
| Detection targets | People, vehicles, threats | Disease, pests, nutrient stress, weeds |
| Core logic | BOLO matching, loitering | Crop health scoring, zone analysis |
| VLM prompts | Security assessment | Agronomic assessment + action plan |
| Mission pattern | Patrol perimeter | Lawnmower grid (photogrammetry-ready) |
| User dispatches | Alert response | "Scan NW Quarter of North Field" |
| Environmental | Not used | BME280: humidity, temp, pressure → fungal risk |
| Output | Evidence + alerts | Scouting reports + health scores + maps |
| Persistence tracking | Loitering pattern | Disease progression across flights |
| Data flywheel | Not present | Farmer verify → retrain → improve |
| 3D/Mapping | Not present | Photogrammetry → orthomosaic + digital twin |
| Dashboard focus | Security monitoring | Crop health + zone dispatch + environmental |

### System Architecture

```
ON THE DRONE (Jetson Orin Nano)
├── EXISTING PIPELINE (unchanged)
│   ├── SIYI A8 → RTSP → image_publisher
│   ├── YOLOv8-nano (retrained on ag classes)
│   └── Temporal reasoning engine
│
├── NEW: AG PERCEPTION NODE (replaces BOLO)
│   ├── Classifies detections as crop health issues
│   ├── Locates detection in field/zone via GPS + farm config
│   ├── Correlates with environmental data (humidity → fungal risk)
│   ├── Checks persistence (same issue seen across flights = worse)
│   ├── Triggers VLM with ag-specific prompts
│   ├── Computes zone health scores (0-100)
│   └── Generates scouting reports at mission end
│
├── NEW: ENVIRONMENTAL SENSOR NODE
│   ├── BME280 via I2C (temp, humidity, pressure)
│   ├── GPS-tagged readings → spatial microclimate map
│   ├── Derived: dew point, VPD, heat index
│   ├── Fungal disease risk assessment
│   └── Publishes /ag/environmental
│
├── NEW: ZONE MISSION PLANNER
│   ├── Reads farm_config.yaml for field/zone boundaries
│   ├── Generates lawnmower waypoint patterns
│   ├── Modes: scouting, photogrammetry, spot_check
│   ├── Estimates flight time vs battery
│   └── Farmer dispatches from dashboard
│
├── EXISTING: DASHBOARD SERVER (modified for ag)
│   └── Serves farmer PWA on Jetson hotspot
│
└── EXISTING: MISSION LOGGER (modified schema for ag)
    └── Writes to ag_missions.db with crop-specific fields

ON YOUR WORKSTATION PC
├── PHOTOGRAMMETRY PIPELINE
│   ├── OpenDroneMap processing
│   ├── Orthomosaic → offline map tiles for Jetson
│   ├── DSM → topographic contours
│   ├── 3D mesh → digital twin viewer
│   └── Point cloud → volumetric analysis
│
├── TRAINING DATA MERGER
│   ├── Merges synthetic + real farmer-annotated data
│   ├── Handles class remapping between datasets
│   ├── Oversamples real data (more valuable per sample)
│   └── Outputs YOLOv8-ready dataset.yaml
│
└── MODEL FINE-TUNING
    ├── YOLOv8-nano transfer learning
    └── Deploy updated weights back to Jetson
```

### File Manifest

```
firmament-ag/
├── config/
│   └── farm_config.yaml          # Field boundaries, zones, crop types, thresholds
│
├── nodes/
│   ├── ag_perception.py          # Crop health analysis (replaces BOLO)
│   ├── environmental_sensor.py   # BME280 microclimate mapping
│   └── zone_mission_planner.py   # Dispatchable zone scanning
│
├── services/
│   └── (ag_dashboard_server.py)  # Use previous dashboard_server.py with ag DB
│
├── static/
│   └── (index.html)              # Enhanced farmer PWA dashboard
│
├── scripts/
│   ├── photogrammetry_pipeline.sh  # OpenDroneMap processing
│   └── merge_training_data.py      # Synthetic + real data merger
│
├── utils/
│   └── (shared utilities)
│
└── README.md                     # This file
```

### Quick Start

```bash
# 1. Copy to Jetson
scp -r firmament-ag/ jetson@<ip>:~/

# 2. Install dependencies
pip3 install fastapi uvicorn pyyaml smbus2 bme280

# 3. Wire BME280 sensor
#    VCC → Pin 1 (3.3V)
#    GND → Pin 6
#    SDA → Pin 3
#    SCL → Pin 5
#    Verify: i2cdetect -y -r 1  (should show 0x76)

# 4. Edit farm config for your fields
nano ~/firmament-ag/config/farm_config.yaml

# 5. Edit topic names in nodes to match YOUR pipeline
#    Check: ros2 topic list

# 6. Launch (3 terminals + your existing pipeline)
#    Terminal 1: your existing pipeline launch
#    Terminal 2: python3 ~/firmament-ag/nodes/environmental_sensor.py
#    Terminal 3: python3 ~/firmament-ag/nodes/ag_perception.py
#    Terminal 4: python3 ~/firmament-ag/nodes/zone_mission_planner.py
#    Terminal 5: python3 ~/scout_dashboard/dashboard_server.py
```

### Environmental Sensing — Why Farmers Will Want This

The BME280 gives you three readings that combine into actionable intelligence:

**Humidity at canopy level** — Weather stations measure humidity at 6 feet. 
Your drone measures it AT the crop canopy. Humidity varies significantly across 
a field — low spots hold moisture longer, areas near tree lines have different 
airflow. This spatial humidity map directly predicts where fungal diseases will 
appear BEFORE symptoms are visible.

**Vapor Pressure Deficit (VPD)** — This derived metric is what advanced growers 
actually use for irrigation decisions. VPD below 0.4 kPa means the air is too 
saturated (fungal risk). VPD above 1.6 kPa means plants are closing stomata 
(water stress). The sweet spot for most crops is 0.8-1.2 kPa. Your drone 
creates a SPATIAL VPD map of the field — no other tool does this at this cost.

**Pressure trends** — A 5+ hPa drop in one hour means a storm is incoming. 
The drone can alert the farmer and auto-RTL before weather arrives.

### Photogrammetry / 3D Mapping — The Digital Twin Pipeline

Your RGB camera can produce legitimate topographic data through photogrammetry.
No LiDAR required. The process:

1. **Fly a photogrammetry mission** (dense grid, 80% overlap, 75% sidelap)
2. **Transfer geotagged images** to workstation via USB
3. **Run photogrammetry pipeline** (`./photogrammetry_pipeline.sh images/`)
4. **Outputs:**
   - Orthomosaic: geo-referenced 2D map of the field (2 cm/pixel resolution)
   - DSM: Digital Surface Model showing elevation (crop height visible)
   - DTM: Digital Terrain Model showing bare ground (drainage analysis)
   - 3D mesh: textured 3D model the farmer can orbit in a browser
   - Point cloud: LAS format for volumetric analysis

5. **Generate offline tiles** from orthomosaic for the Jetson dashboard:
   ```
   gdal2tiles.py -z 13-20 -w none orthophoto.tif tiles/
   ```
   Copy to Jetson → dashboard shows YOUR aerial imagery as the basemap.

6. **Extract contour lines** from DSM for topographic analysis:
   ```
   gdal_contour -a elevation -i 0.5 dsm.tif contours.gpkg
   ```

### Training Data Strategy

**Phase 1: Bootstrap (before first real flights)**
- Use your synthetic data factory to generate 500+ training images
- Cover all 11 agricultural detection classes
- Vary lighting, altitude angle, crop growth stages
- Train initial YOLOv8-nano model on synthetic only

**Phase 2: Field Validation (first 10 flights)**
- Fly with synthetic-trained model
- Dashboard shows detections — farmer confirms/rejects/relabels
- Export verified annotations from dashboard
- Expect ~50-100 verified samples from 10 flights

**Phase 3: Merged Training (after 50+ flights)**
- Merge real annotations with synthetic data
- Real data oversampled 3x (more valuable per sample)
- Fine-tune model on merged dataset
- Deploy to Jetson → fly again → collect more annotations

```bash
# Merge and train
python3 scripts/merge_training_data.py \
  --real ~/firmament_data/training_export \
  --synthetic ~/synthetic_data/ag_v1 \
  --output ~/training/merged_v1 \
  --real-weight 3.0

# Fine-tune
yolo detect train \
  data=~/training/merged_v1/dataset.yaml \
  model=yolov8n.pt \
  epochs=80 \
  imgsz=640 \
  batch=16

# Deploy
scp runs/detect/train/weights/best.pt jetson@<ip>:~/firmament_models/yolov8n_ag_v2.pt
```

### How This All Connects — The Complete Farmer Experience

1. **Farmer opens phone, connects to drone WiFi**
2. **Dashboard shows their fields and zones on a map**
3. **Farmer taps "NW Quarter" and hits "Scout"**
4. **Drone takes off, flies autonomous grid pattern over that zone**
5. **During flight:**
   - YOLO detects crop anomalies in real time
   - Environmental sensor maps humidity/temp/VPD across the zone
   - Ag perception correlates detections with environmental data
   - High-severity detections trigger VLM for detailed assessment
   - All data streams to the dashboard in real time
6. **Drone lands, farmer reviews results:**
   - Zone health score (e.g., "NW Quarter: 72/100")
   - Map pins showing each detection with AI assessment
   - Environmental overlay showing humidity/VPD hotspots
   - Scouting report: "URGENT: Fungal disease detected in rows 12-18,
     humidity 89% RH — scout on foot immediately"
7. **Farmer confirms/rejects detections (training the AI)**
8. **After several flights:** images processed into 3D digital twin
9. **Periodically:** annotations exported, model retrained, deployed

### What This Competes With

| Solution | Cost | Requires Internet | Edge AI | Microclimate | 3D Mapping |
|----------|------|-------------------|---------|--------------|------------|
| **Florio (this)** | Grant-funded | No | Yes | Yes (BME280) | Yes (photogrammetry) |
| DJI M3M + Terra | $5,000+ | Yes (cloud processing) | No | No | Basic |
| Sentera FieldAgent | $12,000+ | Yes | No | No | No |
| Satellite (Planet) | $1,000+/yr | Yes | No | No | No |
| Manual scouting | Labor cost | No | No | No | No |

Your edge-native architecture is the differentiator. Nobody else does 
on-device AI perception + environmental sensing + autonomous operation 
without internet. That's not a feature — that's a different product category.
