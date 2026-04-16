# Florio Industries — Scout Dashboard System
## Edge-Native Agricultural Scouting Platform

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    JETSON ORIN NANO                       │
│                                                          │
│  ┌──────────────┐   YOUR EXISTING PIPELINE (unchanged)  │
│  │  SIYI A8     │──→ RTSP ──→ image_publisher           │
│  │  Camera      │              │                         │
│  └──────────────┘              ▼                         │
│                          YOLOv8-nano                     │
│                              │                           │
│                              ▼                           │
│                      Temporal Engine                     │
│                              │                           │
│                              ▼                           │
│                      Cosmos Reason 2                     │
│                              │                           │
│                              ▼                           │
│                      Alert Matcher                       │
│                              │                           │
│                              ▼                           │
│                     Evidence Recorder                    │
│                              │                           │
│  ┌───────────────────────────┼────── NEW ADDITIONS ──┐  │
│  │                           ▼                        │  │
│  │              ┌─────────────────────┐               │  │
│  │              │   Mission Logger    │  (ROS 2 node) │  │
│  │              │   PASSIVE listener  │  subscribes   │  │
│  │              │   zero interference │  only         │  │
│  │              └─────────┬───────────┘               │  │
│  │                        │ writes                    │  │
│  │                        ▼                           │  │
│  │              ┌─────────────────────┐               │  │
│  │              │   SQLite Database   │               │  │
│  │              │   ~/firmament_data/ │               │  │
│  │              │   missions.db       │               │  │
│  │              └─────────┬───────────┘               │  │
│  │                        │ reads                     │  │
│  │                        ▼                           │  │
│  │              ┌─────────────────────┐               │  │
│  │              │  Dashboard Server   │  (FastAPI)    │  │
│  │              │  port 8090          │  separate     │  │
│  │              │  serves HTML + API  │  from bridge  │  │
│  │              └─────────┬───────────┘               │  │
│  └────────────────────────┼───────────────────────────┘  │
│                           │                              │
└───────────────────────────┼──────────────────────────────┘
                            │  WiFi hotspot
                            ▼
                 ┌─────────────────────┐
                 │   Farmer's Phone    │
                 │   or Tablet         │
                 │                     │
                 │  Opens browser to   │
                 │  192.168.x.x:8090   │
                 │                     │
                 │  • Sees field map   │
                 │  • Taps detections  │
                 │  • Reads AI assess  │
                 │  • Confirms/rejects │
                 └─────────────────────┘
```

### What Each File Does

| File | Runs On | Purpose |
|------|---------|---------|
| `mission_logger.py` | Jetson (ROS 2 node) | Subscribes to your existing detection and alert topics. Writes every detection to SQLite with metadata and saves the frame as JPEG. Never publishes, never modifies existing topics. |
| `dashboard_server.py` | Jetson (FastAPI) | Reads from SQLite and serves the HTML dashboard + REST API on port 8090. Completely independent from your existing dashboard bridge. |
| `static/index.html` | Farmer's browser | The farmer-facing dashboard. Leaflet map with detection pins, severity filters, AI reasoning display, and farmer verification buttons. |
| `finetune_agricultural.py` | Your big PC | Takes farmer-verified annotations exported from the Jetson, splits into train/val, and fine-tunes YOLOv8-nano. Run after USB transfer. |

### Why This Doesn't Interfere With Anything

1. **Mission Logger is a subscriber only.** In ROS 2, adding a new subscriber to a topic has zero effect on existing publishers or other subscribers. Your detection pipeline keeps running at the same speed. The logger just listens and writes to disk.

2. **Dashboard Server runs on a different port.** Your existing secured dashboard bridge (with its API key auth and rate limiting) stays on its port. The new dashboard server runs on port 8090. They are completely independent processes.

3. **SQLite uses WAL mode.** The mission logger writes to the database while the dashboard server reads from it. WAL (Write-Ahead Logging) mode allows concurrent reads and writes without locking.

4. **Images are saved separately.** The mission logger saves detection frames to `~/firmament_data/detections/` — a completely separate directory from your evidence recorder output. No file conflicts.

### Setup on Jetson

```bash
# 1. Install dependencies (you likely have most of these already)
pip3 install fastapi uvicorn aiofiles

# 2. Copy files to Jetson
scp -r scout_dashboard/ jetson@<jetson-ip>:~/

# 3. Create data directories
mkdir -p ~/firmament_data/detections
mkdir -p ~/firmament_data/training_export

# 4. IMPORTANT: Edit mission_logger.py topic names
#    Open mission_logger.py and change these to match YOUR topics:
#      DETECTION_TOPIC = "/your_actual_detection_topic"
#      ALERT_TOPIC     = "/your_actual_alert_topic"
#      IMAGE_TOPIC     = "/your_actual_image_topic"
#      GPS_TOPIC        = "/your_actual_gps_topic"
#
#    Run `ros2 topic list` while your pipeline is active to see your topics.
#    Run `ros2 topic echo /topic_name` to see the message format.

# 5. Run the mission logger alongside your existing pipeline
#    Terminal 1 (your existing launch — unchanged):
ros2 launch firmament your_existing_launch.py

#    Terminal 2 (new — the mission logger):
python3 ~/scout_dashboard/mission_logger.py

#    Terminal 3 (new — the dashboard server):
python3 ~/scout_dashboard/dashboard_server.py

# 6. Connect farmer's phone to Jetson hotspot, open browser:
#    http://192.168.x.x:8090
```

### The Data Flywheel (How Farmer Pilots Make Your AI Smarter)

```
       ┌──────────────────────────────────────────────┐
       │                                              │
       ▼                                              │
   FLY MISSION                                        │
       │                                              │
       ▼                                              │
   AI DETECTS anomalies                               │
   (current model)                                    │
       │                                              │
       ▼                                              │
   FARMER REVIEWS on dashboard                        │
   ✓ Confirms real detections                         │
   ✗ Rejects false positives                          │
   📝 Corrects labels ("that's actually rust")        │
       │                                              │
       ▼                                              │
   EXPORT verified data                               │
   POST /api/export/training                          │
       │                                              │
       ▼                                              │
   TRANSFER to training PC                            │
   (USB drive)                                        │
       │                                              │
       ▼                                              │
   FINE-TUNE YOLOv8-nano                              │
   python3 finetune_agricultural.py --data export/    │
       │                                              │
       ▼                                              │
   DEPLOY updated weights to Jetson                   │
   (USB drive → ~/firmament_models/)                  │
       │                                              │
       └──────────────────────────────────────────────┘
       
   Each cycle:
   • Fewer false positives (model learns what's NOT a problem)
   • Better crop-specific detection (model learns YOUR crops)
   • More accurate severity ratings
   • Adapts to local conditions (your soil, your climate, your varieties)
```

### How Many Annotations Before First Retrain?

- **50 verified annotations:** Minimum viable. You'll see some improvement in false positive reduction.
- **200 verified annotations:** Good baseline. Expect meaningful improvement in crop-specific detection accuracy.
- **500+ verified annotations:** Strong. The model will start recognizing condition-specific patterns (e.g., distinguishing gray leaf spot from northern corn leaf blight).
- **Ongoing:** Every season adds new data. The model gets better every year.

For the NSF Phase I, your goal is to accumulate 200+ verified annotations across 50+ flights at 3+ farm sites. That's both a credible technical milestone for the grant AND the dataset that powers your first real ag-specific model.

### Offline Map Tiles (For Fully Disconnected Operation)

The dashboard uses OpenStreetMap tiles by default, which require internet on first load. For fully offline operation:

**Option A — Cache tiles beforehand:**
```bash
# Use a tile downloader to pre-cache your operating area
# Example with mobac (Mobile Atlas Creator):
# Download tiles for your farm areas at zoom levels 13-18
# Save as MBTiles format
# Serve locally with a lightweight tile server on the Jetson
```

**Option B — Use your own aerial imagery:**
If you've done a mapping flight, use the orthomosaic as a custom overlay:
```javascript
// In index.html, replace the tile layer with:
L.imageOverlay("field_ortho.jpg", [
  [40.585, -83.140],  // southwest corner lat/lon
  [40.595, -83.125],  // northeast corner lat/lon  
]).addTo(map);
```

**Option C — No basemap (simplest for demo):**
The dashboard works fine with no basemap — detection pins render over a gray background. The GPS coordinates are still accurate. This is perfectly acceptable for a demo video.

### Adapting the Message Formats

The mission_logger.py expects JSON-encoded String messages on your detection and alert topics. If your nodes publish different message types, you have two options:

**Option 1: Modify the logger's callbacks** to parse your actual message format. Look at `_on_detection()` and `_on_alert()` and adjust the JSON field names.

**Option 2: Add a thin adapter node** that subscribes to your native message type and republishes as JSON String messages. This keeps both the logger and your existing nodes untouched.

To figure out your message formats:
```bash
ros2 topic list                           # see all active topics
ros2 topic info /your_detection_topic     # see message type
ros2 topic echo /your_detection_topic     # see actual messages
```
