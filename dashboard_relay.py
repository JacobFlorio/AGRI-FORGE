"""
Florio Industries — Persistent Dashboard Relay
================================================
Runs on your workstation PC or a Raspberry Pi on the farm network.
Connects to the Jetson dashboard API whenever the drone is powered on,
pulls all mission data, and saves it to a local database.
Serves its own dashboard that is ALWAYS available — even when
the drone is powered off and sitting in the barn.

The farmer's experience:
  - Drone flies, data collects on Jetson as usual
  - This relay automatically detects the Jetson on the network
  - Pulls all new missions, detections, and images
  - Saves everything locally
  - Farmer opens their phone/tablet anytime — the relay dashboard
    is always running, showing all historical data from every flight

This is NOT cloud. This runs on a device on the same local network
as the farm. Data never leaves the property.

Hardware options:
  - Your workstation PC (already have it)
  - Raspberry Pi 4 ($35-50, runs 24/7 on 5W, perfect for this)
  - Any old laptop or mini PC on the farm WiFi

Usage:
  python3 dashboard_relay.py

  # Or with custom settings:
  python3 dashboard_relay.py \
    --jetson-ip 192.168.1.100 \
    --jetson-port 8090 \
    --relay-port 8080 \
    --sync-interval 15

Dependencies:
  pip install fastapi uvicorn requests aiofiles
"""

import argparse
import sqlite3
import requests
import time
import os
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# ─── CONFIGURATION ──────────────────────────────────────────────

DEFAULT_JETSON_IP = "192.168.4.1"  # Jetson hotspot default IP
DEFAULT_JETSON_PORT = 8090
DEFAULT_RELAY_PORT = 8080
DEFAULT_SYNC_INTERVAL = 15  # seconds between sync attempts
DB_PATH = os.path.expanduser("~/firmament_relay/relay.db")
IMAGE_DIR = os.path.expanduser("~/firmament_relay/images")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ─── DATABASE ────────────────────────────────────────────────────

def init_relay_db(db_path: str) -> sqlite3.Connection:
    """Initialize the relay's local database — mirrors the Jetson schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS missions (
            id TEXT PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            field_id TEXT,
            zone_id TEXT,
            crop_type TEXT,
            status TEXT,
            total_detections INTEGER DEFAULT 0,
            health_score REAL DEFAULT 100.0,
            synced_at TEXT
        );

        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            mission_id TEXT,
            timestamp TEXT,
            detection_class TEXT,
            display_name TEXT,
            confidence REAL,
            severity TEXT,
            health_impact REAL,
            bbox_x1 REAL, bbox_y1 REAL,
            bbox_x2 REAL, bbox_y2 REAL,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            field_id TEXT,
            zone_id TEXT,
            image_filename TEXT,
            vlm_reasoning TEXT,
            vlm_severity_score INTEGER,
            vlm_recommended_action TEXT,
            env_temperature REAL,
            env_humidity REAL,
            env_pressure REAL,
            env_fungal_risk TEXT,
            farmer_verified INTEGER DEFAULT 0,
            farmer_label TEXT,
            farmer_notes TEXT,
            is_persistent INTEGER DEFAULT 0,
            times_detected INTEGER DEFAULT 1,
            synced_at TEXT,
            FOREIGN KEY (mission_id) REFERENCES missions(id)
        );

        CREATE TABLE IF NOT EXISTS environmental_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mission_id TEXT,
            latitude REAL,
            longitude REAL,
            temperature_c REAL,
            humidity_pct REAL,
            pressure_hpa REAL,
            vpd_kpa REAL,
            fungal_risk TEXT,
            synced_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            missions_synced INTEGER DEFAULT 0,
            detections_synced INTEGER DEFAULT 0,
            images_synced INTEGER DEFAULT 0,
            error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_det_mission ON detections(mission_id);
        CREATE INDEX IF NOT EXISTS idx_det_zone ON detections(zone_id);
        CREATE INDEX IF NOT EXISTS idx_env_mission ON environmental_readings(mission_id);
    """)
    conn.commit()
    return conn


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ─── JETSON SYNC ENGINE ─────────────────────────────────────────

class JetsonSync:
    """
    Background thread that periodically checks if the Jetson is
    reachable and pulls any new data.

    When the Jetson is off: sync sleeps and retries.
    When the Jetson comes online: sync pulls all new missions,
    detections, environmental readings, and detection images.
    """

    def __init__(self, jetson_ip: str, jetson_port: int, interval: int):
        self.base_url = f"http://{jetson_ip}:{jetson_port}"
        self.interval = interval
        self.running = False
        self.connected = False
        self.last_sync = None
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _sync_loop(self):
        while self.running:
            try:
                self._attempt_sync()
            except Exception as e:
                self.connected = False
                self._log_sync("error", error=str(e))
            time.sleep(self.interval)

    def _check_connection(self) -> bool:
        """Ping the Jetson dashboard to see if it's reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/status", timeout=3)
            if resp.status_code == 200:
                self.connected = True
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        self.connected = False
        return False

    def _attempt_sync(self):
        if not self._check_connection():
            return

        missions_count = 0
        detections_count = 0
        images_count = 0

        with get_db() as db:
            # Sync missions
            try:
                resp = requests.get(f"{self.base_url}/api/missions?limit=100", timeout=10)
                if resp.status_code == 200:
                    missions = resp.json()
                    for m in missions:
                        existing = db.execute(
                            "SELECT id FROM missions WHERE id = ?", (m["id"],)
                        ).fetchone()
                        if not existing:
                            db.execute(
                                """INSERT OR REPLACE INTO missions
                                   (id, start_time, end_time, field_id, zone_id,
                                    crop_type, status, total_detections, health_score, synced_at)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    m.get("id"), m.get("start_time"), m.get("end_time"),
                                    m.get("field_id"), m.get("zone_id"),
                                    m.get("crop_type"), m.get("status"),
                                    m.get("total_detections", 0),
                                    m.get("health_score", 100),
                                    datetime.now(timezone.utc).isoformat(),
                                )
                            )
                            missions_count += 1
                    db.commit()
            except Exception as e:
                print(f"  Mission sync error: {e}")

            # Sync detections (GeoJSON format from the Jetson API)
            try:
                resp = requests.get(
                    f"{self.base_url}/api/detections?limit=1000",
                    timeout=30
                )
                if resp.status_code == 200:
                    geojson = resp.json()
                    features = geojson.get("features", [])

                    for f in features:
                        p = f.get("properties", {})
                        coords = f.get("geometry", {}).get("coordinates", [0, 0])
                        det_id = p.get("id")

                        existing = db.execute(
                            "SELECT id FROM detections WHERE id = ?", (det_id,)
                        ).fetchone()
                        if existing:
                            # Update farmer verification if changed
                            if p.get("farmer_verified"):
                                db.execute(
                                    """UPDATE detections SET
                                       farmer_verified = ?, farmer_label = ?, farmer_notes = ?
                                       WHERE id = ?""",
                                    (p.get("farmer_verified"), p.get("farmer_label"),
                                     p.get("farmer_notes"), det_id)
                                )
                            continue

                        # Download detection image if available
                        image_filename = None
                        image_url = p.get("image_url")
                        if image_url:
                            try:
                                img_resp = requests.get(
                                    f"{self.base_url}{image_url}", timeout=10
                                )
                                if img_resp.status_code == 200:
                                    image_filename = f"{det_id}.jpg"
                                    img_path = os.path.join(IMAGE_DIR, image_filename)
                                    with open(img_path, "wb") as img_file:
                                        img_file.write(img_resp.content)
                                    images_count += 1
                            except Exception:
                                pass

                        db.execute(
                            """INSERT INTO detections
                               (id, mission_id, timestamp, detection_class, display_name,
                                confidence, severity, health_impact,
                                latitude, longitude,
                                field_id, zone_id, image_filename,
                                vlm_reasoning,
                                env_temperature, env_humidity, env_pressure, env_fungal_risk,
                                farmer_verified, farmer_label, farmer_notes,
                                is_persistent, times_detected, synced_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                                       ?, ?,
                                       ?, ?, ?,
                                       ?,
                                       ?, ?, ?, ?,
                                       ?, ?, ?,
                                       ?, ?, ?)""",
                            (
                                det_id, p.get("mission_id"), p.get("timestamp"),
                                p.get("class"), p.get("display_name", p.get("class")),
                                p.get("confidence", 0), p.get("severity", "low"),
                                p.get("health_impact", 0),
                                coords[1] if len(coords) > 1 else 0,  # lat
                                coords[0] if len(coords) > 0 else 0,  # lon
                                p.get("field_id"), p.get("zone_id"),
                                image_filename,
                                p.get("vlm_reasoning"),
                                p.get("env_temperature"), p.get("env_humidity"),
                                p.get("env_pressure"), p.get("env_fungal_risk"),
                                p.get("farmer_verified", 0),
                                p.get("farmer_label"), p.get("farmer_notes"),
                                p.get("is_persistent", 0),
                                p.get("times_detected", 1),
                                datetime.now(timezone.utc).isoformat(),
                            )
                        )
                        detections_count += 1
                    db.commit()
            except Exception as e:
                print(f"  Detection sync error: {e}")

        self.last_sync = datetime.now(timezone.utc).isoformat()
        if missions_count > 0 or detections_count > 0:
            print(
                f"  Synced: {missions_count} missions, "
                f"{detections_count} detections, {images_count} images"
            )
        self._log_sync("ok", missions_count, detections_count, images_count)

    def _log_sync(self, status, missions=0, detections=0, images=0, error=None):
        try:
            with get_db() as db:
                db.execute(
                    """INSERT INTO sync_log
                       (timestamp, status, missions_synced, detections_synced,
                        images_synced, error)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (datetime.now(timezone.utc).isoformat(),
                     status, missions, detections, images, error)
                )
                db.commit()
        except Exception:
            pass


# ─── RELAY DASHBOARD API ────────────────────────────────────────

app = FastAPI(
    title="Florio Industries — Farm Dashboard",
    description="Always-on crop scouting dashboard. Syncs from drone automatically.",
    version="0.1.0",
)

# Serve synced detection images
if os.path.isdir(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Global reference to sync engine (set in main)
sync_engine: JetsonSync = None


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the farmer dashboard — works identically to the Jetson version."""
    # Use the same index.html from the ag branch
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    # Fallback: check for a relay-specific dashboard
    relay_index = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "relay_dashboard.html"
    )
    if os.path.exists(relay_index):
        return FileResponse(relay_index)
    return HTMLResponse(
        "<h1>Dashboard files not found.</h1>"
        "<p>Copy index.html from products/agriculture/static/ to the static/ directory.</p>"
    )


@app.get("/api/status")
async def relay_status():
    """Status including Jetson connection state and sync info."""
    with get_db() as db:
        total_missions = db.execute("SELECT COUNT(*) FROM missions").fetchone()[0]
        total_detections = db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        last_sync_log = db.execute(
            "SELECT * FROM sync_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

    return {
        "status": "online",
        "type": "relay",
        "jetson_connected": sync_engine.connected if sync_engine else False,
        "last_sync": sync_engine.last_sync if sync_engine else None,
        "total_missions": total_missions,
        "total_detections": total_detections,
        "last_sync_status": dict(last_sync_log) if last_sync_log else None,
        "edge_native": True,
        "cloud_dependency": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/missions")
async def list_missions(limit: int = Query(50, ge=1, le=200)):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM missions ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: str):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM missions WHERE id = ?", (mission_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Mission not found")
    return dict(row)


@app.get("/api/detections")
async def list_detections(
    mission_id: str = None,
    severity: str = None,
    field_id: str = None,
    zone_id: str = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(500, ge=1, le=5000),
):
    """Returns GeoJSON — same format as the Jetson dashboard API."""
    query = "SELECT * FROM detections WHERE confidence >= ?"
    params = [min_confidence]

    if mission_id:
        query += " AND mission_id = ?"
        params.append(mission_id)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if field_id:
        query += " AND field_id = ?"
        params.append(field_id)
    if zone_id:
        query += " AND zone_id = ?"
        params.append(zone_id)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_db() as db:
        rows = db.execute(query, params).fetchall()

    features = []
    for r in rows:
        r = dict(r)
        image_url = None
        if r.get("image_filename"):
            image_url = f"/images/{r['image_filename']}"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    r.get("longitude", 0),
                    r.get("latitude", 0)
                ],
            },
            "properties": {
                "id": r["id"],
                "mission_id": r.get("mission_id"),
                "timestamp": r.get("timestamp"),
                "class": r.get("detection_class"),
                "display_name": r.get("display_name"),
                "confidence": r.get("confidence"),
                "severity": r.get("severity"),
                "health_impact": r.get("health_impact"),
                "vlm_reasoning": r.get("vlm_reasoning"),
                "is_bolo_match": 0,
                "is_persistent": r.get("is_persistent", 0),
                "image_url": image_url,
                "farmer_verified": r.get("farmer_verified", 0),
                "farmer_label": r.get("farmer_label"),
                "farmer_notes": r.get("farmer_notes"),
                "env_temperature": r.get("env_temperature"),
                "env_humidity": r.get("env_humidity"),
                "env_fungal_risk": r.get("env_fungal_risk"),
            },
        })

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/detections/{detection_id}")
async def get_detection(detection_id: str):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")
    result = dict(row)
    if result.get("image_filename"):
        result["image_url"] = f"/images/{result['image_filename']}"
    return result


@app.post("/api/detections/{detection_id}/verify")
async def verify_detection(detection_id: str, body: dict):
    """
    Farmer verification — works even when Jetson is offline.
    Annotations are stored locally. When the Jetson comes back online,
    a future sync can push annotations back (not yet implemented,
    but the data is preserved).
    """
    with get_db() as db:
        existing = db.execute(
            "SELECT id FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Detection not found")

        db.execute(
            """UPDATE detections SET
               farmer_verified = ?,
               farmer_label = ?,
               farmer_notes = ?
               WHERE id = ?""",
            (
                1 if body.get("verified", True) else -1,
                body.get("corrected_label"),
                body.get("notes"),
                detection_id,
            )
        )
        db.commit()

    return {"status": "ok", "detection_id": detection_id}


@app.get("/api/history/zone/{zone_id}")
async def zone_history(zone_id: str, limit: int = Query(20, ge=1, le=100)):
    """
    Historical health data for a zone across all flights.
    This is the time-series view — how has this zone's health
    changed over the season?
    """
    with get_db() as db:
        missions = db.execute(
            """SELECT m.id, m.start_time, m.health_score,
                      COUNT(d.id) as detection_count,
                      SUM(CASE WHEN d.severity = 'high' THEN 1 ELSE 0 END) as high_count,
                      AVG(d.env_humidity) as avg_humidity,
                      AVG(d.env_temperature) as avg_temperature
               FROM missions m
               LEFT JOIN detections d ON d.mission_id = m.id AND d.zone_id = ?
               WHERE m.zone_id = ? OR d.zone_id = ?
               GROUP BY m.id
               ORDER BY m.start_time DESC
               LIMIT ?""",
            (zone_id, zone_id, zone_id, limit)
        ).fetchall()

    return [dict(r) for r in missions]


@app.get("/api/sync/log")
async def sync_log(limit: int = Query(20, ge=1, le=100)):
    """View sync history — when did the relay last talk to the Jetson?"""
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM sync_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─── ENTRY POINT ────────────────────────────────────────────────

def main():
    global sync_engine

    parser = argparse.ArgumentParser(
        description="Florio Industries — Persistent Farm Dashboard Relay"
    )
    parser.add_argument("--jetson-ip", default=DEFAULT_JETSON_IP,
                        help=f"Jetson IP address (default: {DEFAULT_JETSON_IP})")
    parser.add_argument("--jetson-port", type=int, default=DEFAULT_JETSON_PORT,
                        help=f"Jetson dashboard port (default: {DEFAULT_JETSON_PORT})")
    parser.add_argument("--relay-port", type=int, default=DEFAULT_RELAY_PORT,
                        help=f"Relay dashboard port (default: {DEFAULT_RELAY_PORT})")
    parser.add_argument("--sync-interval", type=int, default=DEFAULT_SYNC_INTERVAL,
                        help=f"Seconds between sync attempts (default: {DEFAULT_SYNC_INTERVAL})")
    args = parser.parse_args()

    # Initialize database
    init_relay_db(DB_PATH)

    print(f"\n{'='*60}")
    print(f"  FLORIO INDUSTRIES — Farm Dashboard Relay")
    print(f"  Always-on. No cloud. Data stays on the farm.")
    print(f"{'='*60}")
    print(f"  Relay dashboard:  http://0.0.0.0:{args.relay_port}")
    print(f"  Jetson target:    http://{args.jetson_ip}:{args.jetson_port}")
    print(f"  Sync interval:    {args.sync_interval}s")
    print(f"  Database:         {DB_PATH}")
    print(f"  Images:           {IMAGE_DIR}")
    print(f"{'='*60}")
    print(f"  Farmer connects to: http://<this-pc-ip>:{args.relay_port}")
    print(f"  Works even when the drone is powered off.")
    print(f"{'='*60}\n")

    # Start sync engine
    sync_engine = JetsonSync(args.jetson_ip, args.jetson_port, args.sync_interval)
    sync_engine.start()
    print("Sync engine started. Waiting for Jetson connection...\n")

    # Start relay dashboard
    uvicorn.run(app, host="0.0.0.0", port=args.relay_port, log_level="info")


if __name__ == "__main__":
    main()
