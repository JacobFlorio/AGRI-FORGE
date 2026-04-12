"""
Florio Industries — Local Dashboard Server
============================================
Lightweight FastAPI server running ON the Jetson Orin Nano.
Serves the farmer-facing dashboard and API endpoints.
Reads from the SQLite database populated by mission_logger.py.

Runs on a SEPARATE port from your existing dashboard bridge.
Zero interference with your existing secured bridge.

Usage:
    python3 dashboard_server.py
    # or with custom port:
    python3 dashboard_server.py --port 8090

Then the farmer connects to the Jetson's hotspot and opens:
    http://192.168.x.x:8090

Dependencies (install on Jetson):
    pip3 install fastapi uvicorn aiofiles
    # You likely already have sqlite3 (stdlib) and cv2
"""

import argparse
import sqlite3
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

# ─── CONFIGURATION ─────────────────────────────────────────────

DB_PATH = os.path.expanduser("~/firmament_data/missions.db")
IMAGE_DIR = os.path.expanduser("~/firmament_data/detections")
TRAINING_EXPORT_DIR = os.path.expanduser("~/firmament_data/training_export")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DEFAULT_PORT = 8090  # Different from your existing dashboard bridge port

# ─── DATABASE HELPER ───────────────────────────────────────────

@contextmanager
def get_db():
    """Thread-safe read connection to the mission database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


# ─── FASTAPI APP ───────────────────────────────────────────────

app = FastAPI(
    title="Florio Industries — Scout Dashboard",
    description="Edge-native agricultural scouting dashboard. No cloud required.",
    version="0.1.0",
)

# Serve detection images directly from local storage
if os.path.isdir(IMAGE_DIR):
    app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Serve static frontend files (HTML/JS/CSS)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Dashboard HTML ──

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Dashboard files not found. Place index.html in /static/</h1>")


# ── Mission Endpoints ──

@app.get("/api/missions")
async def list_missions(limit: int = Query(20, ge=1, le=100)):
    """List all missions, most recent first."""
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM missions ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/missions/{mission_id}")
async def get_mission(mission_id: str):
    """Get details for a specific mission."""
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM missions WHERE id = ?", (mission_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Mission not found")
    return dict(row)


@app.get("/api/missions/{mission_id}/summary")
async def get_mission_summary(mission_id: str):
    """Get summary stats for a mission — detection counts by class and severity."""
    with get_db() as db:
        by_class = db.execute(
            """SELECT detection_class, COUNT(*) as count, AVG(confidence) as avg_conf
               FROM detections WHERE mission_id = ?
               GROUP BY detection_class ORDER BY count DESC""",
            (mission_id,)
        ).fetchall()

        by_severity = db.execute(
            """SELECT severity, COUNT(*) as count
               FROM detections WHERE mission_id = ?
               GROUP BY severity""",
            (mission_id,)
        ).fetchall()

        bolo_count = db.execute(
            "SELECT COUNT(*) FROM detections WHERE mission_id = ? AND is_bolo_match = 1",
            (mission_id,)
        ).fetchone()[0]

    return {
        "mission_id": mission_id,
        "by_class": [dict(r) for r in by_class],
        "by_severity": [dict(r) for r in by_severity],
        "bolo_matches": bolo_count,
    }


# ── Detection Endpoints ──

@app.get("/api/detections")
async def list_detections(
    mission_id: str = None,
    severity: str = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    List detections with optional filters.
    This is the main data source for the map view.
    Returns GeoJSON-compatible data.
    """
    query = "SELECT * FROM detections WHERE confidence >= ?"
    params = [min_confidence]

    if mission_id:
        query += " AND mission_id = ?"
        params.append(mission_id)
    if severity:
        query += " AND severity = ?"
        params.append(severity)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with get_db() as db:
        rows = db.execute(query, params).fetchall()

    # Convert to GeoJSON FeatureCollection for direct Leaflet consumption
    features = []
    for r in rows:
        r = dict(r)
        # Convert image path to serveable URL
        if r.get("image_path"):
            r["image_url"] = f"/images/{os.path.basename(r['image_path'])}"
        else:
            r["image_url"] = None

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [r["longitude"], r["latitude"]],
            },
            "properties": {
                "id": r["id"],
                "mission_id": r["mission_id"],
                "timestamp": r["timestamp"],
                "class": r["detection_class"],
                "confidence": r["confidence"],
                "severity": r["severity"],
                "vlm_reasoning": r["vlm_reasoning"],
                "is_bolo_match": r["is_bolo_match"],
                "image_url": r["image_url"],
                "farmer_verified": r["farmer_verified"],
                "farmer_label": r["farmer_label"],
                "farmer_notes": r["farmer_notes"],
            },
        })

    return {"type": "FeatureCollection", "features": features}


@app.get("/api/detections/{detection_id}")
async def get_detection(detection_id: str):
    """Get full details for a single detection."""
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")
    result = dict(row)
    if result.get("image_path"):
        result["image_url"] = f"/images/{os.path.basename(result['image_path'])}"
    return result


# ── Farmer Annotation Endpoints (the training data flywheel) ──

@app.post("/api/detections/{detection_id}/verify")
async def verify_detection(detection_id: str, body: dict):
    """
    Farmer confirms or corrects a detection.
    This is how pilot data becomes training data.

    Body JSON:
    {
        "verified": true,           // farmer confirms this is a real detection
        "corrected_label": null,    // or "fungal_disease", "pest_damage", etc.
        "notes": "Looks like gray leaf spot"  // optional farmer notes
    }
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
                1 if body.get("verified", True) else -1,  # 1=confirmed, -1=rejected
                body.get("corrected_label"),
                body.get("notes"),
                detection_id,
            )
        )
        db.commit()

    return {"status": "ok", "detection_id": detection_id}


# ── Training Data Export ──

@app.post("/api/export/training")
async def export_training_data(body: dict = {}):
    """
    Export farmer-verified detections as a YOLO-format training dataset.
    Run this after collecting annotations, then transfer the export
    directory to your training PC via USB.

    Generates:
        ~/firmament_data/training_export/
            images/
                <detection_id>.jpg
            labels/
                <detection_id>.txt    (YOLO format: class x_center y_center w h)
            classes.txt               (class name mapping)
            dataset.yaml              (YOLOv8 training config)
    """
    images_dir = os.path.join(TRAINING_EXPORT_DIR, "images")
    labels_dir = os.path.join(TRAINING_EXPORT_DIR, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Only export verified detections (farmer_verified = 1)
    # If farmer provided a corrected label, use that instead of the original
    with get_db() as db:
        rows = db.execute(
            """SELECT * FROM detections
               WHERE farmer_verified = 1 AND image_path IS NOT NULL"""
        ).fetchall()

    if not rows:
        return {"status": "no_verified_detections", "count": 0}

    # Build class mapping from all unique labels
    class_set = set()
    for r in rows:
        r = dict(r)
        label = r.get("farmer_label") or r.get("detection_class", "unknown")
        class_set.add(label)
    class_list = sorted(class_set)
    class_map = {name: idx for idx, name in enumerate(class_list)}

    exported = 0
    for r in rows:
        r = dict(r)
        det_id = r["id"]
        src_image = r["image_path"]

        if not src_image or not os.path.exists(src_image):
            continue

        # Copy image to export directory
        dst_image = os.path.join(images_dir, f"{det_id}.jpg")
        shutil.copy2(src_image, dst_image)

        # Get image dimensions for YOLO normalization
        import cv2
        img = cv2.imread(src_image)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Convert bbox from absolute to YOLO normalized format
        # YOLO format: class_id x_center y_center width height (all 0-1)
        x1 = r.get("bbox_x1", 0) or 0
        y1 = r.get("bbox_y1", 0) or 0
        x2 = r.get("bbox_x2", 0) or 0
        y2 = r.get("bbox_y2", 0) or 0

        if img_w > 0 and img_h > 0 and (x2 - x1) > 0:
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            label = r.get("farmer_label") or r.get("detection_class", "unknown")
            class_id = class_map.get(label, 0)

            label_path = os.path.join(labels_dir, f"{det_id}.txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            exported += 1

    # Write class list
    classes_path = os.path.join(TRAINING_EXPORT_DIR, "classes.txt")
    with open(classes_path, "w") as f:
        for name in class_list:
            f.write(f"{name}\n")

    # Write YOLOv8 dataset.yaml
    yaml_path = os.path.join(TRAINING_EXPORT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# Florio Industries — Agricultural Detection Training Data\n")
        f.write(f"# Exported: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# Verified by farmer annotations from field pilots\n\n")
        f.write(f"path: {TRAINING_EXPORT_DIR}\n")
        f.write(f"train: images\n")
        f.write(f"val: images  # Split manually or use train/val script\n\n")
        f.write(f"nc: {len(class_list)}\n")
        f.write(f"names: {class_list}\n")

    return {
        "status": "exported",
        "count": exported,
        "classes": class_list,
        "export_path": TRAINING_EXPORT_DIR,
        "next_step": "Transfer this directory to your training PC and run: "
                     "yolo detect train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640",
    }


# ── System Status ──

@app.get("/api/status")
async def system_status():
    """Basic system health check — useful for the dashboard to show connection status."""
    db_exists = os.path.exists(DB_PATH)
    total_detections = 0
    total_missions = 0
    if db_exists:
        with get_db() as db:
            total_detections = db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            total_missions = db.execute("SELECT COUNT(*) FROM missions").fetchone()[0]

    return {
        "status": "online",
        "database": "connected" if db_exists else "missing",
        "total_missions": total_missions,
        "total_detections": total_detections,
        "edge_native": True,
        "cloud_dependency": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── ENTRY POINT ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Florio Industries Scout Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  FLORIO INDUSTRIES — Scout Dashboard")
    print(f"  Edge-native. No cloud. No internet required.")
    print(f"{'='*60}")
    print(f"  Dashboard:  http://{args.host}:{args.port}")
    print(f"  API docs:   http://{args.host}:{args.port}/docs")
    print(f"  Database:   {DB_PATH}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
