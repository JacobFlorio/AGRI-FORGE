"""
Florio Industries — Mission Logger ROS 2 Node
================================================
Passive subscriber that listens to existing detection/alert topics
and writes structured data to a local SQLite database.

DOES NOT interfere with any existing pipeline nodes.
It only subscribes — never publishes or modifies existing topics.

Integration:
  - Subscribes to your existing detection topic (adjust DETECTION_TOPIC)
  - Subscribes to your existing alert topic (adjust ALERT_TOPIC)
  - Saves detection frames to disk as JPEGs
  - Writes all metadata to SQLite for the local dashboard

Usage:
  ros2 run firmament mission_logger

Or standalone:
  python3 mission_logger.py
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import sqlite3
import json
import os
import uuid
from datetime import datetime, timezone
from cv_bridge import CvBridge
import cv2
from pathlib import Path

# ─── CONFIGURATION ─────────────────────────────────────────────
# Adjust these topic names to match YOUR existing pipeline's published topics.
# Run `ros2 topic list` to see what your pipeline currently publishes.

DETECTION_TOPIC = "/detections"        # Topic where YOLOv8 publishes detection results
ALERT_TOPIC = "/bolo_alerts"           # Topic where BOLO match / alert system publishes
IMAGE_TOPIC = "/camera/image_raw"      # Topic where image_publisher publishes camera frames
GPS_TOPIC = "/gps/fix"                 # Topic for GPS data (adjust to your Pixhawk topic)

# Storage paths
DB_PATH = os.path.expanduser("~/firmament_data/missions.db")
IMAGE_DIR = os.path.expanduser("~/firmament_data/detections")
TRAINING_EXPORT_DIR = os.path.expanduser("~/firmament_data/training_export")

# ─── DATABASE SETUP ────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database with schema for missions, detections, and annotations."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS missions (
            id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT DEFAULT 'active',
            total_detections INTEGER DEFAULT 0,
            total_alerts INTEGER DEFAULT 0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            mission_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            detection_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            image_path TEXT,
            severity TEXT DEFAULT 'low',
            vlm_reasoning TEXT,
            is_bolo_match INTEGER DEFAULT 0,
            farmer_verified INTEGER DEFAULT 0,
            farmer_label TEXT,
            farmer_notes TEXT,
            FOREIGN KEY (mission_id) REFERENCES missions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_detections_mission
            ON detections(mission_id);
        CREATE INDEX IF NOT EXISTS idx_detections_severity
            ON detections(severity);
        CREATE INDEX IF NOT EXISTS idx_detections_verified
            ON detections(farmer_verified);
    """)
    conn.commit()
    return conn


# ─── ROS 2 NODE ────────────────────────────────────────────────

class MissionLogger(Node):
    """
    Passively subscribes to existing pipeline topics and logs
    detections + alerts to SQLite. Zero interference with
    existing nodes — this is a read-only observer.
    """

    def __init__(self):
        super().__init__("mission_logger")
        self.get_logger().info("Mission Logger starting — passive subscriber mode")

        # Storage setup
        os.makedirs(IMAGE_DIR, exist_ok=True)
        os.makedirs(TRAINING_EXPORT_DIR, exist_ok=True)
        self.db = init_db(DB_PATH)
        self.bridge = CvBridge()

        # State
        self.current_mission_id = None
        self.latest_image = None
        self.latest_gps = {"lat": 0.0, "lon": 0.0, "alt": 0.0}

        # Start a new mission on node startup
        self._start_mission()

        # Subscribe to existing topics — THESE DO NOT AFFECT YOUR PIPELINE
        # They are additional subscribers on topics that already exist.
        self.create_subscription(
            Image, IMAGE_TOPIC, self._on_image, 1  # QoS depth 1, only latest frame
        )
        self.create_subscription(
            String, DETECTION_TOPIC, self._on_detection, 10
        )
        self.create_subscription(
            String, ALERT_TOPIC, self._on_alert, 10
        )

        # NOTE: For GPS, you may need to use sensor_msgs/NavSatFix instead of String.
        # Adjust the message type and callback based on your Pixhawk's GPS topic.
        self.create_subscription(
            String, GPS_TOPIC, self._on_gps, 10
        )

        self.get_logger().info(f"Logging to: {DB_PATH}")
        self.get_logger().info(f"Images to: {IMAGE_DIR}")
        self.get_logger().info(f"Mission started: {self.current_mission_id}")

    def _start_mission(self):
        """Create a new mission record."""
        self.current_mission_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT INTO missions (id, start_time) VALUES (?, ?)",
            (self.current_mission_id, now)
        )
        self.db.commit()

    def _on_image(self, msg: Image):
        """Cache the latest camera frame for saving with detections."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")

    def _on_gps(self, msg):
        """
        Cache latest GPS fix.
        IMPORTANT: Adjust this based on your actual GPS message type.
        If using sensor_msgs/NavSatFix, access msg.latitude, msg.longitude, msg.altitude.
        If your pipeline publishes GPS as JSON on a String topic, parse accordingly.
        """
        try:
            # Example for JSON-encoded GPS on a String topic:
            data = json.loads(msg.data)
            self.latest_gps = {
                "lat": data.get("latitude", 0.0),
                "lon": data.get("longitude", 0.0),
                "alt": data.get("altitude", 0.0),
            }
        except Exception:
            pass  # Silently skip malformed GPS — don't crash the logger

    def _on_detection(self, msg: String):
        """
        Handle a detection event from your YOLOv8 node.
        IMPORTANT: Adjust the JSON parsing below to match whatever format
        your detection node publishes. Run `ros2 topic echo /detections`
        to see the exact format.
        """
        try:
            data = json.loads(msg.data)
            detection_id = str(uuid.uuid4())[:12]
            now = datetime.now(timezone.utc).isoformat()

            # Save detection frame
            image_path = None
            if self.latest_image is not None:
                image_filename = f"{self.current_mission_id}_{detection_id}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                cv2.imwrite(image_path, self.latest_image)

            # Write to database
            # ADJUST these field names to match your detection message format
            self.db.execute(
                """INSERT INTO detections
                   (id, mission_id, timestamp, detection_class, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    latitude, longitude, altitude, image_path, severity)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    detection_id,
                    self.current_mission_id,
                    now,
                    data.get("class", "unknown"),
                    data.get("confidence", 0.0),
                    data.get("bbox", [0, 0, 0, 0])[0],
                    data.get("bbox", [0, 0, 0, 0])[1],
                    data.get("bbox", [0, 0, 0, 0])[2],
                    data.get("bbox", [0, 0, 0, 0])[3],
                    self.latest_gps["lat"],
                    self.latest_gps["lon"],
                    self.latest_gps["alt"],
                    image_path,
                    data.get("severity", "low"),
                )
            )

            # Update mission detection count
            self.db.execute(
                "UPDATE missions SET total_detections = total_detections + 1 WHERE id = ?",
                (self.current_mission_id,)
            )
            self.db.commit()

            self.get_logger().info(
                f"Detection logged: {data.get('class', '?')} "
                f"({data.get('confidence', 0):.2f}) @ "
                f"{self.latest_gps['lat']:.6f}, {self.latest_gps['lon']:.6f}"
            )

        except Exception as e:
            self.get_logger().error(f"Detection logging failed: {e}")

    def _on_alert(self, msg: String):
        """
        Handle high-severity alert from your alert system.
        This updates an existing detection record with VLM reasoning
        and marks it as an alert match, OR creates a new record if
        the alert includes its own detection data.
        """
        try:
            data = json.loads(msg.data)
            detection_id = str(uuid.uuid4())[:12]
            now = datetime.now(timezone.utc).isoformat()

            image_path = None
            if self.latest_image is not None:
                image_filename = f"{self.current_mission_id}_alert_{detection_id}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                cv2.imwrite(image_path, self.latest_image)

            self.db.execute(
                """INSERT INTO detections
                   (id, mission_id, timestamp, detection_class, confidence,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    latitude, longitude, altitude, image_path,
                    severity, vlm_reasoning, is_bolo_match)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    detection_id,
                    self.current_mission_id,
                    now,
                    data.get("class", "alert"),
                    data.get("confidence", 0.0),
                    data.get("bbox", [0, 0, 0, 0])[0],
                    data.get("bbox", [0, 0, 0, 0])[1],
                    data.get("bbox", [0, 0, 0, 0])[2],
                    data.get("bbox", [0, 0, 0, 0])[3],
                    self.latest_gps["lat"],
                    self.latest_gps["lon"],
                    self.latest_gps["alt"],
                    image_path,
                    data.get("severity", "high"),
                    data.get("vlm_reasoning", ""),
                )
            )

            self.db.execute(
                "UPDATE missions SET total_alerts = total_alerts + 1 WHERE id = ?",
                (self.current_mission_id,)
            )
            self.db.commit()

            self.get_logger().info(f"ALERT logged: {data.get('class', '?')} — BOLO MATCH")

        except Exception as e:
            self.get_logger().error(f"Alert logging failed: {e}")

    def end_mission(self):
        """Call this when the mission ends (landing detected or manual trigger)."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE missions SET end_time = ?, status = 'complete' WHERE id = ?",
            (now, self.current_mission_id)
        )
        self.db.commit()
        self.get_logger().info(f"Mission {self.current_mission_id} ended.")

    def destroy_node(self):
        self.end_mission()
        self.db.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down mission logger...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
