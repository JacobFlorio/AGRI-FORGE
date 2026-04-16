"""
Florio Industries — Agricultural Perception Node
==================================================
Agricultural perception node for crop health analysis.

Pipeline position:
  SIYI A8 → RTSP → image_publisher → YOLOv8 → temporal_engine → [THIS NODE]

What this node does:
  - Receives detections and classifies them as CROP HEALTH issues
  - Scores detections against agricultural classes (disease, pest, nutrient, etc.)
  - Triggers Cosmos Reason 2 with ag-specific prompts
  - Generates zone-aware scouting reports
  - Correlates detections with environmental sensor data (humidity, temp)
  - Tracks persistent anomalies across flights for trend analysis

Subscribes to:
  /ag/detections          - from YOLOv8 (ag-retrained model)
  /ag/temporal_events     - from temporal engine
  /ag/environmental       - from BME280 environmental sensor node
  /gps/fix                - from Pixhawk

Publishes to:
  /ag/crop_alerts         - structured crop health alerts
  /ag/scouting_report     - per-zone summary after mission
  /ag/vlm_request         - triggers Cosmos Reason 2 analysis
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import os
import uuid
import sqlite3
import yaml
import math
from datetime import datetime, timezone
from cv_bridge import CvBridge
import cv2
from pathlib import Path

# ─── CONFIGURATION ──────────────────────────────────────────────

CONFIG_PATH = os.path.expanduser("~/firmament-ag/config/farm_config.yaml")
DB_PATH = os.path.expanduser("~/firmament_data/ag_missions.db")
IMAGE_DIR = os.path.expanduser("~/firmament_data/ag_detections")

DETECTION_TOPIC = "/ag/detections"
TEMPORAL_TOPIC = "/ag/temporal_events"
ENVIRONMENTAL_TOPIC = "/ag/environmental"
IMAGE_TOPIC = "/camera/image_raw"
GPS_TOPIC = "/gps/fix"

CROP_ALERT_TOPIC = "/ag/crop_alerts"
SCOUTING_REPORT_TOPIC = "/ag/scouting_report"
VLM_REQUEST_TOPIC = "/ag/vlm_request"


def init_ag_db(db_path: str) -> sqlite3.Connection:
    """Initialize agricultural mission database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ag_missions (
            id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            field_id TEXT,
            zone_id TEXT,
            crop_type TEXT,
            status TEXT DEFAULT 'active',
            total_detections INTEGER DEFAULT 0,
            health_score REAL DEFAULT 100.0,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS crop_detections (
            id TEXT PRIMARY KEY,
            mission_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            detection_class TEXT NOT NULL,
            display_name TEXT,
            confidence REAL NOT NULL,
            severity TEXT DEFAULT 'low',
            health_impact REAL DEFAULT 0.0,
            bbox_x1 REAL, bbox_y1 REAL,
            bbox_x2 REAL, bbox_y2 REAL,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            field_id TEXT,
            zone_id TEXT,
            image_path TEXT,
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
            first_seen TEXT,
            times_detected INTEGER DEFAULT 1,
            FOREIGN KEY (mission_id) REFERENCES ag_missions(id)
        );

        CREATE TABLE IF NOT EXISTS zone_health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            field_id TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            health_score REAL NOT NULL,
            detection_count INTEGER,
            high_severity_count INTEGER,
            avg_humidity REAL,
            avg_temperature REAL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS environmental_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            mission_id TEXT,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            temperature_c REAL,
            humidity_pct REAL,
            pressure_hpa REAL,
            fungal_risk TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_crop_det_mission ON crop_detections(mission_id);
        CREATE INDEX IF NOT EXISTS idx_crop_det_zone ON crop_detections(zone_id);
        CREATE INDEX IF NOT EXISTS idx_crop_det_class ON crop_detections(detection_class);
        CREATE INDEX IF NOT EXISTS idx_env_mission ON environmental_readings(mission_id);
        CREATE INDEX IF NOT EXISTS idx_zone_health ON zone_health_history(zone_id, timestamp);
    """)
    conn.commit()
    return conn


def load_farm_config(config_path: str) -> dict:
    """Load farm configuration from YAML."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def point_in_polygon(lat: float, lon: float, polygon: list) -> bool:
    """Ray-casting algorithm for GPS point-in-polygon check."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][0] > lat) != (polygon[j][0] > lat)) and \
           (lon < (polygon[j][1] - polygon[i][1]) * (lat - polygon[i][0]) /
            (polygon[j][0] - polygon[i][0]) + polygon[i][1]):
            inside = not inside
        j = i
    return inside


def find_zone(lat: float, lon: float, config: dict) -> tuple:
    """Determine which field and zone a GPS coordinate falls in."""
    for field in config.get("fields", []):
        if point_in_polygon(lat, lon, field.get("boundary", [])):
            for zone in field.get("zones", []):
                if point_in_polygon(lat, lon, zone.get("boundary", [])):
                    return field["id"], zone["id"], field.get("crop", "unknown")
            return field["id"], None, field.get("crop", "unknown")
    return None, None, "unknown"


class AgPerception(Node):
    """
    Agricultural perception and crop health analysis node.
    
    Core logic:
    1. Receives detections from YOLOv8 (retrained on ag classes)
    2. Locates detection in a field/zone using GPS + farm config
    3. Correlates with environmental data (humidity → fungal risk)
    4. Checks for persistent anomalies (same location across flights)
    5. Triggers VLM analysis with ag-specific prompts for high-severity
    6. Publishes structured crop health alerts
    7. Maintains zone-level health scores over time
    """

    def __init__(self):
        super().__init__("ag_perception")
        self.get_logger().info("Agricultural Perception Node starting")

        # Load config and database
        self.config = load_farm_config(CONFIG_PATH)
        self.db = init_ag_db(DB_PATH)
        self.bridge = CvBridge()
        os.makedirs(IMAGE_DIR, exist_ok=True)

        # Detection class config
        self.detection_classes = {}
        for cls in self.config.get("detection", {}).get("classes", []):
            self.detection_classes[cls["name"]] = cls

        # VLM prompts
        self.vlm_config = self.config.get("vlm", {})
        self.env_thresholds = self.config.get("environmental", {})

        # State
        self.current_mission_id = None
        self.latest_image = None
        self.latest_gps = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.latest_env = {"temperature": 0.0, "humidity": 0.0, "pressure": 0.0, "fungal_risk": "unknown"}
        self.zone_detection_counts = {}  # zone_id → {class → count}
        self.mission_field_id = None
        self.mission_zone_id = None

        # Start mission
        self._start_mission()

        # Publishers
        self.alert_pub = self.create_publisher(String, CROP_ALERT_TOPIC, 10)
        self.report_pub = self.create_publisher(String, SCOUTING_REPORT_TOPIC, 10)
        self.vlm_pub = self.create_publisher(String, VLM_REQUEST_TOPIC, 10)

        # Subscribers — passive listeners on existing topics
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, 1)
        self.create_subscription(String, DETECTION_TOPIC, self._on_detection, 10)
        self.create_subscription(String, TEMPORAL_TOPIC, self._on_temporal, 10)
        self.create_subscription(String, ENVIRONMENTAL_TOPIC, self._on_environmental, 10)
        self.create_subscription(String, GPS_TOPIC, self._on_gps, 10)

        # Periodic zone health scoring (every 60 seconds)
        self.create_timer(60.0, self._compute_zone_health)

        self.get_logger().info(f"Farm config loaded: {len(self.config.get('fields', []))} fields")
        self.get_logger().info(f"Detection classes: {list(self.detection_classes.keys())}")

    def _start_mission(self, field_id=None, zone_id=None):
        """Create a new agricultural mission record."""
        self.current_mission_id = f"ag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mission_field_id = field_id
        self.mission_zone_id = zone_id
        now = datetime.now(timezone.utc).isoformat()

        crop = "unknown"
        if field_id:
            for f in self.config.get("fields", []):
                if f["id"] == field_id:
                    crop = f.get("crop", "unknown")
                    break

        self.db.execute(
            """INSERT INTO ag_missions (id, start_time, field_id, zone_id, crop_type)
               VALUES (?, ?, ?, ?, ?)""",
            (self.current_mission_id, now, field_id, zone_id, crop)
        )
        self.db.commit()
        self.get_logger().info(f"Mission started: {self.current_mission_id} | Field: {field_id} | Zone: {zone_id}")

    def _on_image(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _on_gps(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_gps = {
                "lat": data.get("latitude", 0.0),
                "lon": data.get("longitude", 0.0),
                "alt": data.get("altitude", 0.0),
            }
        except Exception:
            pass

    def _on_environmental(self, msg):
        """Receive environmental sensor readings from BME280 node."""
        try:
            data = json.loads(msg.data)
            self.latest_env = {
                "temperature": data.get("temperature_c", 0.0),
                "humidity": data.get("humidity_pct", 0.0),
                "pressure": data.get("pressure_hpa", 0.0),
                "fungal_risk": self._assess_fungal_risk(
                    data.get("humidity_pct", 0.0),
                    data.get("temperature_c", 0.0)
                ),
            }

            # Log environmental reading with GPS
            self.db.execute(
                """INSERT INTO environmental_readings
                   (timestamp, mission_id, latitude, longitude, altitude,
                    temperature_c, humidity_pct, pressure_hpa, fungal_risk)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    self.current_mission_id,
                    self.latest_gps["lat"], self.latest_gps["lon"], self.latest_gps["alt"],
                    data.get("temperature_c"), data.get("humidity_pct"),
                    data.get("pressure_hpa"), self.latest_env["fungal_risk"],
                )
            )
            self.db.commit()

        except Exception as e:
            self.get_logger().warn(f"Environmental parse error: {e}")

    def _assess_fungal_risk(self, humidity: float, temperature: float) -> str:
        """Assess fungal disease risk based on environmental conditions."""
        thresholds = self.env_thresholds.get("humidity", {})
        high_thresh = thresholds.get("fungal_risk_threshold", 85)
        mod_thresh = thresholds.get("moderate_risk_threshold", 70)

        # Fungal risk increases with humidity AND warm temps (15-30°C)
        temp_favorable = 15 <= temperature <= 30
        
        if humidity >= high_thresh and temp_favorable:
            return "high"
        elif humidity >= high_thresh or (humidity >= mod_thresh and temp_favorable):
            return "moderate"
        elif humidity >= mod_thresh:
            return "low"
        return "minimal"

    def _on_detection(self, msg):
        """
        Process agricultural detection from YOLOv8.
        Core agricultural perception logic.
        """
        try:
            data = json.loads(msg.data)
            det_class = data.get("class", "unknown")
            confidence = data.get("confidence", 0.0)

            # Look up class configuration
            class_config = self.detection_classes.get(det_class, {})
            threshold = class_config.get("confidence_threshold", 0.5)

            # Skip low-confidence or healthy detections
            if confidence < threshold:
                return
            if det_class == "healthy":
                return  # Don't alert on healthy — but still log for training

            # Determine field/zone from GPS
            field_id, zone_id, crop_type = find_zone(
                self.latest_gps["lat"], self.latest_gps["lon"], self.config
            )

            # Check for persistent anomaly (same class near same location in past flights)
            is_persistent, times_seen, first_seen = self._check_persistence(
                det_class, self.latest_gps["lat"], self.latest_gps["lon"]
            )

            # Save detection image
            image_path = None
            if self.latest_image is not None:
                det_id = str(uuid.uuid4())[:12]
                image_filename = f"{self.current_mission_id}_{det_id}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                cv2.imwrite(image_path, self.latest_image)
            else:
                det_id = str(uuid.uuid4())[:12]

            # Calculate health impact score (0-100, higher = worse)
            severity = class_config.get("severity", "low")
            health_impact = self._calculate_health_impact(
                severity, confidence, is_persistent, times_seen
            )

            # Write to database
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute(
                """INSERT INTO crop_detections
                   (id, mission_id, timestamp, detection_class, display_name,
                    confidence, severity, health_impact,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    latitude, longitude, altitude,
                    field_id, zone_id, image_path,
                    env_temperature, env_humidity, env_pressure, env_fungal_risk,
                    is_persistent, first_seen, times_detected)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    det_id, self.current_mission_id, now,
                    det_class, class_config.get("display", det_class),
                    confidence, severity, health_impact,
                    data.get("bbox", [0,0,0,0])[0], data.get("bbox", [0,0,0,0])[1],
                    data.get("bbox", [0,0,0,0])[2], data.get("bbox", [0,0,0,0])[3],
                    self.latest_gps["lat"], self.latest_gps["lon"], self.latest_gps["alt"],
                    field_id, zone_id, image_path,
                    self.latest_env["temperature"], self.latest_env["humidity"],
                    self.latest_env["pressure"], self.latest_env["fungal_risk"],
                    1 if is_persistent else 0, first_seen, times_seen,
                )
            )

            self.db.execute(
                "UPDATE ag_missions SET total_detections = total_detections + 1 WHERE id = ?",
                (self.current_mission_id,)
            )
            self.db.commit()

            # Track zone-level detection counts
            if zone_id:
                if zone_id not in self.zone_detection_counts:
                    self.zone_detection_counts[zone_id] = {}
                counts = self.zone_detection_counts[zone_id]
                counts[det_class] = counts.get(det_class, 0) + 1

            # Publish crop alert
            alert = {
                "id": det_id,
                "type": "crop_alert",
                "timestamp": now,
                "detection_class": det_class,
                "display_name": class_config.get("display", det_class),
                "confidence": confidence,
                "severity": severity,
                "health_impact": health_impact,
                "latitude": self.latest_gps["lat"],
                "longitude": self.latest_gps["lon"],
                "field_id": field_id,
                "zone_id": zone_id,
                "crop_type": crop_type,
                "environmental": self.latest_env,
                "is_persistent": is_persistent,
                "times_detected": times_seen,
                "image_path": image_path,
            }

            alert_msg = String()
            alert_msg.data = json.dumps(alert)
            self.alert_pub.publish(alert_msg)

            # Trigger VLM analysis for high-severity or persistent detections
            if severity == "high" or (is_persistent and times_seen >= 3):
                self._request_vlm_analysis(det_id, alert, crop_type)

            self.get_logger().info(
                f"Crop alert: {class_config.get('display', det_class)} "
                f"({confidence:.0%}) | Zone: {zone_id} | "
                f"Fungal risk: {self.latest_env['fungal_risk']} | "
                f"{'PERSISTENT' if is_persistent else 'new'}"
            )

        except Exception as e:
            self.get_logger().error(f"Detection processing error: {e}")

    def _on_temporal(self, msg):
        """Handle temporal pattern events — patterns across time, not just single frames."""
        try:
            data = json.loads(msg.data)
            # Temporal events indicate sustained patterns — treat as higher confidence
            data["confidence"] = min(data.get("confidence", 0.5) * 1.2, 0.99)
            
            # Re-publish as detection for unified processing
            det_msg = String()
            det_msg.data = json.dumps(data)
            self._on_detection(det_msg)
        except Exception:
            pass

    def _check_persistence(self, det_class: str, lat: float, lon: float,
                           radius_m: float = 30.0) -> tuple:
        """
        Check if this type of detection has been seen near this location before.
        This is the ag equivalent of "pattern of life" analysis —
        a disease that shows up in the same spot across multiple flights is more
        concerning than a one-time false positive.
        """
        # Approximate degree-to-meter conversion for Ohio latitude
        lat_delta = radius_m / 111000
        lon_delta = radius_m / (111000 * math.cos(math.radians(lat)))

        rows = self.db.execute(
            """SELECT MIN(timestamp) as first_seen, COUNT(*) as times_seen
               FROM crop_detections
               WHERE detection_class = ?
                 AND latitude BETWEEN ? AND ?
                 AND longitude BETWEEN ? AND ?
                 AND mission_id != ?""",
            (det_class,
             lat - lat_delta, lat + lat_delta,
             lon - lon_delta, lon + lon_delta,
             self.current_mission_id)
        ).fetchone()

        if rows and rows[1] > 0:
            return True, rows[1] + 1, rows[0]
        return False, 1, datetime.now(timezone.utc).isoformat()

    def _calculate_health_impact(self, severity: str, confidence: float,
                                  is_persistent: bool, times_seen: int) -> float:
        """
        Calculate a 0-100 health impact score for this detection.
        Higher = worse for crop health.
        """
        base_scores = {"high": 70, "medium": 40, "low": 15, "none": 0}
        score = base_scores.get(severity, 20)

        # Scale by confidence
        score *= confidence

        # Persistent issues are worse
        if is_persistent:
            score *= (1 + min(times_seen * 0.1, 0.5))

        # Environmental correlation boosts score
        if self.latest_env["fungal_risk"] == "high" and severity == "high":
            score *= 1.3  # High humidity + disease detection = very concerning

        return min(round(score, 1), 100.0)

    def _request_vlm_analysis(self, det_id: str, alert: dict, crop_type: str):
        """Request Cosmos Reason 2 analysis with agriculture-specific prompts."""
        query_template = self.vlm_config.get("query_template", "Analyze this image.")

        # Build prior detection context for the VLM
        prior = ""
        if alert.get("zone_id"):
            counts = self.zone_detection_counts.get(alert["zone_id"], {})
            if counts:
                prior = ", ".join([f"{v}x {k}" for k, v in counts.items()])
            else:
                prior = "none"

        query = query_template.format(
            crop_type=crop_type,
            altitude=f"{alert.get('latitude', 0):.1f}",
            temperature=self.latest_env["temperature"],
            humidity=self.latest_env["humidity"],
            pressure=self.latest_env["pressure"],
            prior_detections=prior,
            detection_class=alert.get("display_name", "unknown"),
            confidence=f"{alert.get('confidence', 0)*100:.0f}",
        )

        vlm_request = {
            "detection_id": det_id,
            "image_path": alert.get("image_path"),
            "system_prompt": self.vlm_config.get("system_prompt", ""),
            "query": query,
            "callback_topic": CROP_ALERT_TOPIC,
        }

        msg = String()
        msg.data = json.dumps(vlm_request)
        self.vlm_pub.publish(msg)

        self.get_logger().info(f"VLM analysis requested for detection {det_id}")

    def _compute_zone_health(self):
        """
        Periodic computation of zone-level health scores.
        This gives the farmer a single number per zone: "how healthy is this area?"
        100 = perfect, 0 = disaster.
        """
        if not self.current_mission_id:
            return

        for field in self.config.get("fields", []):
            for zone in field.get("zones", []):
                zone_id = zone["id"]

                # Get all detections for this zone in the current mission
                rows = self.db.execute(
                    """SELECT severity, health_impact, COUNT(*) as count
                       FROM crop_detections
                       WHERE zone_id = ? AND mission_id = ?
                       GROUP BY severity""",
                    (zone_id, self.current_mission_id)
                ).fetchall()

                if not rows:
                    continue

                # Calculate zone health score
                total_impact = sum(r[1] * r[2] for r in rows)
                total_detections = sum(r[2] for r in rows)
                high_count = sum(r[2] for r in rows if r[0] == "high")

                # Health score: start at 100, subtract weighted impact
                health = max(0, 100 - (total_impact / max(total_detections, 1)))

                # Get average environmental conditions
                env = self.db.execute(
                    """SELECT AVG(humidity_pct), AVG(temperature_c)
                       FROM environmental_readings
                       WHERE mission_id = ?""",
                    (self.current_mission_id,)
                ).fetchone()

                # Record zone health history
                self.db.execute(
                    """INSERT INTO zone_health_history
                       (timestamp, field_id, zone_id, health_score,
                        detection_count, high_severity_count,
                        avg_humidity, avg_temperature)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        field["id"], zone_id, health,
                        total_detections, high_count,
                        env[0] if env else None, env[1] if env else None,
                    )
                )
                self.db.commit()

    def generate_scouting_report(self):
        """
        Generate a structured scouting report at end of mission.
        Published to the scouting report topic and saved to database.
        Farmer reads this on the dashboard as a summary.
        """
        report = {
            "mission_id": self.current_mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "field_id": self.mission_field_id,
            "zones": [],
        }

        for field in self.config.get("fields", []):
            if self.mission_field_id and field["id"] != self.mission_field_id:
                continue

            for zone in field.get("zones", []):
                zone_data = {
                    "zone_id": zone["id"],
                    "zone_name": zone["name"],
                    "detections": [],
                    "health_score": 100.0,
                    "top_concern": None,
                    "environmental_summary": {},
                    "recommendation": "No issues detected.",
                }

                # Get detections for this zone
                rows = self.db.execute(
                    """SELECT detection_class, display_name, severity,
                              COUNT(*) as count, AVG(confidence) as avg_conf,
                              AVG(health_impact) as avg_impact
                       FROM crop_detections
                       WHERE zone_id = ? AND mission_id = ?
                       GROUP BY detection_class
                       ORDER BY avg_impact DESC""",
                    (zone["id"], self.current_mission_id)
                ).fetchall()

                for r in rows:
                    zone_data["detections"].append({
                        "class": r[0],
                        "display_name": r[1],
                        "severity": r[2],
                        "count": r[3],
                        "avg_confidence": round(r[4], 2),
                        "avg_health_impact": round(r[5], 1),
                    })

                if rows:
                    zone_data["top_concern"] = rows[0][1]  # Highest impact class
                    total_impact = sum(r[5] * r[3] for r in rows)
                    total_count = sum(r[3] for r in rows)
                    zone_data["health_score"] = round(
                        max(0, 100 - total_impact / max(total_count, 1)), 1
                    )

                    # Generate actionable recommendation
                    if rows[0][2] == "high":
                        zone_data["recommendation"] = (
                            f"URGENT: {rows[0][1]} detected ({rows[0][3]} instances). "
                            f"Scout this zone on foot immediately. "
                            f"Fungal risk is {self.latest_env['fungal_risk']}."
                        )
                    elif rows[0][2] == "medium":
                        zone_data["recommendation"] = (
                            f"Monitor: {rows[0][1]} detected ({rows[0][3]} instances). "
                            f"Schedule ground scouting within 3-5 days."
                        )

                # Environmental summary
                env = self.db.execute(
                    """SELECT AVG(humidity_pct), AVG(temperature_c), AVG(pressure_hpa)
                       FROM environmental_readings WHERE mission_id = ?""",
                    (self.current_mission_id,)
                ).fetchone()

                if env and env[0]:
                    zone_data["environmental_summary"] = {
                        "avg_humidity": round(env[0], 1),
                        "avg_temperature_c": round(env[1], 1),
                        "avg_pressure_hpa": round(env[2], 1),
                        "fungal_risk": self._assess_fungal_risk(env[0], env[1]),
                    }

                report["zones"].append(zone_data)

        msg = String()
        msg.data = json.dumps(report)
        self.report_pub.publish(msg)

        self.get_logger().info(
            f"Scouting report generated: {len(report['zones'])} zones"
        )
        return report

    def destroy_node(self):
        self.generate_scouting_report()
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE ag_missions SET end_time = ?, status = 'complete' WHERE id = ?",
            (now, self.current_mission_id)
        )
        self.db.commit()
        self.db.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AgPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down ag perception...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
