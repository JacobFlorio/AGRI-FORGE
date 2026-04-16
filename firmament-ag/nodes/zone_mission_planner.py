"""
Florio Industries — Zone Mission Planner
==========================================
Generates autonomous waypoint missions for farmer-selected field zones.
The farmer says "scan NW Quarter" and this node generates the flight plan.

Agricultural grid scanning patterns:
  - Lawnmower pattern optimized for image overlap (photogrammetry)
  - Altitude adjusted for crop type and growth stage
  - Speed optimized for detection FPS vs coverage tradeoff
  - Automatic RTL on low battery or zone completion

Reads: config/farm_config.yaml for zone boundaries and mission params
Publishes: /ag/mission_waypoints — waypoint list for flight controller
Subscribes: /ag/mission_command — receives dispatch commands from dashboard

Usage from the dashboard:
  POST /api/mission/dispatch
  {"field_id": "field_a", "zone_id": "field_a_nw", "mode": "scouting"}

Modes:
  "scouting"       — standard grid at detection altitude (25m default)
  "photogrammetry" — dense grid at mapping altitude with high overlap
  "spot_check"     — fly to specific GPS point and hover for VLM analysis
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import math
import yaml
import os
from datetime import datetime, timezone

CONFIG_PATH = os.path.expanduser("~/firmament-ag/config/farm_config.yaml")
WAYPOINT_TOPIC = "/ag/mission_waypoints"
COMMAND_TOPIC = "/ag/mission_command"
STATUS_TOPIC = "/ag/mission_status"


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distance in meters between two GPS points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def polygon_bounds(polygon: list) -> tuple:
    """Get bounding box of a polygon: (min_lat, max_lat, min_lon, max_lon)."""
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    return min(lats), max(lats), min(lons), max(lons)


def point_in_polygon(lat, lon, polygon):
    """Ray-casting point-in-polygon test."""
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


def generate_lawnmower_waypoints(
    polygon: list,
    altitude_m: float,
    line_spacing_m: float,
    speed_m_s: float,
    camera_trigger_m: float,
) -> list:
    """
    Generate a lawnmower (boustrophedon) pattern within a polygon boundary.
    
    This is the standard agricultural drone scanning pattern:
    - Parallel lines covering the field
    - Alternating direction (serpentine) for efficient coverage
    - Line spacing calculated from camera FOV and desired overlap
    - Only waypoints inside the polygon boundary are included
    
    Returns list of waypoints: [{lat, lon, alt, speed, action}]
    """
    min_lat, max_lat, min_lon, max_lon = polygon_bounds(polygon)

    # Convert line spacing from meters to degrees (approximate for Ohio ~40°N)
    lat_per_meter = 1 / 111000
    lon_per_meter = 1 / (111000 * math.cos(math.radians((min_lat + max_lat) / 2)))

    line_spacing_deg = line_spacing_m * lon_per_meter  # Spacing in longitude
    point_spacing_deg = camera_trigger_m * lat_per_meter  # Along-track spacing

    waypoints = []
    current_lon = min_lon
    line_index = 0

    while current_lon <= max_lon:
        line_points = []

        # Generate points along this line
        if line_index % 2 == 0:
            # South to North
            current_lat = min_lat
            while current_lat <= max_lat:
                if point_in_polygon(current_lat, current_lon, polygon):
                    line_points.append({
                        "latitude": round(current_lat, 8),
                        "longitude": round(current_lon, 8),
                        "altitude_m": altitude_m,
                        "speed_m_s": speed_m_s,
                        "action": "photo",
                    })
                current_lat += point_spacing_deg
        else:
            # North to South (serpentine)
            current_lat = max_lat
            while current_lat >= min_lat:
                if point_in_polygon(current_lat, current_lon, polygon):
                    line_points.append({
                        "latitude": round(current_lat, 8),
                        "longitude": round(current_lon, 8),
                        "altitude_m": altitude_m,
                        "speed_m_s": speed_m_s,
                        "action": "photo",
                    })
                current_lat -= point_spacing_deg

        waypoints.extend(line_points)
        current_lon += line_spacing_deg
        line_index += 1

    return waypoints


def estimate_mission_time(waypoints: list, speed_m_s: float) -> dict:
    """Estimate total mission time and distance."""
    total_distance = 0
    for i in range(1, len(waypoints)):
        total_distance += haversine_m(
            waypoints[i - 1]["latitude"], waypoints[i - 1]["longitude"],
            waypoints[i]["latitude"], waypoints[i]["longitude"]
        )

    flight_time_s = total_distance / max(speed_m_s, 0.1)

    return {
        "total_waypoints": len(waypoints),
        "total_distance_m": round(total_distance, 1),
        "estimated_flight_time_s": round(flight_time_s, 1),
        "estimated_flight_time_min": round(flight_time_s / 60, 1),
    }


class ZoneMissionPlanner(Node):
    """
    Generates and publishes waypoint missions for agricultural scanning.
    
    The farmer selects a field and zone from the dashboard.
    This node generates the optimal flight pattern and publishes
    waypoints for the flight controller to execute.
    """

    def __init__(self):
        super().__init__("zone_mission_planner")
        self.get_logger().info("Zone Mission Planner starting")

        self.config = load_config()
        self.mission_params = self.config.get("mission", {})

        # Build lookup for fields and zones
        self.fields = {}
        self.zones = {}
        for field in self.config.get("fields", []):
            self.fields[field["id"]] = field
            for zone in field.get("zones", []):
                self.zones[zone["id"]] = {**zone, "field_id": field["id"], "crop": field.get("crop")}

        # Publishers
        self.waypoint_pub = self.create_publisher(String, WAYPOINT_TOPIC, 10)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        # Command subscriber (receives dispatch commands from dashboard)
        self.create_subscription(String, COMMAND_TOPIC, self._on_command, 10)

        self.get_logger().info(
            f"Ready — {len(self.fields)} fields, {len(self.zones)} zones"
        )

    def _on_command(self, msg):
        """Handle mission dispatch commands."""
        try:
            cmd = json.loads(msg.data)
            mode = cmd.get("mode", "scouting")
            field_id = cmd.get("field_id")
            zone_id = cmd.get("zone_id")

            if mode == "spot_check":
                self._generate_spot_check(cmd)
            elif zone_id:
                self._generate_zone_mission(zone_id, mode)
            elif field_id:
                self._generate_field_mission(field_id, mode)
            else:
                self._publish_status("error", "No field or zone specified")

        except Exception as e:
            self.get_logger().error(f"Command error: {e}")
            self._publish_status("error", str(e))

    def _generate_zone_mission(self, zone_id: str, mode: str):
        """Generate a scanning mission for a single zone."""
        zone = self.zones.get(zone_id)
        if not zone:
            self._publish_status("error", f"Zone {zone_id} not found")
            return

        self.get_logger().info(f"Planning {mode} mission for zone: {zone['name']}")

        # Calculate line spacing from camera FOV and overlap
        altitude = self.mission_params.get("default_altitude_m", 25)
        overlap = self.mission_params.get("overlap_percent", 70)
        sidelap = self.mission_params.get("sidelap_percent", 65)
        speed = self.mission_params.get("speed_m_s", 5)
        trigger_interval = self.mission_params.get("camera_trigger_interval_m", 8)

        if mode == "photogrammetry":
            # Tighter grid for 3D reconstruction
            altitude = max(15, altitude - 5)  # Lower for more detail
            overlap = 80
            sidelap = 75
            speed = 3  # Slower for sharper images
            trigger_interval = 5

        # SIYI A8 Mini FOV is ~85° horizontal
        # At 25m altitude: ground width ≈ 2 * 25 * tan(42.5°) ≈ 46m
        camera_fov_deg = 85
        ground_width = 2 * altitude * math.tan(math.radians(camera_fov_deg / 2))
        line_spacing = ground_width * (1 - sidelap / 100)

        # Generate waypoints
        waypoints = generate_lawnmower_waypoints(
            polygon=zone["boundary"],
            altitude_m=altitude,
            line_spacing_m=line_spacing,
            speed_m_s=speed,
            camera_trigger_m=trigger_interval,
        )

        if not waypoints:
            self._publish_status("error", "No valid waypoints generated")
            return

        # Estimate mission time
        estimate = estimate_mission_time(waypoints, speed)

        # Check against battery constraints
        max_flight_min = 25  # Conservative estimate for your dual 6S setup
        if estimate["estimated_flight_time_min"] > max_flight_min:
            self.get_logger().warn(
                f"Mission exceeds battery: {estimate['estimated_flight_time_min']:.0f} min "
                f"vs {max_flight_min} min limit. Consider splitting into sorties."
            )

        # Package mission
        mission = {
            "type": "zone_scan",
            "mode": mode,
            "mission_id": f"m_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "field_id": zone.get("field_id"),
            "zone_id": zone_id,
            "zone_name": zone["name"],
            "crop": zone.get("crop", "unknown"),
            "parameters": {
                "altitude_m": altitude,
                "speed_m_s": speed,
                "line_spacing_m": round(line_spacing, 1),
                "overlap_pct": overlap,
                "sidelap_pct": sidelap,
            },
            "waypoints": waypoints,
            "estimate": estimate,
            "return_to_launch": self.mission_params.get("return_to_launch", True),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Publish waypoints
        msg = String()
        msg.data = json.dumps(mission)
        self.waypoint_pub.publish(msg)

        self._publish_status("planned", (
            f"{zone['name']}: {estimate['total_waypoints']} waypoints, "
            f"{estimate['estimated_flight_time_min']:.0f} min, "
            f"{estimate['total_distance_m']:.0f}m"
        ))

        self.get_logger().info(
            f"Mission planned: {estimate['total_waypoints']} waypoints | "
            f"{estimate['estimated_flight_time_min']:.0f} min | "
            f"{estimate['total_distance_m']:.0f}m distance"
        )

    def _generate_field_mission(self, field_id: str, mode: str):
        """Generate missions for all zones in a field (multi-sortie if needed)."""
        field = self.fields.get(field_id)
        if not field:
            self._publish_status("error", f"Field {field_id} not found")
            return

        for zone in field.get("zones", []):
            self._generate_zone_mission(zone["id"], mode)

    def _generate_spot_check(self, cmd: dict):
        """Generate a single-point mission to hover and analyze a specific location."""
        lat = cmd.get("latitude")
        lon = cmd.get("longitude")
        alt = cmd.get("altitude_m", 15)  # Lower for spot checks

        if not lat or not lon:
            self._publish_status("error", "Spot check requires latitude and longitude")
            return

        mission = {
            "type": "spot_check",
            "mode": "spot_check",
            "mission_id": f"spot_{datetime.now().strftime('%H%M%S')}",
            "waypoints": [
                {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude_m": alt,
                    "speed_m_s": 3,
                    "action": "hover_and_analyze",
                    "hover_duration_s": 30,
                }
            ],
            "estimate": {
                "total_waypoints": 1,
                "estimated_flight_time_min": 2,
            },
            "return_to_launch": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        msg = String()
        msg.data = json.dumps(mission)
        self.waypoint_pub.publish(msg)

        self._publish_status("planned", f"Spot check at {lat:.6f}, {lon:.6f}")

    def _publish_status(self, status: str, message: str):
        msg = String()
        msg.data = json.dumps({
            "status": status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.status_pub.publish(msg)


# ─── STANDALONE CLI ─────────────────────────────────────────────
# Can also be used as a command-line tool to preview missions

def main(args=None):
    rclpy.init(args=args)
    node = ZoneMissionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down mission planner...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        # Preview mode: generate and print mission without ROS
        config = load_config()
        params = config.get("mission", {})

        zone_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not zone_id:
            print("Available zones:")
            for field in config.get("fields", []):
                print(f"\n  Field: {field['name']} ({field['id']})")
                for zone in field.get("zones", []):
                    print(f"    Zone: {zone['name']} ({zone['id']})")
            sys.exit(0)

        # Find the zone
        target_zone = None
        for field in config.get("fields", []):
            for zone in field.get("zones", []):
                if zone["id"] == zone_id:
                    target_zone = zone
                    break

        if not target_zone:
            print(f"Zone {zone_id} not found")
            sys.exit(1)

        altitude = params.get("default_altitude_m", 25)
        ground_width = 2 * altitude * math.tan(math.radians(42.5))
        line_spacing = ground_width * (1 - params.get("sidelap_percent", 65) / 100)

        waypoints = generate_lawnmower_waypoints(
            polygon=target_zone["boundary"],
            altitude_m=altitude,
            line_spacing_m=line_spacing,
            speed_m_s=params.get("speed_m_s", 5),
            camera_trigger_m=params.get("camera_trigger_interval_m", 8),
        )

        estimate = estimate_mission_time(waypoints, params.get("speed_m_s", 5))

        print(f"\nMission Preview: {target_zone['name']}")
        print(f"  Waypoints:     {estimate['total_waypoints']}")
        print(f"  Distance:      {estimate['total_distance_m']:.0f} m")
        print(f"  Flight time:   {estimate['estimated_flight_time_min']:.1f} min")
        print(f"  Altitude:      {altitude} m")
        print(f"  Line spacing:  {line_spacing:.1f} m")
        print(f"\nFirst 5 waypoints:")
        for wp in waypoints[:5]:
            print(f"  {wp['latitude']:.8f}, {wp['longitude']:.8f}")
    else:
        main()
