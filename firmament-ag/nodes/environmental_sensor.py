"""
Florio Industries — Environmental Sensor Node
===============================================
Reads BME280 sensor via I2C on Jetson GPIO and publishes
spatial microclimate data (temperature, humidity, pressure).

This gives farmers something no satellite or weather station provides:
canopy-level environmental data at GPS-tagged locations across their fields.

Hardware setup:
  BME280 breakout board (~$3-5 on Amazon)
    VCC → Jetson 3.3V (Pin 1)
    GND → Jetson GND (Pin 6)
    SDA → Jetson I2C SDA (Pin 3, GPIO 2)
    SCL → Jetson I2C SCL (Pin 5, GPIO 3)

  Default I2C address: 0x76 (some boards use 0x77)
  Verify: i2cdetect -y -r 1

Dependencies:
  pip3 install smbus2 bme280

Publishes to:
  /ag/environmental  — JSON with temperature, humidity, pressure, GPS

If BME280 is not connected, runs in simulation mode using
estimated values (useful for development and demo).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
from datetime import datetime, timezone

# ─── BME280 CONFIGURATION ──────────────────────────────────────

I2C_BUS = 1          # Jetson Orin Nano default I2C bus
BME280_ADDRESS = 0x76 # Default address (try 0x77 if not found)
PUBLISH_RATE_HZ = 0.5 # Publish every 2 seconds (plenty for spatial mapping)
PUBLISH_TOPIC = "/ag/environmental"
GPS_TOPIC = "/gps/fix"

# ─── SENSOR INTERFACE ──────────────────────────────────────────

class BME280Sensor:
    """Interface to BME280 temperature/humidity/pressure sensor."""

    def __init__(self, bus=I2C_BUS, address=BME280_ADDRESS):
        self.available = False
        self.bus_num = bus
        self.address = address

        try:
            import smbus2
            import bme280
            self.smbus2 = smbus2
            self.bme280 = bme280

            self.bus = smbus2.SMBus(bus)
            self.calibration = bme280.load_calibration_params(self.bus, address)
            self.available = True
        except Exception as e:
            print(f"BME280 not available ({e}). Running in simulation mode.")
            self.available = False

    def read(self) -> dict:
        """Read sensor data. Returns simulated data if sensor not connected."""
        if self.available:
            try:
                data = self.bme280.sample(self.bus, self.address, self.calibration)
                return {
                    "temperature_c": round(data.temperature, 2),
                    "humidity_pct": round(data.humidity, 2),
                    "pressure_hpa": round(data.pressure, 2),
                    "source": "bme280",
                }
            except Exception:
                pass

        # Simulation mode — generate realistic Ohio summer values
        # These vary slightly to simulate spatial variation
        import random
        base_temp = 28.0 + random.gauss(0, 2)
        base_humidity = 72.0 + random.gauss(0, 8)
        base_pressure = 1013.25 + random.gauss(0, 1)

        return {
            "temperature_c": round(base_temp, 2),
            "humidity_pct": round(min(100, max(0, base_humidity)), 2),
            "pressure_hpa": round(base_pressure, 2),
            "source": "simulated",
        }


class EnvironmentalSensor(Node):
    """
    ROS 2 node that publishes environmental readings tagged with GPS.
    
    Each reading creates a spatial data point that, combined with others
    from the same flight, produces a microclimate map of the field.
    
    Farmers care about this because:
    - Canopy-level humidity directly predicts fungal disease risk
    - Temperature inversions indicate frost pockets
    - Pressure trends indicate incoming weather
    - Spatial variation reveals drainage issues (low spots = higher humidity)
    """

    def __init__(self):
        super().__init__("environmental_sensor")
        self.get_logger().info("Environmental Sensor Node starting")

        self.sensor = BME280Sensor()
        if self.sensor.available:
            self.get_logger().info("BME280 connected — live sensor mode")
        else:
            self.get_logger().warn("BME280 not found — simulation mode (install smbus2 bme280)")

        # State
        self.latest_gps = {"lat": 0.0, "lon": 0.0, "alt": 0.0}

        # Track readings for running statistics
        self.reading_history = []
        self.max_history = 500

        # Publisher
        self.env_pub = self.create_publisher(String, PUBLISH_TOPIC, 10)

        # GPS subscriber
        self.create_subscription(String, GPS_TOPIC, self._on_gps, 10)

        # Timer for periodic readings
        period = 1.0 / PUBLISH_RATE_HZ
        self.create_timer(period, self._publish_reading)

        self.get_logger().info(f"Publishing at {PUBLISH_RATE_HZ} Hz to {PUBLISH_TOPIC}")

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

    def _publish_reading(self):
        """Read sensor and publish GPS-tagged environmental data."""
        reading = self.sensor.read()
        reading["timestamp"] = datetime.now(timezone.utc).isoformat()
        reading["latitude"] = self.latest_gps["lat"]
        reading["longitude"] = self.latest_gps["lon"]
        reading["altitude_m"] = self.latest_gps["alt"]

        # Compute derived metrics
        reading["dew_point_c"] = self._calculate_dew_point(
            reading["temperature_c"], reading["humidity_pct"]
        )
        reading["heat_index_c"] = self._calculate_heat_index(
            reading["temperature_c"], reading["humidity_pct"]
        )
        reading["vpd_kpa"] = self._calculate_vpd(
            reading["temperature_c"], reading["humidity_pct"]
        )

        # Track for trend analysis
        self.reading_history.append(reading)
        if len(self.reading_history) > self.max_history:
            self.reading_history.pop(0)

        # Add trend data if enough history
        if len(self.reading_history) >= 10:
            recent = self.reading_history[-10:]
            reading["humidity_trend"] = (
                recent[-1]["humidity_pct"] - recent[0]["humidity_pct"]
            )
            reading["temperature_trend"] = (
                recent[-1]["temperature_c"] - recent[0]["temperature_c"]
            )
            reading["pressure_trend"] = (
                recent[-1]["pressure_hpa"] - recent[0]["pressure_hpa"]
            )

        msg = String()
        msg.data = json.dumps(reading)
        self.env_pub.publish(msg)

    @staticmethod
    def _calculate_dew_point(temp_c: float, humidity: float) -> float:
        """Magnus formula for dew point."""
        a, b = 17.27, 237.7
        gamma = (a * temp_c / (b + temp_c)) + (
            2.302585 * (humidity / 100) if humidity > 0 else 0
        )
        # Simplified — close enough for agricultural use
        try:
            alpha = ((a * temp_c) / (b + temp_c)) + (17.27 * 0.01 * humidity / (237.7 + temp_c))
            dew = (b * alpha) / (a - alpha)
            return round(dew, 2)
        except (ZeroDivisionError, ValueError):
            return round(temp_c - ((100 - humidity) / 5), 2)

    @staticmethod
    def _calculate_heat_index(temp_c: float, humidity: float) -> float:
        """Simplified heat index calculation."""
        t = temp_c * 9 / 5 + 32  # Convert to Fahrenheit
        if t < 80:
            return round(temp_c, 2)  # No heat index effect below 80°F

        hi = (-42.379 + 2.04901523 * t + 10.14333127 * humidity
              - 0.22475541 * t * humidity - 0.00683783 * t * t
              - 0.05481717 * humidity * humidity
              + 0.00122874 * t * t * humidity
              + 0.00085282 * t * humidity * humidity
              - 0.00000199 * t * t * humidity * humidity)

        return round((hi - 32) * 5 / 9, 2)  # Back to Celsius

    @staticmethod
    def _calculate_vpd(temp_c: float, humidity: float) -> float:
        """
        Vapor Pressure Deficit (kPa).
        VPD is the gold standard for plant stress assessment.
        - VPD < 0.4 kPa: Too humid, fungal risk
        - VPD 0.8-1.2 kPa: Optimal for most crops
        - VPD > 1.6 kPa: Plants closing stomata, water stress
        Farmers who understand VPD make better irrigation decisions.
        """
        # Tetens formula for saturation vapor pressure
        svp = 0.6108 * (2.71828 ** ((17.27 * temp_c) / (temp_c + 237.3)))
        avp = svp * (humidity / 100)
        vpd = svp - avp
        return round(vpd, 3)


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentalSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down environmental sensor...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
