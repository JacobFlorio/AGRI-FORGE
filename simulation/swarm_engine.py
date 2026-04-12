"""
simulation/swarm_engine.py — MiroFish-inspired multi-drone swarm simulation
============================================================================
Simulates 50–200 autonomous drone agents coordinating crop scouting over
a 100-acre field.  Each agent has:
  - Persona (scout / inspector / relay)
  - Local memory of observed anomalies
  - Boid-like flocking + pheromone-inspired priority zones
  - Battery model with RTB (return-to-base) logic

Outputs:
  - Mission plan JSON (waypoints, assignments, timing)
  - Coverage heatmap (% of field scanned)
  - Detection log (simulated anomaly finds)
  - SBIR metrics (coverage rate, overlap, chemical savings)
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Agent personas ──────────────────────────────────────────────────
PERSONA_SPECS = {
    "scout": {
        "speed_factor": 1.0,
        "altitude_m": 30,
        "scan_width_m": 40,
        "priority": "coverage",
        "ratio": 0.6,          # 60% of fleet
    },
    "inspector": {
        "speed_factor": 0.5,   # slower, more careful
        "altitude_m": 15,
        "scan_width_m": 15,
        "priority": "detail",
        "ratio": 0.25,
    },
    "relay": {
        "speed_factor": 0.8,
        "altitude_m": 50,      # higher for comms
        "scan_width_m": 0,     # no scanning
        "priority": "mesh",
        "ratio": 0.15,
    },
}


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> "Vec2":
        return Vec2(self.x * s, self.y * s)

    def mag(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalized(self) -> "Vec2":
        m = self.mag()
        if m < 1e-6:
            return Vec2(0, 0)
        return Vec2(self.x / m, self.y / m)

    def dist(self, other: "Vec2") -> float:
        return (self - other).mag()


@dataclass
class DroneAgent:
    """Single drone agent in the swarm."""
    id: int
    persona: str
    pos: Vec2 = field(default_factory=Vec2)
    vel: Vec2 = field(default_factory=Vec2)
    heading: float = 0.0          # degrees
    battery_pct: float = 100.0
    state: str = "active"         # active, rtb, landed, inspecting
    memory: list = field(default_factory=list)   # observed anomalies
    waypoints: list = field(default_factory=list)
    scan_log: list = field(default_factory=list)
    total_distance_m: float = 0.0


class SwarmEngine:
    """MiroFish-inspired swarm simulation engine."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["swarm"]
        self.field_w, self.field_h = self.cfg.get("field_dimensions_m", [640, 640])
        self.base_speed = self.cfg.get("speed_mps", 8)
        self.battery_min = self.cfg.get("battery_minutes", 25)
        self.comm_range = self.cfg.get("communication_range_m", 500)
        self.overlap_pct = self.cfg.get("overlap_pct", 20)
        self.dt = 1.0  # simulation timestep in seconds

        self.output_dir = Path("./reports/swarm")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Anomaly map: grid of disease probabilities
        self.anomaly_grid = None
        self.coverage_grid = None
        self.pheromone_grid = None

    def _init_agents(self, num_agents: int) -> list[DroneAgent]:
        """Spawn agents with persona distribution at base station."""
        agents = []
        base = Vec2(self.field_w / 2, 0)  # south-center base

        for i in range(num_agents):
            # Assign persona based on ratios
            r = random.random()
            cumulative = 0.0
            persona = "scout"
            for p, spec in PERSONA_SPECS.items():
                cumulative += spec["ratio"]
                if r <= cumulative:
                    persona = p
                    break

            agent = DroneAgent(
                id=i,
                persona=persona,
                pos=Vec2(base.x + random.uniform(-5, 5),
                         base.y + random.uniform(-5, 5)),
                vel=Vec2(0, 0),
                heading=random.uniform(0, 360),
            )
            agents.append(agent)

        return agents

    def _init_anomaly_map(self) -> None:
        """Generate a realistic anomaly distribution over the field."""
        # Grid resolution: 1 cell = 5m x 5m
        gw = self.field_w // 5
        gh = self.field_h // 5
        self.anomaly_grid = np.zeros((gh, gw), dtype=np.float32)
        self.coverage_grid = np.zeros((gh, gw), dtype=np.float32)
        self.pheromone_grid = np.zeros((gh, gw), dtype=np.float32)

        # Place 3-8 disease hotspots
        num_hotspots = random.randint(3, 8)
        for _ in range(num_hotspots):
            cx = random.randint(0, gw - 1)
            cy = random.randint(0, gh - 1)
            radius = random.randint(5, 20)
            intensity = random.uniform(0.5, 1.0)

            for y in range(max(0, cy - radius), min(gh, cy + radius)):
                for x in range(max(0, cx - radius), min(gw, cx + radius)):
                    d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if d < radius:
                        val = intensity * (1 - d / radius)
                        self.anomaly_grid[y, x] = max(self.anomaly_grid[y, x], val)

    def _boid_forces(self, agent: DroneAgent, agents: list[DroneAgent]) -> Vec2:
        """Compute MiroFish boid-like steering forces."""
        separation = Vec2(0, 0)
        alignment = Vec2(0, 0)
        cohesion = Vec2(0, 0)
        pheromone_pull = Vec2(0, 0)
        n_neighbors = 0

        for other in agents:
            if other.id == agent.id or other.state != "active":
                continue
            d = agent.pos.dist(other.pos)
            if d > self.comm_range:
                continue

            n_neighbors += 1

            # Separation: avoid collisions (strong at close range)
            if d < 20:
                diff = agent.pos - other.pos
                separation = separation + diff * (1.0 / max(d, 0.1))

            # Alignment: match heading of neighbors
            alignment = alignment + other.vel

            # Cohesion: move toward center of neighbors
            cohesion = cohesion + other.pos

        if n_neighbors > 0:
            alignment = alignment * (1.0 / n_neighbors)
            cohesion = (cohesion * (1.0 / n_neighbors)) - agent.pos

        # Pheromone attraction: pull toward unexplored high-anomaly areas
        if self.pheromone_grid is not None:
            gx = int(agent.pos.x / 5) % self.pheromone_grid.shape[1]
            gy = int(agent.pos.y / 5) % self.pheromone_grid.shape[0]
            # Look at surrounding cells for strongest signal
            best_val = -1
            best_dir = Vec2(0, 0)
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.pheromone_grid.shape[1] and \
                       0 <= ny < self.pheromone_grid.shape[0]:
                        val = self.pheromone_grid[ny, nx]
                        if val > best_val:
                            best_val = val
                            best_dir = Vec2(dx * 5.0, dy * 5.0)
            pheromone_pull = best_dir.normalized() * best_val

        # Weight forces by persona
        spec = PERSONA_SPECS[agent.persona]
        if spec["priority"] == "coverage":
            # Scouts spread out more
            return separation * 3.0 + alignment * 0.5 + cohesion * 0.3 + pheromone_pull * 2.0
        elif spec["priority"] == "detail":
            # Inspectors are drawn to pheromones
            return separation * 2.0 + alignment * 0.3 + cohesion * 0.2 + pheromone_pull * 5.0
        else:
            # Relays maintain mesh coverage
            return separation * 1.5 + alignment * 1.0 + cohesion * 2.0

    def _update_agent(self, agent: DroneAgent, agents: list[DroneAgent]) -> None:
        """Step one agent forward by dt."""
        if agent.state == "landed":
            return

        spec = PERSONA_SPECS[agent.persona]
        max_speed = self.base_speed * spec["speed_factor"]

        # Battery drain: ~0.067%/s at 25 min flight
        drain_rate = 100.0 / (self.battery_min * 60)
        agent.battery_pct -= drain_rate * self.dt

        # RTB if battery low
        if agent.battery_pct <= 15:
            agent.state = "rtb"

        if agent.state == "rtb":
            base = Vec2(self.field_w / 2, 0)
            to_base = (base - agent.pos).normalized()
            agent.vel = to_base * max_speed
            agent.pos = agent.pos + agent.vel * self.dt
            agent.total_distance_m += max_speed * self.dt
            if agent.pos.dist(base) < 5:
                agent.state = "landed"
            return

        # Boid steering
        force = self._boid_forces(agent, agents)
        # Add random exploration noise
        noise = Vec2(random.gauss(0, 1), random.gauss(0, 1))
        force = force + noise * 0.5

        # Update velocity with damping
        agent.vel = (agent.vel + force * self.dt) * 0.95
        speed = agent.vel.mag()
        if speed > max_speed:
            agent.vel = agent.vel.normalized() * max_speed

        # Update position with boundary clamping
        agent.pos = agent.pos + agent.vel * self.dt
        agent.pos.x = max(0, min(self.field_w, agent.pos.x))
        agent.pos.y = max(0, min(self.field_h, agent.pos.y))
        agent.total_distance_m += agent.vel.mag() * self.dt

        # Update heading
        if agent.vel.mag() > 0.1:
            agent.heading = math.degrees(math.atan2(agent.vel.y, agent.vel.x))

        # Scan: mark coverage and detect anomalies
        if spec["scan_width_m"] > 0:
            scan_r = spec["scan_width_m"] / 2
            gx = int(agent.pos.x / 5)
            gy = int(agent.pos.y / 5)
            scan_cells = int(scan_r / 5)

            for dy in range(-scan_cells, scan_cells + 1):
                for dx in range(-scan_cells, scan_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.coverage_grid.shape[1] and \
                       0 <= ny < self.coverage_grid.shape[0]:
                        self.coverage_grid[ny, nx] = 1.0

                        # Check for anomaly
                        anom = self.anomaly_grid[ny, nx]
                        if anom > 0.3 and random.random() < anom:
                            detection = {
                                "agent_id": agent.id,
                                "pos": (nx * 5, ny * 5),
                                "severity": float(anom),
                                "time_s": 0,  # filled by caller
                            }
                            if detection not in agent.memory:
                                agent.memory.append(detection)
                                # Deposit pheromone to attract inspectors
                                self.pheromone_grid[ny, nx] += anom * 2

        # Pheromone decay
        # (done globally per step, but we mark visited cells)

    def run(self, num_agents: int = 50, duration_min: int = 25) -> dict:
        """Execute the full swarm simulation."""
        num_agents = max(2, min(num_agents, self.cfg.get("max_agents", 200)))
        duration_s = duration_min * 60
        steps = int(duration_s / self.dt)

        print(f"[Swarm] Initializing {num_agents} agents over "
              f"{self.field_w}x{self.field_h}m field")
        print(f"[Swarm] Duration: {duration_min}min ({steps} steps)")

        agents = self._init_agents(num_agents)
        self._init_anomaly_map()

        # Persona breakdown
        persona_counts = {}
        for a in agents:
            persona_counts[a.persona] = persona_counts.get(a.persona, 0) + 1
        print(f"[Swarm] Fleet: {persona_counts}")

        t0 = time.perf_counter()
        all_detections = []

        for step in range(steps):
            # Update all agents
            for agent in agents:
                self._update_agent(agent, agents)
                # Tag detections with time
                for det in agent.memory:
                    if det.get("time_s") == 0:
                        det["time_s"] = step

            # Global pheromone decay
            if self.pheromone_grid is not None:
                self.pheromone_grid *= 0.999

            # Progress
            if (step + 1) % (steps // 10) == 0:
                active = sum(1 for a in agents if a.state == "active")
                coverage = np.mean(self.coverage_grid) * 100
                print(f"  [{step+1}/{steps}] active={active}, "
                      f"coverage={coverage:.1f}%")

        elapsed = time.perf_counter() - t0
        print(f"[Swarm] Simulation complete in {elapsed:.1f}s")

        # ── Compute metrics ─────────────────────────────────────────
        total_coverage = float(np.mean(self.coverage_grid) * 100)
        total_detections = sum(len(a.memory) for a in agents)
        total_distance = sum(a.total_distance_m for a in agents)

        # Chemical savings estimate (targeted spray vs blanket)
        anomaly_area = float(np.sum(self.anomaly_grid > 0.3)) * 25  # m²
        total_area = self.field_w * self.field_h
        targeted_ratio = anomaly_area / total_area if total_area > 0 else 1.0
        chemical_savings_pct = (1 - targeted_ratio) * 100

        metrics = {
            "num_agents": num_agents,
            "duration_min": duration_min,
            "field_area_m2": self.field_w * self.field_h,
            "field_acres": round(self.field_w * self.field_h / 4046.86, 2),
            "coverage_pct": round(total_coverage, 2),
            "total_detections": total_detections,
            "total_distance_km": round(total_distance / 1000, 2),
            "avg_distance_per_agent_km": round(total_distance / num_agents / 1000, 2),
            "chemical_savings_pct": round(chemical_savings_pct, 1),
            "anomaly_hotspots": int(np.sum(self.anomaly_grid > 0.5)),
            "persona_breakdown": persona_counts,
            "simulation_time_s": round(elapsed, 1),
        }

        # ── Generate mission plan ──────────────────────────────────
        mission_plan = {
            "metadata": metrics,
            "agents": [],
        }

        for agent in agents:
            mission_plan["agents"].append({
                "id": agent.id,
                "persona": agent.persona,
                "final_state": agent.state,
                "battery_remaining_pct": round(agent.battery_pct, 1),
                "distance_km": round(agent.total_distance_m / 1000, 2),
                "detections": len(agent.memory),
                "final_pos": {"x": round(agent.pos.x, 1), "y": round(agent.pos.y, 1)},
            })

        # ── Save outputs ────────────────────────────────────────────
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(self.output_dir / "mission_plan.json", "w") as f:
            json.dump(mission_plan, f, indent=2, cls=_NumpyEncoder)

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=_NumpyEncoder)

        # Save coverage heatmap as CSV
        np.savetxt(
            self.output_dir / "coverage_heatmap.csv",
            self.coverage_grid, delimiter=",", fmt="%.2f",
        )

        # Print summary
        print(f"\n{'='*50}")
        print(f" SWARM SIMULATION RESULTS")
        print(f"{'='*50}")
        print(f" Agents:           {num_agents}")
        print(f" Field:            {metrics['field_acres']:.1f} acres")
        print(f" Coverage:         {metrics['coverage_pct']:.1f}%")
        print(f" Detections:       {metrics['total_detections']}")
        print(f" Total distance:   {metrics['total_distance_km']:.1f} km")
        print(f" Chemical savings: {metrics['chemical_savings_pct']:.1f}%")
        print(f" Output:           {self.output_dir.resolve()}")
        print(f"{'='*50}")

        return metrics
