# AGRI-FORGE

> **The open training-data and model-forging infrastructure behind the Firmament drone platform's agricultural branch.** Synthetic data generation, Isaac Sim scene authoring, VLM dataset building, fine-tuning pipelines, plus the public ag-specific ROS nodes and the farmer dashboard.

AGRI-FORGE is a companion repo to [Firmament](#where-this-fits), the private edge-AI drone platform that actually flies. What lives here is the part of that work I can publish safely: the **data factory and model-forging pipeline** that trains the ag-specific perception stack, plus a **public mirror of the agricultural branch code** (perception nodes, dashboard, mission planner) that parallels what runs on the drone. The flight stack, security pipeline, and deployment glue stay private in Firmament for obvious reasons — publishing a working autonomous-drone perception system is a bad idea for me and for anyone who'd try to fly it without understanding what they're doing.

What's here is valuable on its own, though. The data-generation and VLM dataset pipelines are the interesting engineering — the drone is just the thing that runs the resulting model.

---

## Where this fits

```
Firmament (private) ─── production flight + security + ag pipeline
                      │
                      └──→ eats models/datasets produced by AGRI-FORGE

AGRI-FORGE (public) ─── training data & model factory
  ├─ data/                 ── synthetic + scraped training data
  ├─ simulation/           ── Isaac Sim scene generation
  ├─ training/             ── VLM dataset building + unsloth fine-tuning
  ├─ firmament-ag/         ── public ag-branch ROS nodes (mirror of the
  │                           agricultural perception running in Firmament)
  └─ scout_dashboard/      ── farmer-facing PWA dashboard
```

---

## What's actually in here

### 🏭 The data + training factory (the center of mass)

This is the bulk of AGRI-FORGE and the reason it exists. Everything the drone's ag branch does — detecting crop disease, classifying pest damage, assessing nutrient stress — starts with data that comes out of this pipeline.

| File | Size | What it does |
|---|---|---|
| [`data/synthetic_generator.py`](./data/synthetic_generator.py) | 30 KB | Synthetic crop imagery generator across the 11 ag classes in `ag_class_taxonomy.yaml`. Varies lighting, altitude, angle, growth stage. |
| [`data/scraper.py`](./data/scraper.py) | 10 KB | Real-world data scraping for augmentation. |
| [`simulation/isaac_scene_generator.py`](./simulation/isaac_scene_generator.py) | 38 KB | Isaac Sim scene authoring — builds photorealistic ag environments programmatically for synthetic dataset generation. |
| [`simulation/isaac_bridge.py`](./simulation/isaac_bridge.py) | 18 KB | Bridge between Isaac Sim and the training data pipeline. |
| [`simulation/swarm_engine.py`](./simulation/swarm_engine.py) | 17 KB | Multi-drone swarm simulation for scenario-diverse data. |
| [`training/vlm_dataset_builder.py`](./training/vlm_dataset_builder.py) | 62 KB | Builds prompt/image/response triples for fine-tuning vision-language models on agronomic assessment. This is the biggest file in the repo. |
| [`training/unsloth_trainer.py`](./training/unsloth_trainer.py) | 13 KB | LLM fine-tuning via [unsloth](https://github.com/unslothai/unsloth) — memory-efficient training pipeline. |

**The interesting part:** this isn't just "YOLO fine-tuning on Roboflow data." The VLM dataset builder is generating **chain-of-thought agronomic assessments** — not classification labels but structured reasoning outputs ("leaf yellowing pattern consistent with nitrogen deficiency in V4 corn, recommend soil test before foliar application"). That's what makes the drone output an actionable scouting report instead of a list of bounding boxes.

The Isaac Sim integration matters for the same reason: real agricultural training data is scarce, expensive, and farmer-specific. Synthetic scenes let you generate thousands of examples with controlled variation — sun angle, growth stage, disease severity — that no real dataset would contain.

### 🌾 `firmament-ag/` — public mirror of the ag perception branch

The ag-specific ROS 2 nodes that parallel what runs on the drone in Firmament-Private. This is a **clean-room public version** — the architecture, interfaces, and algorithms are the same, but this copy is free of security-pipeline glue, private credentials, and any code paths that would let someone spin up a working autonomous drone without understanding what they're doing.

| File | Size | What it does |
|---|---|---|
| [`firmament-ag/nodes/ag_perception.py`](./firmament-ag/nodes/ag_perception.py) | 29 KB | The ag perception node: classifies YOLO detections as crop health issues, correlates with environmental data, triggers VLM assessments on severe cases, computes per-zone health scores. |
| [`firmament-ag/nodes/environmental_sensor.py`](./firmament-ag/nodes/environmental_sensor.py) | 9 KB | BME280 integration — humidity, temperature, pressure → derived VPD, dew point, fungal risk. Publishes a spatial microclimate stream. |
| [`firmament-ag/nodes/zone_mission_planner.py`](./firmament-ag/nodes/zone_mission_planner.py) | 16 KB | Reads farm config, generates lawnmower waypoint patterns per zone, estimates flight time vs. battery. |
| [`firmament-ag/scripts/photogrammetry_pipeline.sh`](./firmament-ag/scripts/photogrammetry_pipeline.sh) | 8 KB | OpenDroneMap driver for orthomosaic / DSM / 3D mesh generation from flight imagery. |
| [`firmament-ag/scripts/merge_training_data.py`](./firmament-ag/scripts/merge_training_data.py) | 11 KB | Merges synthetic + farmer-verified real data for retraining. |
| [`firmament-ag/config/farm_config.yaml`](./firmament-ag/config/farm_config.yaml) | 9 KB | Example farm: field boundaries, zones, crop types, thresholds. |

**Why microclimate sensing matters:** weather stations measure humidity at 6 feet above the ground. The drone measures it *at the crop canopy*, geo-tagged, across the field. Canopy humidity predicts where fungal disease will appear *before* it's visible. VPD (vapor pressure deficit) is the metric advanced growers use for irrigation decisions — spatial VPD maps at this price point don't exist elsewhere.

### 📱 `scout_dashboard/` — the farmer-facing PWA

A self-contained farmer dashboard. Runs on the Jetson, serves a web UI on port 8090 via FastAPI, and the farmer connects from any phone on the drone's WiFi hotspot. No cloud, no account required.

| File | Size | What it does |
|---|---|---|
| [`scout_dashboard/static/index.html`](./scout_dashboard/static/index.html) | 22 KB | Leaflet-map dashboard with detection pins, severity filters, AI reasoning display, and farmer verification UI. |
| [`scout_dashboard/dashboard_server.py`](./scout_dashboard/dashboard_server.py) | 14 KB | FastAPI server — reads from the mission SQLite, serves the HTML + REST API. |
| [`scout_dashboard/mission_logger.py`](./scout_dashboard/mission_logger.py) | 13 KB | Passive ROS 2 subscriber that writes detections to SQLite. **Pure subscriber, never publishes** — zero interference with an existing flight stack. |
| [`scout_dashboard/finetune_agricultural.py`](./scout_dashboard/finetune_agricultural.py) | 9 KB | Takes farmer-verified annotations exported from the dashboard and fine-tunes a YOLOv8-nano on the merged real+synthetic dataset. |

The "farmer-verify training flywheel" closes the loop: drone detects → dashboard presents → farmer confirms/rejects/relabels → annotations exported → model fine-tunes → next flight is sharper, specifically on *that farm's* conditions.

### 🛠 Top-level orchestration + ops

| File | Size | Purpose |
|---|---|---|
| [`agri_forge.py`](./agri_forge.py) | 16 KB | Top-level pipeline orchestration entry point. |
| [`dashboard_relay.py`](./dashboard_relay.py) | 25 KB | Relay between the perception pipeline and the dashboard backend. |
| [`export/jetson_deploy.py`](./export/jetson_deploy.py) | 8 KB | Model + config deployment to the Jetson target. |
| [`ag_class_taxonomy.yaml`](./ag_class_taxonomy.yaml) | 11 KB | The 11 ag detection classes and their hierarchy. |
| [`cosmos_calibration_training.yaml`](./cosmos_calibration_training.yaml) | 21 KB | VLM prompt calibration profiles for agronomic assessment — the hand-tuned prompts the VLM uses to produce structured reports. |
| [`tests/test_agri_forge.py`](./tests/test_agri_forge.py) | 24 KB | Test suite. |
| [`setup_agri_forge.sh`](./setup_agri_forge.sh) | 7 KB | Environment setup script. |

---

## What's NOT in here

Deliberately absent, living privately in Firmament:

- The production flight stack (PX4 offboard control, safety gates, RTH logic)
- The security-branch perception pipeline (people / vehicles / threats)
- Evidence recording and BOLO matching
- Any hardware credentials, RTSP keys, or deployment targets
- The full integrated runtime that combines all of the above into a live autonomous system

If you're trying to fly an autonomous drone, this repo alone won't get you there — and that's intentional. What this repo *will* teach you is how to build the training pipeline and perception architecture that an agricultural drone needs.

---

## Why any of this is interesting

Most open-source ag-drone work falls into two buckets: **(1)** thin wrappers around commercial APIs (DJI Terra, Sentera FieldAgent) that require an internet connection and a $12 K subscription, or **(2)** academic code that publishes one model checkpoint with no training pipeline, no dataset tooling, and no path to reproducing the result on a new farm.

AGRI-FORGE is neither. It's the forge — the machinery for generating training data, building VLM datasets with real agronomic reasoning, fine-tuning edge-deployable models, and shipping them to a Jetson. That's the hard part that usually stays hidden inside commercial platforms. The drone-side code is here too as a reference implementation so you can see what the trained model actually does once it flies, but the valuable engineering is in the pipeline that produces it.

---

## Status

Active R&D, self-funded independent project. The data factory and training pipeline are the most mature pieces. The `firmament-ag/` perception nodes are a public mirror of what runs in Firmament-Private — the architecture is stable; the public copy has received less shake-out than the private production path. Expect rough edges. This is a research repo, not a product.

If you want to play with the data pipeline or study the perception architecture, clone away. If you want to fly an autonomous drone, talk to me first — there's a reason the flight stack is private.

## License

MIT.
