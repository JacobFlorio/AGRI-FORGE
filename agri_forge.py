#!/usr/bin/env python3
"""
AGRI-FORGE — Agricultural Data Forge & Model Training Pipeline
==============================================================
CLI entry point.  Modes:
    --mode synthetic     Generate procedural aerial crop imagery (PIL + Cosmos)
    --mode isaac         Photorealistic rendering via Isaac Sim (WSL -> Windows)
    --mode scrape        Pull NAIP / USDA / Kaggle ag imagery
    --mode train         LoRA fine-tune YOLOv8n + Cosmos adapter
    --mode swarm         Run MiroFish multi-drone simulation
    --mode export-jetson Push models to Jetson Orin Nano via SSH
    --mode validate      Score detections through JARVIS vision agent
    --mode metrics       Generate SBIR-ready tables
    --mode finetune-vlm  Build VLM fine-tuning dataset (Isaac Sim → Q&A pairs)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml


# ── ROS2 environment isolation ──────────────────────────────────────
# Prevent ROS Humble python path pollution from breaking pytest/lark.
# We only need ROS at runtime for the Isaac bridge, not for general use.
_ROS_VARS_TO_ISOLATE = [
    "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "ROS_PYTHON_VERSION",
    "PYTHONPATH",  # ROS prepends its own site-packages which conflicts
]
_saved_ros = {}
if "ROS_DISTRO" in os.environ:
    for var in _ROS_VARS_TO_ISOLATE:
        val = os.environ.pop(var, None)
        if val is not None:
            _saved_ros[var] = val
    # Keep ROS_DISTRO itself so isaac_bridge can detect + re-source if needed
    _saved_ros["ROS_DISTRO"] = os.environ.get("ROS_DISTRO", "")


def _load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration and normalize to canonical schema."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path.resolve()}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return normalize_config(cfg)


def normalize_config(cfg: dict) -> dict:
    """Map user-facing config keys to the canonical keys all modules expect.

    This handles the schema drift between the user's config.yaml (which uses
    simplified keys like 'anomaly_classes', 'isaacsim', flat training params)
    and what the pipeline modules read (disease_classes, isaac_sim, nested lora).
    """
    synth = cfg.setdefault("synthetic", {})

    # disease_classes ← anomaly_classes (canonical name for all modules)
    if "disease_classes" not in synth and "anomaly_classes" in synth:
        synth["disease_classes"] = synth["anomaly_classes"]
    elif "disease_classes" not in synth:
        synth["disease_classes"] = [
            "gray_leaf_spot", "northern_corn_leaf_blight", "common_rust",
            "sudden_death_syndrome", "frogeye_leaf_spot", "nitrogen_deficiency",
            "water_stress", "weed_pressure", "stand_gap", "ponding", "healthy",
        ]

    # num_images ← total_target_images
    if "num_images" not in synth:
        synth["num_images"] = synth.get("total_target_images", 2000)

    # resolution (default 640x640 for YOLO training)
    if "resolution" not in synth:
        res = cfg.get("isaacsim", {}).get("render_resolution", [640, 640])
        synth["resolution"] = res

    # inference_resolution — YOLO training input size (square, downscaled from render res)
    synth.setdefault("inference_resolution", 640)

    # cosmos_model (default)
    synth.setdefault("cosmos_model", "nvidia/Cosmos-Reason2-2B")

    # augmentation sub-keys with procedural defaults
    aug = synth.setdefault("augmentation", {})
    aug.setdefault("flip", aug.get("enabled", True))
    aug.setdefault("rotate_max", 15)
    aug.setdefault("brightness_range", [0.8, 1.2])
    aug.setdefault("noise_std", 0.02)

    # ── isaac_sim ← isaacsim ────────────────────────────────────────
    if "isaac_sim" not in cfg and "isaacsim" in cfg:
        isaacsim = cfg["isaacsim"]
        cfg["isaac_sim"] = {
            "install_path": isaacsim.get("path", r"C:\isaacsim"),
            "python_exe": isaacsim.get("python_exe", r"C:\isaacsim\python.bat"),
            "shared_dir": r"C:\agri_forge_isaac_data",
            "headless": isaacsim.get("headless", True),
            "ros2_bridge": isaacsim.get("ros2_bridge", True),
            "altitude_range_m": isaacsim.get("camera_altitude_m", [30, 80]),
            "field_size_m": 200,
            "crop_types": synth.get("crop_types", ["corn", "soybean"]),
            "camera_hfov_deg": 70,
            "num_scenes_per_batch": isaacsim.get("max_batch_size", 50),
        }
    cfg.setdefault("isaac_sim", {})

    # ── swarm defaults ──────────────────────────────────────────────
    swarm = cfg.setdefault("swarm", {})
    # field_dimensions_m ← field_size_m (list)
    if "field_dimensions_m" not in swarm and "field_size_m" in swarm:
        fs = swarm["field_size_m"]
        swarm["field_dimensions_m"] = fs if isinstance(fs, list) else [fs, fs]
    swarm.setdefault("field_dimensions_m", [640, 640])
    swarm.setdefault("num_agents", 50)
    swarm.setdefault("speed_mps", 8)
    swarm.setdefault("battery_minutes", 25)
    swarm.setdefault("communication_range_m", 500)
    swarm.setdefault("overlap_pct", 20)

    # ── training defaults ───────────────────────────────────────────
    train = cfg.setdefault("training", {})
    train.setdefault("imgsz", 640)
    train.setdefault("optimizer", "AdamW")
    if "lr0" not in train and "learning_rate" in train:
        train["lr0"] = train["learning_rate"]
    train.setdefault("lr0", 0.001)
    train.setdefault("lrf", 0.01)
    train.setdefault("warmup_epochs", 3)
    train.setdefault("patience", 20)
    # Nest lora config if only flat keys exist
    if "lora" not in train and "lora_rank" in train:
        train["lora"] = {
            "rank": train.get("lora_rank", 16),
            "alpha": train.get("lora_alpha", 32),
            "dropout": 0.05,
            "target_modules": train.get("target_modules", []),
        }
    train.setdefault("cosmos_adapter", {"enabled": False})
    if train.get("use_tensorrt"):
        train.setdefault("export_formats", ["torchscript", "engine"])
    else:
        train.setdefault("export_formats", ["torchscript"])

    # ── jetson defaults ─────────────────────────────────────────────
    jetson = cfg.setdefault("jetson", {})
    if "ssh_key" not in jetson and "key_path" in jetson:
        jetson["ssh_key"] = jetson["key_path"]
    jetson.setdefault("ssh_key", "~/.ssh/id_rsa")
    if "deploy_path" not in jetson and "models_dir" in jetson:
        jetson["deploy_path"] = jetson["models_dir"]
    jetson.setdefault("deploy_path", "/home/jetson/models")
    jetson.setdefault("tensorrt_precision", "fp16")
    jetson.setdefault("max_inference_ms", 30)

    # ── jarvis defaults (section removed in new config) ─────────────
    cfg.setdefault("jarvis", {
        "enabled": False,
        "memory_endpoint": "http://localhost:7860",
        "inject_memory": False,
        "validation_threshold": cfg.get("validation", {}).get("mAP_threshold", 0.5),
    })

    # ── metrics defaults ────────────────────────────────────────────
    met = cfg.setdefault("metrics", {})
    met.setdefault("output_csv", "./reports/sbir_metrics.csv")
    met.setdefault("output_latex", "./reports/sbir_table.tex")
    met.setdefault("benchmark_runs", 100)
    if "chemical_cost_per_acre" not in met:
        met["chemical_cost_per_acre"] = met.get("chemical_savings_baseline_pct", 45.0)
    met.setdefault("target_reduction_pct", 30)

    # ── scraper defaults ────────────────────────────────────────────
    scraper = cfg.setdefault("scraper", {})
    scraper.setdefault("naip_years", [2021, 2022, 2023])
    scraper.setdefault("max_images_per_source", scraper.get("max_images", 5000))

    # ── vlm_finetune defaults ────────────────────────────────────────
    vlm = cfg.setdefault("vlm_finetune", {})
    vlm.setdefault("num_images", 100)
    vlm.setdefault("qa_pairs_per_image", 3)
    vlm.setdefault("include_general_description", True)
    vlm.setdefault("output_format", "sharegpt")
    vlm.setdefault("base_model", "nvidia/Cosmos-Reason2-2B")
    # Merge vlm_finetune settings from synthetic section (convenience alias)
    synth_vlm = synth.get("vlm_finetune", {})
    vlm.setdefault("vlm_resolution", synth_vlm.get("vlm_resolution", 256))
    vlm.setdefault("vlm_jpeg_quality", synth_vlm.get("vlm_jpeg_quality", 70))

    # ── hardware defaults ───────────────────────────────────────────
    cfg.setdefault("hardware", {"gpu_device": 0, "max_vram_gb": 14.0, "cuda_version": "12.6"})

    # ── paths defaults ──────────────────────────────────────────────
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("data_root", "./datasets")
    cfg["paths"].setdefault("model_output", "./models")
    cfg["paths"].setdefault("export_dir", "./export/builds")
    cfg["paths"].setdefault("jarvis_root", "~/JARVIS")

    return cfg


def cmd_scrape(cfg: dict, args: argparse.Namespace) -> None:
    from data.scraper import AgriScraper
    scraper = AgriScraper(cfg)
    scraper.run(max_images=args.max_images)


def _run_synthetic_yolo(cfg: dict, args: argparse.Namespace) -> None:
    """Generate synthetic data in YOLO label format."""
    num_images = args.num_images or cfg["synthetic"]["num_images"]

    # Attempt Isaac Sim first, fall back to procedural PIL+Cosmos
    try:
        from simulation.isaac_scene_generator import IsaacSceneGenerator
        isaac_gen = IsaacSceneGenerator(cfg)
        isaac_available = isaac_gen._check_isaac_available()
    except Exception as e:
        print(f"[Synthetic] Isaac Sim check skipped (import error): {e}")
        isaac_available = False

    if isaac_available:
        print("[Synthetic] Isaac Sim available — using photorealistic rendering")
        isaac_gen.generate(num_images=num_images, headless=args.headless)
    else:
        print("[Synthetic] Isaac Sim unavailable — falling back to procedural "
              "generation (PIL + Cosmos)")
        from data.synthetic_generator import SyntheticGenerator
        gen = SyntheticGenerator(cfg)
        gen.generate(num_images=num_images, resume=args.resume)


def _run_synthetic_vlm(cfg: dict, args: argparse.Namespace) -> None:
    """Generate synthetic data in VLM fine-tuning format."""
    num_images = args.num_images or cfg.get("vlm_finetune", {}).get("num_images", 100)
    from training.vlm_dataset_builder import VLMDatasetBuilder
    builder = VLMDatasetBuilder(cfg)
    builder.build(num_images=num_images, headless=args.headless)


def cmd_synthetic(cfg: dict, args: argparse.Namespace) -> None:
    fmt = getattr(args, "format", "yolo")

    if fmt == "yolo":
        _run_synthetic_yolo(cfg, args)
    elif fmt == "vlm-finetune":
        _run_synthetic_vlm(cfg, args)
    elif fmt == "both":
        _run_synthetic_yolo(cfg, args)
        _run_synthetic_vlm(cfg, args)
    else:
        print(f"[Synthetic] Unknown format: {fmt}")
        return


def cmd_isaac(cfg: dict, args: argparse.Namespace) -> None:
    from simulation.isaac_scene_generator import IsaacSceneGenerator
    gen = IsaacSceneGenerator(cfg)
    gen.generate(
        num_images=args.num_images or cfg["isaac_sim"].get("num_scenes_per_batch", 10),
        headless=args.headless,
    )


def cmd_train(cfg: dict, args: argparse.Namespace) -> None:
    from training.unsloth_trainer import AgriTrainer
    trainer = AgriTrainer(cfg)
    trainer.train(resume=args.resume)


def cmd_swarm(cfg: dict, args: argparse.Namespace) -> None:
    from simulation.swarm_engine import SwarmEngine
    engine = SwarmEngine(cfg)
    engine.run(
        num_agents=args.num_agents or cfg["swarm"].get("num_agents", 50),
        duration_min=args.duration,
    )


def cmd_export_jetson(cfg: dict, args: argparse.Namespace) -> None:
    from export.jetson_deploy import JetsonDeployer
    deployer = JetsonDeployer(cfg)
    deployer.deploy(model_path=args.model_path, dry_run=args.dry_run)


def cmd_validate(cfg: dict, args: argparse.Namespace) -> None:
    from validation.jarvis_validator import JarvisValidator
    validator = JarvisValidator(cfg)
    validator.validate(predictions_dir=args.predictions_dir)


def cmd_metrics(cfg: dict, args: argparse.Namespace) -> None:
    from utils.metrics import MetricsGenerator
    gen = MetricsGenerator(cfg)
    gen.generate_sbir_report()


def cmd_finetune_vlm(cfg: dict, args: argparse.Namespace) -> None:
    from training.vlm_dataset_builder import VLMDatasetBuilder
    builder = VLMDatasetBuilder(cfg)
    builder.build(
        num_images=args.num_images or cfg.get("vlm_finetune", {}).get("num_images", 100),
        headless=args.headless,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agri_forge",
        description="AGRI-FORGE: Agricultural Data Forge & Model Training Pipeline",
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "scrape", "synthetic", "isaac", "train", "swarm",
            "export-jetson", "validate", "metrics", "finetune-vlm",
        ],
        help="Pipeline mode to execute",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--inject-memory", action="store_true",
                    help="Inject results into JARVIS memory (requires jarvis.enabled=true)")

    # Isaac / Synthetic shared
    p.add_argument("--headless", action="store_true",
                    help="Run Isaac Sim in headless mode (no GUI, for batch rendering)")
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--format", default="yolo",
                    choices=["yolo", "vlm-finetune", "both"],
                    help="Output format: yolo labels, vlm-finetune JSONL, or both")

    # Scraper
    p.add_argument("--max-images", type=int, default=None)

    # Training
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    # Swarm
    p.add_argument("--num-agents", type=int, default=None)
    p.add_argument("--duration", type=int, default=25, help="Sim duration in minutes")

    # Export
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")

    # Validate
    p.add_argument("--predictions-dir", type=str, default=None)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _load_config(args.config)

    # Override JARVIS injection if flag set
    if args.inject_memory:
        cfg.setdefault("jarvis", {})["inject_memory"] = True

    dispatch = {
        "scrape": cmd_scrape,
        "synthetic": cmd_synthetic,
        "isaac": cmd_isaac,
        "train": cmd_train,
        "swarm": cmd_swarm,
        "export-jetson": cmd_export_jetson,
        "validate": cmd_validate,
        "metrics": cmd_metrics,
        "finetune-vlm": cmd_finetune_vlm,
    }

    print(f"{'='*60}")
    print(f" AGRI-FORGE  |  mode: {args.mode}")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    dispatch[args.mode](cfg, args)

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] {args.mode} completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
