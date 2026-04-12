"""
export/jetson_deploy.py — Deploy fine-tuned models to Jetson Orin Nano
======================================================================
Handles:
  1. Model format validation (TensorRT .engine, .pt, .safetensors)
  2. Config file generation for inference on Jetson
  3. SSH/SCP transfer to Jetson device
  4. Remote health check (verify model loads + runs dummy inference)
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import yaml


class JetsonDeployer:
    """Push fine-tuned AGRI-FORGE models to Jetson Orin Nano."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.jetson_cfg = cfg["jetson"]
        self.train_cfg = cfg["training"]
        self.export_dir = Path(cfg["paths"]["export_dir"]).expanduser()

        self.host = self.jetson_cfg["host"]
        self.user = self.jetson_cfg["user"]
        self.ssh_key = Path(self.jetson_cfg["ssh_key"]).expanduser()
        self.deploy_path = self.jetson_cfg["deploy_path"]
        self.precision = self.jetson_cfg.get("tensorrt_precision", "fp16")

    def _ssh_cmd(self, cmd: str, timeout: int = 30) -> tuple[int, str]:
        """Execute a command on the Jetson via SSH."""
        ssh = [
            "ssh", "-i", str(self.ssh_key),
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={timeout}",
            f"{self.user}@{self.host}",
            cmd,
        ]
        try:
            result = subprocess.run(
                ssh, capture_output=True, text=True, timeout=timeout + 10,
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "SSH timeout"
        except FileNotFoundError:
            return -1, "ssh not found"

    def _scp_file(self, local: Path, remote: str, timeout: int = 120) -> bool:
        """Copy a file to the Jetson via SCP."""
        scp = [
            "scp", "-i", str(self.ssh_key),
            "-o", "StrictHostKeyChecking=no",
            str(local),
            f"{self.user}@{self.host}:{remote}",
        ]
        try:
            result = subprocess.run(
                scp, capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _generate_inference_config(self, model_name: str) -> dict:
        """Generate Jetson-side inference configuration."""
        return {
            "model": {
                "path": f"{self.deploy_path}/{model_name}",
                "type": "yolov8",
                "precision": self.precision,
                "imgsz": self.train_cfg.get("imgsz", 640),
            },
            "classes": self.cfg["synthetic"]["disease_classes"],
            "inference": {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 100,
                "target_fps": 30,
                "max_latency_ms": self.jetson_cfg.get("max_inference_ms", 30),
            },
            "camera": {
                "source": "/dev/video0",
                "width": 1280,
                "height": 720,
                "fps": 30,
            },
            "output": {
                "save_detections": True,
                "log_path": f"{self.deploy_path}/detections.log",
                "alert_threshold": 0.7,
            },
        }

    def _find_best_model(self) -> Optional[Path]:
        """Locate the best model file for deployment."""
        candidates = [
            self.export_dir / "best.engine",       # TensorRT (preferred)
            self.export_dir / "best.pt",            # PyTorch
            self.export_dir / "best.safetensors",   # Safe format
        ]
        for c in candidates:
            if c.exists():
                return c

        # Search model output directory
        model_dir = Path(self.cfg["paths"]["model_output"]).expanduser()
        for pattern in ["**/best.engine", "**/best.pt"]:
            matches = list(model_dir.glob(pattern))
            if matches:
                return matches[0]

        return None

    def deploy(self, model_path: Optional[str] = None, dry_run: bool = False) -> bool:
        """Deploy model to Jetson Orin Nano."""
        # 1. Find model
        if model_path:
            model = Path(model_path).expanduser()
        else:
            model = self._find_best_model()

        if model is None or not model.exists():
            print(f"[Deploy] ERROR: No model found. Run --mode train first.")
            print(f"[Deploy] Searched: {self.export_dir}")
            return False

        model_size_mb = model.stat().st_size / (1024 ** 2)
        print(f"[Deploy] Model: {model.name} ({model_size_mb:.1f} MB)")
        print(f"[Deploy] Target: {self.user}@{self.host}:{self.deploy_path}")

        if dry_run:
            print("[Deploy] DRY RUN — skipping actual transfer")
            config = self._generate_inference_config(model.name)
            print(f"[Deploy] Inference config:")
            print(json.dumps(config, indent=2))
            return True

        # 2. Test SSH connectivity
        print("[Deploy] Testing SSH connection...")
        rc, out = self._ssh_cmd("echo 'AGRI-FORGE connection OK'")
        if rc != 0:
            print(f"[Deploy] SSH failed: {out}")
            print(f"[Deploy] Ensure Jetson is reachable at {self.host}")
            return False
        print(f"[Deploy] SSH OK")

        # 3. Create deploy directory on Jetson
        self._ssh_cmd(f"mkdir -p {self.deploy_path}")

        # 4. Transfer model
        print(f"[Deploy] Uploading {model.name}...")
        if not self._scp_file(model, f"{self.deploy_path}/{model.name}"):
            print("[Deploy] SCP transfer failed")
            return False
        print("[Deploy] Model transferred")

        # 5. Generate and transfer inference config
        config = self._generate_inference_config(model.name)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f, default_flow_style=False)
            config_tmp = Path(f.name)

        self._scp_file(config_tmp, f"{self.deploy_path}/inference_config.yaml")
        config_tmp.unlink()
        print("[Deploy] Inference config transferred")

        # 6. Transfer class names file
        classes_txt = "\n".join(self.cfg["synthetic"]["disease_classes"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(classes_txt)
            classes_tmp = Path(f.name)

        self._scp_file(classes_tmp, f"{self.deploy_path}/classes.txt")
        classes_tmp.unlink()

        # 7. Remote health check
        print("[Deploy] Running remote health check...")
        check_cmd = (
            f"python3 -c \""
            f"from ultralytics import YOLO; "
            f"m = YOLO('{self.deploy_path}/{model.name}'); "
            f"import numpy as np; "
            f"r = m.predict(np.zeros((640,640,3), dtype=np.uint8), verbose=False); "
            f"print('HEALTH_OK')\""
        )
        rc, out = self._ssh_cmd(check_cmd, timeout=60)
        if "HEALTH_OK" in out:
            print("[Deploy] Health check PASSED")
        else:
            print(f"[Deploy] Health check failed (model may need TensorRT rebuild on Jetson)")
            print(f"[Deploy] Output: {out[:200]}")

        print(f"\n[Deploy] Deployment complete!")
        print(f"[Deploy] Run on Jetson:")
        print(f"  cd {self.deploy_path}")
        print(f"  python3 -c \"from ultralytics import YOLO; "
              f"YOLO('{model.name}').predict(source='/dev/video0', show=True)\"")

        return True
