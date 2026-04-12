"""
utils/metrics.py — Auto-generate SBIR-ready metrics tables
==========================================================
Outputs:
  - CSV with mAP, inference latency, coverage, chemical savings
  - LaTeX table for direct inclusion in NSF SBIR proposal
  - Console-friendly summary
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class MetricsGenerator:
    """Generate comprehensive SBIR metrics from AGRI-FORGE outputs."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.metrics_cfg = cfg.get("metrics", {})
        self.output_csv = Path(self.metrics_cfg.get("output_csv", "./reports/sbir_metrics.csv"))
        self.output_latex = Path(self.metrics_cfg.get("output_latex", "./reports/sbir_table.tex"))
        self.benchmark_runs = self.metrics_cfg.get("benchmark_runs", 100)
        self.chemical_cost = self.metrics_cfg.get("chemical_cost_per_acre", 45.0)
        self.target_reduction = self.metrics_cfg.get("target_reduction_pct", 30)

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.output_latex.parent.mkdir(parents=True, exist_ok=True)

    def _benchmark_inference(self, model_path: Optional[Path] = None) -> dict:
        """Benchmark model inference latency on current GPU."""
        if not HAS_YOLO or not HAS_TORCH:
            return {"mean_ms": 0, "std_ms": 0, "p95_ms": 0, "p99_ms": 0, "fps": 0}

        # Find model
        if model_path is None:
            export_dir = Path(self.cfg["paths"]["export_dir"]).expanduser()
            candidates = list(export_dir.glob("best.*"))
            if not candidates:
                model_dir = Path(self.cfg["paths"]["model_output"]).expanduser()
                candidates = list(model_dir.rglob("best.pt"))
            if not candidates:
                print("[Metrics] No model found for benchmarking")
                return {"mean_ms": 0, "std_ms": 0, "p95_ms": 0, "p99_ms": 0, "fps": 0}
            model_path = candidates[0]

        print(f"[Metrics] Benchmarking {model_path.name} ({self.benchmark_runs} runs)...")
        model = YOLO(str(model_path))

        # Warm-up
        imgsz = self.cfg["training"].get("imgsz", 640)
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(10):
            model.predict(dummy, verbose=False)

        # Benchmark
        latencies = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            model.predict(dummy, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        return {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "std_ms": round(float(np.std(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "fps": round(1000.0 / float(np.mean(latencies)), 1) if np.mean(latencies) > 0 else 0,
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
        }

    def _load_validation_results(self) -> dict:
        """Load validation report if available."""
        report = Path("./reports/validation/validation_report.json")
        if report.exists():
            with open(report) as f:
                return json.load(f)
        return {}

    def _load_swarm_results(self) -> dict:
        """Load swarm simulation metrics if available."""
        report = Path("./reports/swarm/metrics.json")
        if report.exists():
            with open(report) as f:
                return json.load(f)
        return {}

    def _compute_economic_impact(self, swarm_metrics: dict) -> dict:
        """Estimate economic impact from targeted vs blanket treatment."""
        chemical_savings_pct = swarm_metrics.get("chemical_savings_pct", self.target_reduction)
        field_acres = swarm_metrics.get("field_acres", 100)

        blanket_cost = self.chemical_cost * field_acres
        targeted_cost = blanket_cost * (1 - chemical_savings_pct / 100)
        savings_per_field = blanket_cost - targeted_cost

        # Scale to typical Ohio farm (300 acres)
        farm_acres = 300
        annual_savings = savings_per_field * (farm_acres / field_acres)

        # County-level estimate (Darke County OH: ~400k cropland acres)
        county_acres = 400_000
        county_savings = savings_per_field * (county_acres / field_acres)

        return {
            "blanket_cost_per_field": round(blanket_cost, 2),
            "targeted_cost_per_field": round(targeted_cost, 2),
            "savings_per_field": round(savings_per_field, 2),
            "savings_pct": round(chemical_savings_pct, 1),
            "annual_savings_300ac_farm": round(annual_savings, 2),
            "county_savings_estimate": round(county_savings, 2),
            "cost_per_acre_chemical": self.chemical_cost,
        }

    def _get_vram_usage(self) -> dict:
        """Get current GPU VRAM usage."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {"used_gb": 0, "total_gb": 0, "free_gb": 0}
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return {
            "used_gb": round(used / (1024**3), 2),
            "total_gb": round(total / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "device": torch.cuda.get_device_name(0),
        }

    def generate_sbir_report(self) -> dict:
        """Generate the full SBIR metrics report."""
        print("[Metrics] Generating SBIR report...")

        # Gather all metrics
        inference = self._benchmark_inference()
        validation = self._load_validation_results()
        swarm = self._load_swarm_results()
        economics = self._compute_economic_impact(swarm)
        vram = self._get_vram_usage()

        report = {
            "model_performance": {
                "mAP@50": validation.get("mAP@50", "N/A"),
                "ap_per_class": validation.get("ap_per_class", {}),
                "total_classes": len(self.cfg["synthetic"]["disease_classes"]),
            },
            "inference_benchmark": inference,
            "swarm_simulation": {
                "agents": swarm.get("num_agents", "N/A"),
                "coverage_pct": swarm.get("coverage_pct", "N/A"),
                "field_acres": swarm.get("field_acres", "N/A"),
                "total_detections": swarm.get("total_detections", "N/A"),
            },
            "economic_impact": economics,
            "hardware": {
                "gpu": vram.get("device", "N/A"),
                "vram_total_gb": vram.get("total_gb", "N/A"),
                "vram_used_gb": vram.get("used_gb", "N/A"),
                "target_device": "Jetson Orin Nano",
                "target_precision": self.cfg["jetson"].get("tensorrt_precision", "fp16"),
            },
        }

        # ── Write CSV ───────────────────────────────────────────────
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value", "Unit", "Notes"])

            writer.writerow(["mAP@50", validation.get("mAP@50", "—"),
                             "", "10 disease/stress classes"])
            writer.writerow(["Inference Mean", inference["mean_ms"],
                             "ms", f"RTX 5080, {self.benchmark_runs} runs"])
            writer.writerow(["Inference P95", inference["p95_ms"],
                             "ms", ""])
            writer.writerow(["FPS", inference["fps"],
                             "frames/s", ""])
            writer.writerow(["Swarm Coverage", swarm.get("coverage_pct", "—"),
                             "%", f"{swarm.get('num_agents', '—')} agents"])
            writer.writerow(["Chemical Savings", economics["savings_pct"],
                             "%", "targeted vs blanket"])
            writer.writerow(["Annual Savings (300ac)", economics["annual_savings_300ac_farm"],
                             "USD", ""])
            writer.writerow(["VRAM Peak", vram.get("used_gb", "—"),
                             "GB", vram.get("device", "")])

            # Per-class AP
            for cls, ap in validation.get("ap_per_class", {}).items():
                writer.writerow([f"AP@50 {cls}", ap, "", ""])

        print(f"[Metrics] CSV: {self.output_csv}")

        # ── Write LaTeX table ───────────────────────────────────────
        with open(self.output_latex, "w") as f:
            f.write("% AGRI-FORGE SBIR Metrics — auto-generated\n")
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\caption{AGRI-FORGE System Performance Metrics}\n")
            f.write("\\label{tab:agri-forge-metrics}\n")
            f.write("\\begin{tabular}{lrrl}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Metric} & \\textbf{Value} & \\textbf{Unit} & \\textbf{Notes} \\\\\n")
            f.write("\\midrule\n")

            rows = [
                ("mAP@50", validation.get("mAP@50", "—"), "", "10 classes"),
                ("Inference Latency", inference["mean_ms"], "ms", "RTX 5080"),
                ("Inference P95", inference["p95_ms"], "ms", ""),
                ("Throughput", inference["fps"], "FPS", ""),
                ("Swarm Coverage", swarm.get("coverage_pct", "—"), "\\%", f"{swarm.get('num_agents', '—')} UAVs"),
                ("Chemical Reduction", economics["savings_pct"], "\\%", "vs blanket spray"),
                ("Annual Savings", f"\\${economics['annual_savings_300ac_farm']:,.0f}", "", "300-acre farm"),
                ("VRAM Usage", vram.get("used_gb", "—"), "GB", "14 GB budget"),
            ]

            for name, val, unit, note in rows:
                f.write(f"{name} & {val} & {unit} & {note} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"[Metrics] LaTeX: {self.output_latex}")

        # ── Save full JSON ──────────────────────────────────────────
        json_path = self.output_csv.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[Metrics] JSON: {json_path}")

        # ── Console summary ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f" SBIR METRICS SUMMARY")
        print(f"{'='*60}")
        print(f" Model:             YOLOv8n + LoRA + Cosmos adapter")
        print(f" mAP@50:            {validation.get('mAP@50', 'N/A')}")
        print(f" Inference:         {inference['mean_ms']}ms mean / {inference['fps']} FPS")
        print(f" VRAM:              {vram.get('used_gb', 'N/A')} / {vram.get('total_gb', 'N/A')} GB")
        print(f" Swarm Coverage:    {swarm.get('coverage_pct', 'N/A')}%")
        print(f" Chemical Savings:  {economics['savings_pct']}%")
        print(f" Annual Savings:    ${economics['annual_savings_300ac_farm']:,.2f} (300-acre farm)")
        print(f" County Impact:     ${economics['county_savings_estimate']:,.2f}")
        print(f"{'='*60}")

        return report
