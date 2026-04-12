"""
tests/test_agri_forge.py — Unit tests + VRAM benchmark
=======================================================
Run: python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import normalize_config so tests see the same canonical schema as runtime
from agri_forge import normalize_config


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────
@pytest.fixture
def cfg():
    """Load the project config (normalized)."""
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    return normalize_config(raw)


@pytest.fixture
def tmp_data_root(tmp_path, cfg):
    """Override data_root to a temp directory."""
    cfg["paths"]["data_root"] = str(tmp_path / "data")
    cfg["paths"]["model_output"] = str(tmp_path / "models")
    cfg["paths"]["export_dir"] = str(tmp_path / "export")
    return cfg


# ────────────────────────────────────────────────────────────────────
# Test 1: Synthetic generator produces valid YOLO labels
# ────────────────────────────────────────────────────────────────────
class TestSyntheticGenerator:
    def test_generate_small_batch(self, tmp_data_root):
        """Generate 10 images and verify label format in train/ dir."""
        from data.synthetic_generator import SyntheticGenerator

        gen = SyntheticGenerator(tmp_data_root)
        gen.generate(num_images=10)

        synth_root = Path(tmp_data_root["paths"]["data_root"]) / "synthetic"
        train_img_dir = synth_root / "train" / "images"
        train_lbl_dir = synth_root / "train" / "labels"

        images = list(train_img_dir.glob("*.png"))
        assert len(images) == 10, f"Expected 10 train images, got {len(images)}"

        labels = list(train_lbl_dir.glob("*.txt"))
        assert len(labels) == 10, f"Expected 10 train labels, got {len(labels)}"

        disease_classes = tmp_data_root["synthetic"]["disease_classes"]
        for lbl in labels:
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5, f"Bad label line: {line}"
                    cls_idx = int(parts[0])
                    cx, cy, bw, bh = [float(p) for p in parts[1:]]
                    assert 0 <= cls_idx < len(disease_classes)
                    assert 0 <= cx <= 1, f"cx out of range: {cx}"
                    assert 0 <= cy <= 1, f"cy out of range: {cy}"
                    assert 0 < bw <= 1, f"bw out of range: {bw}"
                    assert 0 < bh <= 1, f"bh out of range: {bh}"

        ds_yaml = synth_root / "dataset.yaml"
        assert ds_yaml.exists(), "dataset.yaml not generated"

        with open(ds_yaml) as f:
            ds = yaml.safe_load(f)
        assert ds["nc"] == len(disease_classes)

    def test_full_res_and_train_directories(self, tmp_data_root):
        """Verify both full_res/ and train/ are populated with matching counts."""
        from data.synthetic_generator import SyntheticGenerator
        from PIL import Image

        gen = SyntheticGenerator(tmp_data_root)
        gen.generate(num_images=5)

        synth_root = Path(tmp_data_root["paths"]["data_root"]) / "synthetic"
        full_res_imgs = sorted((synth_root / "full_res").glob("*.png"))
        train_imgs = sorted((synth_root / "train" / "images").glob("*.png"))
        train_lbls = sorted((synth_root / "train" / "labels").glob("*.txt"))

        assert len(full_res_imgs) == 5, f"Expected 5 full_res images, got {len(full_res_imgs)}"
        assert len(train_imgs) == 5, f"Expected 5 train images, got {len(train_imgs)}"
        assert len(train_lbls) == 5, f"Expected 5 train labels, got {len(train_lbls)}"

        # Full-res images should match render resolution
        render_res = tuple(tmp_data_root["synthetic"]["resolution"])
        for img_path in full_res_imgs:
            img = Image.open(img_path)
            assert img.size == render_res, \
                f"Full-res {img_path.name} has wrong resolution: {img.size} (expected {render_res})"

        # Train images should match inference resolution
        infer_res = tmp_data_root["synthetic"]["inference_resolution"]
        expected_train_res = (infer_res, infer_res)
        for img_path in train_imgs:
            img = Image.open(img_path)
            assert img.size == expected_train_res, \
                f"Train {img_path.name} has wrong resolution: {img.size} (expected {expected_train_res})"

        # Filenames should match between full_res and train
        full_names = [p.name for p in full_res_imgs]
        train_names = [p.name for p in train_imgs]
        assert full_names == train_names, "full_res and train filenames don't match"

    def test_image_resolution(self, tmp_data_root):
        """Verify full-res images match configured render resolution."""
        from data.synthetic_generator import SyntheticGenerator
        from PIL import Image

        gen = SyntheticGenerator(tmp_data_root)
        gen.generate(num_images=3)

        expected_res = tuple(tmp_data_root["synthetic"]["resolution"])
        img_dir = Path(tmp_data_root["paths"]["data_root"]) / "synthetic" / "full_res"
        for img_path in img_dir.glob("*.png"):
            img = Image.open(img_path)
            assert img.size == expected_res, \
                f"Image {img_path.name} has wrong resolution: {img.size}"


# ────────────────────────────────────────────────────────────────────
# Test 2: Swarm engine produces valid mission plans
# ────────────────────────────────────────────────────────────────────
class TestSwarmEngine:
    def test_swarm_small_run(self, tmp_data_root):
        """Run a small swarm sim and verify outputs."""
        from simulation.swarm_engine import SwarmEngine

        engine = SwarmEngine(tmp_data_root)
        metrics = engine.run(num_agents=5, duration_min=1)

        assert "coverage_pct" in metrics
        assert "total_detections" in metrics
        assert "chemical_savings_pct" in metrics
        assert metrics["num_agents"] == 5
        assert 0 <= metrics["coverage_pct"] <= 100
        assert metrics["chemical_savings_pct"] >= 0

        assert (Path("./reports/swarm/mission_plan.json")).exists()
        assert (Path("./reports/swarm/metrics.json")).exists()
        assert (Path("./reports/swarm/coverage_heatmap.csv")).exists()

        with open("./reports/swarm/mission_plan.json") as f:
            plan = json.load(f)
        assert "agents" in plan
        assert len(plan["agents"]) == 5
        for agent in plan["agents"]:
            assert "id" in agent
            assert "persona" in agent
            assert agent["persona"] in ("scout", "inspector", "relay")


# ────────────────────────────────────────────────────────────────────
# Test 3: Metrics generator produces valid CSV + LaTeX
# ────────────────────────────────────────────────────────────────────
class TestMetricsGenerator:
    def test_economic_impact(self, tmp_data_root):
        """Verify economic calculations are sane."""
        from utils.metrics import MetricsGenerator

        gen = MetricsGenerator(tmp_data_root)
        economics = gen._compute_economic_impact({
            "chemical_savings_pct": 30,
            "field_acres": 100,
        })

        assert economics["savings_pct"] == 30
        assert economics["targeted_cost_per_field"] < economics["blanket_cost_per_field"]
        assert economics["savings_per_field"] > 0
        assert economics["annual_savings_300ac_farm"] > 0


# ────────────────────────────────────────────────────────────────────
# Test 4: VRAM benchmark
# ────────────────────────────────────────────────────────────────────
class TestVRAMBenchmark:
    @pytest.mark.skipif(
        not (True and __import__("torch").cuda.is_available()),
        reason="No GPU available",
    )
    def test_vram_under_budget(self):
        """Verify that synthetic generation stays under 14 GB VRAM."""
        import torch

        torch.cuda.reset_peak_memory_stats()
        initial = torch.cuda.memory_allocated() / (1024**3)

        from data.synthetic_generator import SyntheticGenerator

        cfg_path = Path(__file__).parent.parent / "config.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        cfg = normalize_config(raw)

        cfg["paths"]["data_root"] = "/tmp/agri_forge_vram_test"
        cfg["synthetic"]["cosmos_model"] = "disabled"

        gen = SyntheticGenerator(cfg)
        gen.generate(num_images=20)

        peak = torch.cuda.max_memory_allocated() / (1024**3)
        budget = cfg["hardware"]["max_vram_gb"]

        print(f"\n[VRAM] Initial: {initial:.2f} GB")
        print(f"[VRAM] Peak:    {peak:.2f} GB")
        print(f"[VRAM] Budget:  {budget} GB")

        assert peak < budget, \
            f"VRAM peak ({peak:.2f} GB) exceeds budget ({budget} GB)"


# ────────────────────────────────────────────────────────────────────
# Test 5: Isaac bridge path translation
# ────────────────────────────────────────────────────────────────────
class TestIsaacBridge:
    def test_wsl_to_win_mnt_path(self):
        """Verify /mnt/c/... -> C:\\... translation."""
        from simulation.isaac_bridge import wsl_to_win
        assert wsl_to_win("/mnt/c/isaacsim") == r"C:\isaacsim"
        assert wsl_to_win("/mnt/d/data/foo") == r"D:\data\foo"

    def test_vram_guard_init(self, tmp_data_root):
        """VRAMGuard initializes without error."""
        from simulation.isaac_bridge import VRAMGuard
        guard = VRAMGuard(max_gb=14.0)
        usage = guard.get_usage()
        assert "used_gb" in usage
        assert "free_gb" in usage
        assert "total_gb" in usage

    def test_bridge_shared_dir(self, tmp_data_root):
        """Bridge creates shared directory structure."""
        from simulation.isaac_bridge import IsaacBridge
        tmp_shared = str(Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared")
        tmp_data_root["isaac_sim"]["shared_dir"] = tmp_shared
        tmp_data_root["isaac_sim"]["install_path"] = tmp_shared  # fake
        bridge = IsaacBridge(tmp_data_root)
        shared = bridge.setup_shared_dir()
        assert (shared / "renders").is_dir()
        assert (shared / "labels").is_dir()
        assert (shared / "depth").is_dir()
        assert (shared / "segmentation").is_dir()
        assert (shared / "usd_scenes").is_dir()


# ────────────────────────────────────────────────────────────────────
# Test 6: Isaac scene generator produces valid YOLO boxes
# ────────────────────────────────────────────────────────────────────
class TestIsaacSceneGenerator:
    def test_yolo_box_projection(self, tmp_data_root):
        """Verify YOLO boxes from scene params are valid."""
        tmp_data_root["isaac_sim"]["install_path"] = "/tmp/fake_isaac"
        tmp_data_root["isaac_sim"]["shared_dir"] = str(
            Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared"
        )
        from simulation.isaac_scene_generator import IsaacSceneGenerator
        gen = IsaacSceneGenerator(tmp_data_root)

        all_boxes = []
        for i in range(10):
            scene = gen._random_scene_params(i)
            # Force a disease patch near camera center for reliable test
            for p in scene["diseases"]:
                p["center_x"] = scene["camera"]["x"]
                p["center_y"] = scene["camera"]["y"]
            boxes = gen._compute_yolo_boxes(scene)
            all_boxes.extend(boxes)

        assert len(all_boxes) > 0, "No YOLO boxes generated across 10 scenes"
        for box in all_boxes:
            assert 0 <= box["cx"] <= 1, f"cx out of range: {box['cx']}"
            assert 0 <= box["cy"] <= 1, f"cy out of range: {box['cy']}"
            assert 0 < box["bw"] <= 1, f"bw out of range: {box['bw']}"
            assert 0 < box["bh"] <= 1, f"bh out of range: {box['bh']}"
            assert box["class_idx"] >= 0

    def test_procedural_fallback(self, tmp_data_root):
        """Verify procedural fallback generates images when Isaac is absent."""
        tmp_data_root["isaac_sim"]["install_path"] = "/tmp/fake_isaac"
        tmp_data_root["isaac_sim"]["shared_dir"] = str(
            Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared"
        )
        from simulation.isaac_scene_generator import IsaacSceneGenerator
        gen = IsaacSceneGenerator(tmp_data_root)

        scenes = [gen._random_scene_params(i) for i in range(3)]
        gen._procedural_fallback(scenes, 0)

        img_dir = gen.output_dir / "images"
        images = list(img_dir.glob("isaac_*.png"))
        assert len(images) == 3, f"Expected 3 fallback images, got {len(images)}"


# ────────────────────────────────────────────────────────────────────
# Test 7: VLM dataset builder produces valid instruction-tuned JSONL
# ────────────────────────────────────────────────────────────────────
class TestVLMDatasetBuilder:
    def test_build_small_dataset(self, tmp_data_root):
        """Build a small VLM dataset and verify JSONL format."""
        tmp_data_root["isaac_sim"]["install_path"] = "/tmp/fake_isaac"
        tmp_data_root["isaac_sim"]["shared_dir"] = str(
            Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared"
        )
        from training.vlm_dataset_builder import VLMDatasetBuilder

        builder = VLMDatasetBuilder(tmp_data_root)
        builder.build(num_images=5, headless=True)

        # Check images were generated as JPEG (domain gap mitigation)
        img_dir = builder.images_dir
        images = list(img_dir.glob("*.jpg"))
        assert len(images) == 5, f"Expected 5 JPEG images, got {len(images)}"

        # Verify VLM images are 256x256 JPEG
        from PIL import Image
        for img_path in images:
            img = Image.open(img_path)
            assert img.size == (256, 256), \
                f"VLM image should be 256x256, got {img.size}"
            assert img_path.suffix == ".jpg", \
                f"VLM image should be JPEG, got {img_path.suffix}"

        # Check JSONL was generated with valid format
        jsonl_path = builder.output_dir / "vlm_finetune.jsonl"
        assert jsonl_path.exists(), "vlm_finetune.jsonl not generated"

        records = []
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                records.append(record)

        assert len(records) >= 5, f"Expected at least 5 Q&A records, got {len(records)}"

        for record in records:
            assert "image" in record, "Record missing 'image' field"
            assert "conversations" in record, "Record missing 'conversations' field"
            assert len(record["conversations"]) == 2, "Expected human+gpt turn pair"
            assert record["conversations"][0]["from"] == "human"
            assert record["conversations"][1]["from"] == "gpt"
            assert "<image>" in record["conversations"][0]["value"], \
                "Human turn should contain <image> token"
            # Verify structured output format
            gpt_answer = record["conversations"][1]["value"]
            assert len(gpt_answer) > 20, "GPT answer should be substantive"
            assert "OBSERVATION:" in gpt_answer, \
                "GPT answer must contain OBSERVATION field"
            assert "DIAGNOSIS:" in gpt_answer, \
                "GPT answer must contain DIAGNOSIS field"
            assert "SEVERITY:" in gpt_answer, \
                "GPT answer must contain SEVERITY field"
            assert "CONFIDENCE:" in gpt_answer, \
                "GPT answer must contain CONFIDENCE field"
            assert "ACTION:" in gpt_answer, \
                "GPT answer must contain ACTION field"

        # Check summary file
        summary_path = builder.output_dir / "dataset_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["total_images"] == 5
        assert summary["render_engine"] == "procedural"

    def test_qa_pair_content_quality(self, tmp_data_root):
        """Verify Q&A pairs contain relevant agricultural content in structured format."""
        tmp_data_root["isaac_sim"]["install_path"] = "/tmp/fake_isaac"
        tmp_data_root["isaac_sim"]["shared_dir"] = str(
            Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared"
        )
        from training.vlm_dataset_builder import VLMDatasetBuilder, _classify_growth_stage

        # Test growth stage classification
        stage = _classify_growth_stage(0.3, 0.5, "corn")
        assert "vegetative" in stage["growth_stage"].lower()

        stage = _classify_growth_stage(1.8, 0.9, "corn")
        assert "tassel" in stage["growth_stage"].lower() or "reproductive" in stage["growth_stage"].lower()

        # Test that builder generates structured responses with disease info
        builder = VLMDatasetBuilder(tmp_data_root)
        scene = {
            "crop_type": "corn",
            "altitude_m": 50,
            "camera": {"x": 0, "y": 0, "z": 50},
            "weather": "clear",
            "lighting": "midday",
            "field": {"plant_height_m": 1.2, "plant_density": 0.85, "row_spacing_m": 0.76},
            "diseases": [
                {"disease": "gray_leaf_spot", "radius_m": 5, "num_instances": 8},
                {"disease": "nitrogen_deficiency", "radius_m": 10, "num_instances": 3},
            ],
            "lighting_params": {"sun_elevation": 70},
        }
        pairs = builder._build_qa_pairs(scene)
        assert len(pairs) >= 2, "Should generate at least 2 Q&A pairs"

        # Check structured format and disease mentions
        all_answers = " ".join(p["answer"] for p in pairs)
        assert "OBSERVATION:" in all_answers
        assert "DIAGNOSIS:" in all_answers
        assert "SEVERITY:" in all_answers
        assert "CONFIDENCE:" in all_answers
        assert "ACTION:" in all_answers
        assert "gray leaf spot" in all_answers.lower() or "gray_leaf_spot" in all_answers.lower()

    def test_healthy_field_no_treatment(self, tmp_data_root):
        """Verify healthy-only scenes don't recommend unnecessary treatments."""
        tmp_data_root["isaac_sim"]["install_path"] = "/tmp/fake_isaac"
        tmp_data_root["isaac_sim"]["shared_dir"] = str(
            Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared"
        )
        from training.vlm_dataset_builder import VLMDatasetBuilder

        builder = VLMDatasetBuilder(tmp_data_root)
        scene = {
            "crop_type": "soybean",
            "altitude_m": 40,
            "camera": {"x": 0, "y": 0, "z": 40},
            "weather": "clear",
            "lighting": "morning",
            "field": {"plant_height_m": 0.8, "plant_density": 0.7, "row_spacing_m": 0.76},
            "diseases": [{"disease": "healthy", "radius_m": 5, "num_instances": 1}],
            "lighting_params": {"sun_elevation": 20},
        }
        pairs = builder._build_qa_pairs(scene)
        all_answers = " ".join(p["answer"] for p in pairs)
        # Should NOT recommend fungicide since all diseases are "healthy"
        assert "fungicide" not in all_answers.lower()
        # Structured format should show healthy diagnosis
        assert "SEVERITY: low" in all_answers
        assert "No treatment needed" in all_answers or "No disease" in all_answers


# ────────────────────────────────────────────────────────────────────
# Test 8: cmd_synthetic attempts Isaac Sim first
# ────────────────────────────────────────────────────────────────────
class TestSyntheticIsaacFirst:
    def test_synthetic_checks_isaac(self, tmp_data_root, capsys):
        """Verify --mode synthetic attempts Isaac Sim before procedural."""
        tmp_data_root["isaac_sim"]["install_path"] = "/tmp/fake_isaac"
        tmp_data_root["isaac_sim"]["shared_dir"] = str(
            Path(tmp_data_root["paths"]["data_root"]) / "isaac_shared"
        )
        import argparse
        from agri_forge import cmd_synthetic
        args = argparse.Namespace(
            num_images=3, headless=True, resume=False,
        )
        cmd_synthetic(tmp_data_root, args)

        captured = capsys.readouterr()
        # Should see the Isaac check message (even if it falls back)
        assert "Isaac Sim" in captured.out, \
            "cmd_synthetic should report Isaac Sim check status"
        # Since Isaac is fake, it should fall back
        assert "unavailable" in captured.out.lower() or "fallback" in captured.out.lower() or \
               "procedural" in captured.out.lower(), \
            "Should fall back to procedural when Isaac is absent"


# ────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
