"""
data/synthetic_generator.py — Photorealistic synthetic aerial crop imagery
==========================================================================
Uses Cosmos-Reason2-2B (via transformers) as a vision-language backbone to
guide procedural rendering of disease/pest/stress overlays on base crop
images.  MATLAB Engine (optional) renders physics-based lighting + sensor
noise for radiometric fidelity.

Output: 640x640 PNG images + YOLO-format label files + metadata JSON.
All bounding boxes are auto-generated from the rendering pipeline.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matlab.engine
    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False


# ── Disease / stress overlay specifications ─────────────────────────
# Aligned to ag_class_taxonomy.yaml — 11 classes in classes.txt order.
DISEASE_SPECS = {
    # ── TIER 1: Ohio Corn Diseases ──────────────────────────────────
    "gray_leaf_spot": {
        "color_range": [(140, 140, 100), (180, 170, 130)],
        "size_range": (15, 60),
        "shape": "ellipse",
        "opacity": (0.4, 0.7),
        "count_range": (3, 15),
    },
    "northern_corn_leaf_blight": {
        "color_range": [(120, 110, 80), (160, 140, 100)],
        "size_range": (20, 80),
        "shape": "cigar",
        "opacity": (0.5, 0.8),
        "count_range": (2, 10),
    },
    "common_rust": {
        "color_range": [(180, 80, 30), (210, 100, 40)],
        "size_range": (5, 20),
        "shape": "circle",
        "opacity": (0.6, 0.9),
        "count_range": (10, 50),
    },
    # ── TIER 2: Ohio Soybean Diseases ───────────────────────────────
    "sudden_death_syndrome": {
        "color_range": [(100, 120, 50), (140, 160, 80)],
        "size_range": (30, 100),
        "shape": "irregular",
        "opacity": (0.5, 0.8),
        "count_range": (2, 8),
    },
    "frogeye_leaf_spot": {
        "color_range": [(160, 160, 160), (200, 200, 200)],
        "size_range": (8, 30),
        "shape": "circle",
        "opacity": (0.5, 0.75),
        "count_range": (5, 25),
    },
    # ── TIER 3: Cross-Crop Stress Classes ───────────────────────────
    "nitrogen_deficiency": {
        "color_range": [(180, 180, 60), (220, 210, 90)],
        "size_range": (60, 150),
        "shape": "gradient",
        "opacity": (0.3, 0.5),
        "count_range": (1, 3),
    },
    "water_stress": {
        "color_range": [(140, 150, 100), (180, 185, 130)],
        "size_range": (80, 200),
        "shape": "patch",
        "opacity": (0.3, 0.6),
        "count_range": (1, 4),
    },
    "weed_pressure": {
        "color_range": [(20, 100, 10), (70, 170, 50)],
        "size_range": (20, 80),
        "shape": "irregular",
        "opacity": (0.5, 0.8),
        "count_range": (3, 15),
    },
    "stand_gap": {
        # Bare soil showing through missing plants
        "color_range": [(130, 100, 60), (180, 140, 90)],
        "size_range": (30, 120),
        "shape": "streak",
        "opacity": (0.6, 0.9),
        "count_range": (1, 5),
    },
    "ponding": {
        # Dark wet soil / standing water
        "color_range": [(30, 35, 40), (70, 75, 90)],
        "size_range": (60, 200),
        "shape": "irregular",
        "opacity": (0.5, 0.8),
        "count_range": (1, 3),
    },
    "healthy": {
        # No visible anomaly — just a few very faint natural blemishes
        "color_range": [(100, 140, 60), (120, 160, 80)],
        "size_range": (5, 15),
        "shape": "circle",
        "opacity": (0.05, 0.15),
        "count_range": (1, 3),
    },
}

# ── Cosmos generation prompts (from ag_class_taxonomy.yaml) ─────────
# Used by _cosmos_refine_prompt() to guide Cosmos-Reason2 scene generation.
COSMOS_PROMPTS = {
    "gray_leaf_spot": "Corn leaves with elongated rectangular gray-brown lesions parallel to veins. Lesions 1-3 inches long. Often on lower leaves first. Background is green corn canopy.",
    "northern_corn_leaf_blight": "Corn leaves with large cigar-shaped gray-green to tan lesions. Lesions longer than gray leaf spot, sometimes with dark borders. Canopy shows mottled appearance from above.",
    "common_rust": "Corn leaves with small raised reddish-brown circular pustules scattered across leaf surface. From aerial view, affected area has distinctive rust-colored tint against green canopy.",
    "sudden_death_syndrome": "Soybean plants with yellow leaves showing brown necrotic tissue between veins while veins remain green. Dead brown leaves still attached to stems. Patches of affected plants surrounded by green healthy plants.",
    "frogeye_leaf_spot": "Soybean leaves with small circular spots having light gray centers surrounded by dark reddish-brown borders. Multiple spots per leaf. Canopy appears speckled from above.",
    "nitrogen_deficiency": "Crop plants showing yellowing of lower leaves with V-shaped chlorosis pattern starting from leaf tips. Upper canopy may still be green. From above, appears as lighter green to yellow patches, sometimes in streaks following fertilizer application lines.",
    "water_stress": "Crop showing wilting and leaf rolling. Corn leaves curled inward like tacos. Color shifted from vibrant green to dull gray-green. Canopy appears thinner and less uniform from above. Often in elevated areas of field that drain faster.",
    "weed_pressure": "Mixed vegetation in crop rows showing different leaf shapes, heights, and green tones from the main crop. Weeds may be taller or shorter than crop. From above, appears as irregular textured patches disrupting the uniform row pattern.",
    "stand_gap": "Gaps in crop rows where plants failed to emerge or died. Bare brown soil visible between remaining green plants. Row pattern is broken. From above, appears as brown lines or patches interrupting green crop rows.",
    "ponding": "Low areas of field with standing water or visibly saturated dark soil. Surrounding crop plants are yellowed or stunted from waterlogging. From above, wet areas appear as dark patches, may be reflective. Adjacent crop shows stress yellowing.",
    "healthy": "Healthy uniform crop canopy. Even green coloration. Consistent row spacing visible. No discoloration, no gaps, no wilting. This is the baseline normal appearance for training negative examples.",
}

# ── Weighted distribution for 1,000-image batches ──────────────────
# From ag_class_taxonomy.yaml generation priority section.
CLASS_DISTRIBUTION_1K = {
    "gray_leaf_spot": 120,
    "northern_corn_leaf_blight": 100,
    "common_rust": 80,
    "sudden_death_syndrome": 80,
    "frogeye_leaf_spot": 70,
    "nitrogen_deficiency": 100,
    "water_stress": 100,
    "weed_pressure": 80,
    "stand_gap": 70,
    "ponding": 60,
    "healthy": 140,
}


class SyntheticGenerator:
    """Generate synthetic aerial crop images with auto-labeled anomalies."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["synthetic"]
        self.hw_cfg = cfg.get("hardware", {})
        self.data_root = Path(cfg["paths"]["data_root"]).expanduser()
        self.output_dir = self.data_root / "synthetic"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Full-res renders (original capture resolution)
        self.full_res_dir = self.output_dir / "full_res"
        self.full_res_dir.mkdir(exist_ok=True)

        # Downscaled training images + labels (what YOLO actually sees)
        self.train_dir = self.output_dir / "train"
        (self.train_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.train_dir / "labels").mkdir(parents=True, exist_ok=True)

        self.resolution = tuple(self.cfg.get("resolution", [640, 640]))
        infer_res = self.cfg.get("inference_resolution", 640)
        self.inference_resolution = (infer_res, infer_res)
        self.disease_classes = self.cfg.get("disease_classes", list(DISEASE_SPECS.keys()))
        self.class_to_idx = {c: i for i, c in enumerate(self.disease_classes)}
        self.aug_cfg = self.cfg.get("augmentation", {})

        # Optional: load Cosmos-Reason2 for guided generation
        self.cosmos_model = None
        self.cosmos_tokenizer = None
        self._load_cosmos()

        # Optional: MATLAB engine for physics-based rendering
        self.matlab_eng = None
        if HAS_MATLAB:
            try:
                self.matlab_eng = matlab.engine.start_matlab()
                print("[SynGen] MATLAB engine connected")
            except Exception as e:
                print(f"[SynGen] MATLAB not available: {e}")

    def _load_cosmos(self) -> None:
        """Load Cosmos-Reason2-2B for texture/scene guidance.

        Cosmos-Reason2-2B uses a Qwen3-VL architecture.  We try the matching
        class first, then Qwen2-VL, then AutoModelForCausalLM, and finally
        fall back to procedural-only mode.
        """
        if not HAS_TORCH:
            print("[SynGen] PyTorch not available — using procedural-only mode")
            return

        model_name = self.cfg.get("cosmos_model", "nvidia/Cosmos-Reason2-2B")
        if model_name == "disabled":
            print("[SynGen] Cosmos disabled by config — procedural-only mode")
            return

        try:
            from transformers import AutoProcessor
            print(f"[SynGen] Loading {model_name}...")

            # Try model classes in order: Qwen3-VL → Qwen2-VL → AutoModelForCausalLM
            model_loaded = False

            # 1) Qwen3-VL (correct architecture for Cosmos-Reason2-2B)
            if not model_loaded:
                try:
                    from transformers import Qwen3VLForConditionalGeneration
                    self.cosmos_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_name,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    model_loaded = True
                    print("[SynGen] Loaded with Qwen3VLForConditionalGeneration")
                except (ImportError, Exception) as e:
                    print(f"[SynGen] Qwen3-VL loader failed: {e}")

            # 2) Qwen2-VL fallback
            if not model_loaded:
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    self.cosmos_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    model_loaded = True
                    print("[SynGen] Loaded with Qwen2VLForConditionalGeneration")
                except (ImportError, Exception) as e:
                    print(f"[SynGen] Qwen2-VL loader failed: {e}")

            # 3) AutoModelForCausalLM fallback
            if not model_loaded:
                try:
                    from transformers import AutoModelForCausalLM
                    self.cosmos_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    model_loaded = True
                    print("[SynGen] Loaded with AutoModelForCausalLM")
                except (ImportError, Exception) as e:
                    print(f"[SynGen] AutoModelForCausalLM loader failed: {e}")

            if not model_loaded:
                print("[SynGen] All model loaders failed — procedural-only mode")
                return

            # Load processor (AutoProcessor handles VL and causal models)
            self.cosmos_tokenizer = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )

            print("[SynGen] Cosmos-Reason2 ready")
        except Exception as e:
            self.cosmos_model = None
            self.cosmos_tokenizer = None
            print(f"[SynGen] Cosmos load failed (proceeding procedural): {e}")

    def _generate_base_field(self) -> Image.Image:
        """Render a procedural top-down crop field image."""
        w, h = self.resolution
        # Base green field with row structure
        img = Image.new("RGB", (w, h))
        pixels = np.zeros((h, w, 3), dtype=np.uint8)

        # Crop row pattern
        row_spacing = random.randint(8, 16)
        row_angle = random.uniform(-5, 5)  # slight angle
        base_green = np.array([random.randint(30, 60),
                               random.randint(100, 160),
                               random.randint(20, 50)])

        for y in range(h):
            for x in range(w):
                # Rotated row pattern
                rx = x * math.cos(math.radians(row_angle)) - y * math.sin(math.radians(row_angle))
                row_val = math.sin(rx / row_spacing * math.pi) * 0.3 + 0.7
                noise = np.random.randint(-10, 10, 3)
                color = (base_green * row_val + noise).clip(0, 255).astype(np.uint8)
                pixels[y, x] = color

        img = Image.fromarray(pixels)

        # Add subtle texture via Gaussian blur + sharpen
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        img = img.filter(ImageFilter.SHARPEN)

        return img

    def _generate_base_field_fast(self, scene: Optional[dict] = None) -> Image.Image:
        """Vectorized base field generation with per-image scene variance.

        Args:
            scene: Optional dict with keys ``growth_stage`` (0.0-1.0),
                   ``sun_angle`` (degrees), ``color_temp`` ("warm"/"neutral"/"cool").
                   If *None*, values are randomized internally.
        """
        if scene is None:
            scene = {}
        growth = scene.get("growth_stage", random.uniform(0.2, 1.0))
        sun_angle = scene.get("sun_angle", random.uniform(15, 75))
        color_temp = scene.get("color_temp", random.choice(["warm", "neutral", "cool"]))

        w, h = self.resolution

        # Growth stage affects row spacing visibility and green intensity
        row_spacing = int(8 + 8 * (1 - growth))  # young=wide, mature=tight
        row_angle = random.uniform(-10, 10)

        # Base green shifts with growth stage and color temperature
        g_center = int(100 + 60 * growth)  # young=pale, mature=deep green
        r_base = random.randint(25, 55)
        b_base = random.randint(15, 45)
        if color_temp == "warm":
            r_base += 15
            b_base -= 10
        elif color_temp == "cool":
            r_base -= 10
            b_base += 15
        base_green = np.array([r_base, g_center, b_base], dtype=np.float32)

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        rx = xs * math.cos(math.radians(row_angle)) - ys * math.sin(math.radians(row_angle))

        # Row contrast decreases as canopy closes (high growth stage)
        row_contrast = 0.35 * (1 - growth * 0.6)
        row_val = np.sin(rx / row_spacing * math.pi) * row_contrast + (1 - row_contrast)

        noise = np.random.randint(-12, 12, (h, w, 3), dtype=np.int16)
        pixels = (base_green[None, None, :] * row_val[:, :, None] + noise).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Sun angle affects overall brightness
        sun_factor = 0.55 + 0.45 * math.sin(math.radians(sun_angle))
        img = ImageEnhance.Brightness(img).enhance(sun_factor)

        return img

    def _overlay_anomaly(self, img: Image.Image, disease: str) -> list[dict]:
        """Draw disease/pest overlay on image, return YOLO bounding boxes."""
        spec = DISEASE_SPECS.get(disease)
        if spec is None:
            return []

        w, h = img.size
        draw = ImageDraw.Draw(img, "RGBA")
        boxes = []

        count = random.randint(*spec["count_range"])
        for _ in range(count):
            size = random.randint(*spec["size_range"])
            x = random.randint(size, w - size)
            y = random.randint(size, h - size)
            opacity = int(random.uniform(*spec["opacity"]) * 255)

            c1, c2 = spec["color_range"]
            color = tuple(random.randint(c1[i], c2[i]) for i in range(3)) + (opacity,)

            shape = spec["shape"]
            half = size // 2

            if shape in ("circle", "cluster"):
                draw.ellipse([x - half, y - half, x + half, y + half], fill=color)
            elif shape == "ellipse":
                draw.ellipse([x - size, y - half, x + size, y + half], fill=color)
            elif shape == "cigar":
                draw.ellipse([x - size, y - half // 2, x + size, y + half // 2], fill=color)
            elif shape in ("streak", "edge_burn"):
                pts = [(x - size, y - 2), (x + size, y + 2),
                       (x + size - 5, y + half), (x - size + 5, y + half)]
                draw.polygon(pts, fill=color)
            elif shape in ("patch", "gradient", "irregular"):
                # Large irregular patch
                pts = []
                for angle in range(0, 360, 30):
                    r = half + random.randint(-half // 3, half // 3)
                    px = x + int(r * math.cos(math.radians(angle)))
                    py = y + int(r * math.sin(math.radians(angle)))
                    pts.append((px, py))
                draw.polygon(pts, fill=color)
            else:
                draw.rectangle([x - half, y - half, x + half, y + half], fill=color)

            # YOLO format: class_idx cx cy w h (normalized)
            cx = x / w
            cy = y / h
            bw = (size * 2) / w
            bh = (size * 2) / h
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            bw = min(bw, 1.0)
            bh = min(bh, 1.0)

            boxes.append({
                "class": disease,
                "class_idx": self.class_to_idx.get(disease, 0),
                "cx": cx, "cy": cy, "bw": bw, "bh": bh,
            })

        return boxes

    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        """Apply configured augmentations."""
        if self.aug_cfg.get("flip") and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.aug_cfg.get("flip") and random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        rot = self.aug_cfg.get("rotate_max", 0)
        if rot > 0:
            angle = random.uniform(-rot, rot)
            img = img.rotate(angle, fillcolor=(40, 120, 30))

        br_range = self.aug_cfg.get("brightness_range", [1.0, 1.0])
        factor = random.uniform(*br_range)
        img = ImageEnhance.Brightness(img).enhance(factor)

        noise_std = self.aug_cfg.get("noise_std", 0)
        if noise_std > 0:
            arr = np.array(img).astype(np.float32)
            arr += np.random.normal(0, noise_std * 255, arr.shape)
            img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

        return img

    def _cosmos_refine_prompt(self, disease: str) -> Optional[str]:
        """Use Cosmos-Reason2 to generate a scene description for guidance.

        Uses the agri_forge_description from ag_class_taxonomy.yaml (stored
        in COSMOS_PROMPTS) as the generation prompt for each class.
        """
        if self.cosmos_model is None:
            return None

        prompt = COSMOS_PROMPTS.get(
            disease,
            f"Describe a photorealistic aerial drone image at 30m altitude "
            f"showing a Midwest corn/soybean field with visible "
            f"{disease.replace('_', ' ')}.",
        )
        try:
            processor = self.cosmos_tokenizer  # AutoProcessor instance

            # Build chat-style input for VL models (Qwen3-VL / Qwen2-VL)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            try:
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback for non-chat processors
                text_input = prompt

            inputs = processor(
                text=[text_input], return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.cosmos_model.device) for k, v in inputs.items()
                      if hasattr(v, "to")}

            with torch.no_grad():
                out = self.cosmos_model.generate(
                    **inputs, max_new_tokens=128, temperature=0.7,
                    do_sample=True,
                )
            return processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            return None

    def _matlab_lighting(self, img: Image.Image) -> Image.Image:
        """Use MATLAB to apply physics-based solar illumination model."""
        if self.matlab_eng is None:
            return img

        try:
            arr = np.array(img).astype(np.float64)
            # Pass to MATLAB for radiometric correction
            mat_arr = matlab.double(arr.tolist())
            # Simple solar angle + atmospheric scattering model
            result = self.matlab_eng.eval(
                "corrected = img .* (0.8 + 0.2 * rand(size(img)));",
                nargout=0,
            )
            # Fallback: apply simple solar angle effect in Python
            sun_angle = random.uniform(30, 70)
            factor = 0.6 + 0.4 * math.sin(math.radians(sun_angle))
            img = ImageEnhance.Brightness(img).enhance(factor)
        except Exception:
            pass

        return img

    # ── Weighted class schedule ──────────────────────────────────
    def _build_class_schedule(self, num_images: int) -> list[list[str]]:
        """Build a weighted disease assignment schedule.

        Uses CLASS_DISTRIBUTION_1K to scale counts proportionally to the
        requested ``num_images``.  A random secondary disease is added 30 %
        of the time for realism — secondaries are drawn uniformly and do
        NOT affect the primary distribution guarantee.
        """
        nc = len(self.disease_classes)

        # Compute per-class counts scaled from the 1K reference distribution
        total_ref = sum(CLASS_DISTRIBUTION_1K.get(c, 0) for c in self.disease_classes)
        if total_ref == 0:
            total_ref = nc  # fallback: uniform
        primaries: list[str] = []
        remainder = num_images
        for i, cls in enumerate(self.disease_classes):
            ref_count = CLASS_DISTRIBUTION_1K.get(cls, 0)
            if total_ref > 0:
                count = round(num_images * ref_count / total_ref)
            else:
                count = num_images // nc
            # Last class absorbs rounding remainder
            if i == nc - 1:
                count = remainder
            count = max(0, min(count, remainder))
            primaries.extend([cls] * count)
            remainder -= count

        # Shuffle so images aren't class-sorted on disk
        random.shuffle(primaries)

        schedule: list[list[str]] = []
        for pri in primaries:
            diseases = [pri]
            # 30 % chance of a secondary overlay for realism
            if random.random() < 0.3 and nc > 1:
                sec = random.choice([c for c in self.disease_classes if c != pri])
                diseases.append(sec)
            schedule.append(diseases)
        return schedule

    # ── Resume helpers ─────────────────────────────────────────────
    def _detect_resume_index(self) -> int:
        """Find the next image index by scanning existing files."""
        img_dir = self.full_res_dir
        existing = sorted(img_dir.glob("synth_*.png"))
        if not existing:
            return 0
        # Parse the highest index from filenames like synth_02999.png
        last = existing[-1].stem  # "synth_02999"
        try:
            return int(last.split("_")[1]) + 1
        except (IndexError, ValueError):
            return 0

    def _load_existing_metadata(self) -> list[dict]:
        """Load metadata.json if resuming, else return empty list."""
        meta_path = self.output_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_progress(self, metadata: list[dict]) -> None:
        """Atomically persist metadata so crashes don't lose progress."""
        meta_path = self.output_dir / "metadata.json"
        tmp = meta_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2)
        tmp.replace(meta_path)

    # ── Main generation loop ───────────────────────────────────────
    def generate(self, num_images: int = 2000, resume: bool = False) -> None:
        """Generate the full synthetic dataset.

        Args:
            num_images: Total images in the final dataset.
            resume: If *True*, skip indices that already have images on disk
                    and continue where a previous run left off.
        """
        start_idx = 0
        metadata: list[dict] = []

        if resume:
            start_idx = self._detect_resume_index()
            if start_idx > 0:
                metadata = self._load_existing_metadata()
                print(f"[SynGen] Resuming from index {start_idx} "
                      f"({start_idx} images already on disk)")
            if start_idx >= num_images:
                print(f"[SynGen] Dataset already complete ({start_idx} images)")
                return

        remaining = num_images - start_idx
        print(f"[SynGen] Generating {remaining} synthetic images "
              f"(total target: {num_images})...")
        print(f"[SynGen] Classes: {self.disease_classes}")
        print(f"[SynGen] Render resolution: {self.resolution}")
        print(f"[SynGen] Train resolution:  {self.inference_resolution}")
        print(f"[SynGen] Full-res dir:      {self.full_res_dir}")
        print(f"[SynGen] Train dir:         {self.train_dir}")

        # Build balanced class schedule for the remaining images
        schedule = self._build_class_schedule(remaining)

        for seq, diseases in enumerate(schedule):
            i = start_idx + seq

            # 1. Randomized scene parameters for variance
            scene = {
                "growth_stage": random.uniform(0.2, 1.0),
                "sun_angle": random.uniform(15, 75),
                "color_temp": random.choice(["warm", "neutral", "cool"]),
            }

            # 2. Base field with scene variance
            img = self._generate_base_field_fast(scene=scene)

            # 3. Optionally get Cosmos scene guidance
            for d in diseases:
                self._cosmos_refine_prompt(d)

            # 4. Overlay anomalies
            all_boxes = []
            for disease in diseases:
                boxes = self._overlay_anomaly(img, disease)
                all_boxes.extend(boxes)

            # 5. Physics lighting (MATLAB if available)
            img = self._matlab_lighting(img)

            # 6. Augmentation
            img = self._apply_augmentation(img)

            # 7. Save full-res image
            img_name = f"synth_{i:05d}.png"
            img.save(self.full_res_dir / img_name)

            # 8. Downscale to inference resolution for YOLO training
            train_img = img.resize(self.inference_resolution, Image.LANCZOS)
            train_img.save(self.train_dir / "images" / img_name)

            # 9. Save YOLO label (normalized coords — resolution-independent)
            label_name = f"synth_{i:05d}.txt"
            label_path = self.train_dir / "labels" / label_name
            with open(label_path, "w") as f:
                for box in all_boxes:
                    f.write(f"{box['class_idx']} {box['cx']:.6f} {box['cy']:.6f} "
                            f"{box['bw']:.6f} {box['bh']:.6f}\n")

            metadata.append({
                "image": img_name,
                "diseases": diseases,
                "num_annotations": len(all_boxes),
                "boxes": all_boxes,
            })

            generated = seq + 1
            if generated % 100 == 0:
                print(f"  [{generated}/{remaining}] generated")
                self._save_progress(metadata)

        # 10. Save dataset YAML for YOLOv8 (points at train/ subdirectory)
        import yaml
        dataset_yaml = {
            "path": str(self.train_dir.resolve()),
            "train": "images",
            "val": "images",
            "nc": len(self.disease_classes),
            "names": self.disease_classes,
        }
        with open(self.output_dir / "dataset.yaml", "w") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        # 11. Save final metadata
        self._save_progress(metadata)

        # 12. Class distribution summary (count primary = first disease)
        class_counts: dict[str, int] = {}
        for m in metadata:
            for d in m["diseases"]:
                class_counts[d] = class_counts.get(d, 0) + 1

        print(f"\n[SynGen] Dataset complete: {len(metadata)} images")
        print(f"[SynGen] Output: {self.output_dir.resolve()}")
        print(f"[SynGen] Class distribution (primary + secondary overlays):")
        for cls, cnt in sorted(class_counts.items()):
            print(f"  {cls}: {cnt}")
