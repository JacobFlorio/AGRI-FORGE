"""
Florio Industries — Training Data Merger
==========================================
Merges synthetic data from your data factory with real
farmer-annotated data from field pilots.

Why both?
  - Synthetic data gives you volume and class coverage (thousands of samples)
  - Real data gives you ground truth in YOUR conditions (your camera, your altitude, your crops)
  - Combined, they train a model that's both broadly capable and locally accurate

Input sources:
  1. Real annotations: exported from Jetson dashboard (YOLO format)
  2. Synthetic data: from your data factory (YOLO format)

Output: merged dataset ready for YOLOv8 fine-tuning

Usage:
  python3 merge_training_data.py \
    --real ~/firmament_data/training_export \
    --synthetic ~/synthetic_data/ag_v1 \
    --output ~/training/merged_ag_v1 \
    --real-weight 3.0

The --real-weight flag oversamples real data relative to synthetic.
Real data is more valuable per sample, so we duplicate it to give it
more influence during training. A weight of 3.0 means each real image
appears 3x in the training set.
"""

import argparse
import os
import shutil
import random
import json
from pathlib import Path
from collections import Counter


def count_dataset(data_dir: str) -> dict:
    """Count images, labels, and class distribution in a YOLO dataset."""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    images = set(Path(f).stem for f in os.listdir(images_dir)
                 if f.endswith((".jpg", ".png"))) if os.path.isdir(images_dir) else set()
    labels = set(Path(f).stem for f in os.listdir(labels_dir)
                 if f.endswith(".txt")) if os.path.isdir(labels_dir) else set()

    matched = images & labels

    # Count class distribution
    class_counts = Counter()
    for stem in matched:
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    return {
        "images": len(images),
        "labels": len(labels),
        "matched": len(matched),
        "class_distribution": dict(class_counts),
        "stems": matched,
    }


def merge_class_maps(real_classes_path: str, synth_classes_path: str) -> tuple:
    """
    Merge class name lists from real and synthetic datasets.
    Returns unified class list and remapping dicts.
    """
    real_classes = []
    synth_classes = []

    if os.path.exists(real_classes_path):
        with open(real_classes_path) as f:
            real_classes = [line.strip() for line in f if line.strip()]

    if os.path.exists(synth_classes_path):
        with open(synth_classes_path) as f:
            synth_classes = [line.strip() for line in f if line.strip()]

    # Build unified class list — real classes first, then any new synthetic-only classes
    unified = list(real_classes)
    for cls in synth_classes:
        if cls not in unified:
            unified.append(cls)

    # Create remapping: old_id → new_id
    real_remap = {i: unified.index(cls) for i, cls in enumerate(real_classes)}
    synth_remap = {i: unified.index(cls) for i, cls in enumerate(synth_classes)}

    return unified, real_remap, synth_remap


def copy_and_remap(
    stems: set,
    src_images: str,
    src_labels: str,
    dst_images: str,
    dst_labels: str,
    class_remap: dict,
    prefix: str,
    copies: int = 1,
):
    """
    Copy image/label pairs with optional class remapping and oversampling.
    Prefix prevents filename collisions between real and synthetic.
    """
    copied = 0
    for copy_idx in range(copies):
        for stem in stems:
            # Find image (jpg or png)
            src_img = None
            for ext in [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]:
                candidate = os.path.join(src_images, f"{stem}{ext}")
                if os.path.exists(candidate):
                    src_img = candidate
                    break

            src_lbl = os.path.join(src_labels, f"{stem}.txt")
            if not src_img or not os.path.exists(src_lbl):
                continue

            # Destination filenames with prefix and copy index
            suffix = f"_c{copy_idx}" if copies > 1 else ""
            dst_stem = f"{prefix}_{stem}{suffix}"
            img_ext = Path(src_img).suffix
            dst_img = os.path.join(dst_images, f"{dst_stem}{img_ext}")
            dst_lbl = os.path.join(dst_labels, f"{dst_stem}.txt")

            # Copy image
            shutil.copy2(src_img, dst_img)

            # Copy and remap labels
            with open(src_lbl) as f_in, open(dst_lbl, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class = int(parts[0])
                        new_class = class_remap.get(old_class, old_class)
                        f_out.write(f"{new_class} {' '.join(parts[1:])}\n")

            copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Merge real and synthetic training data for agricultural YOLO model"
    )
    parser.add_argument("--real", required=True, help="Path to real (farmer-annotated) dataset")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic dataset")
    parser.add_argument("--output", required=True, help="Output path for merged dataset")
    parser.add_argument("--real-weight", type=float, default=3.0,
                        help="Oversample real data by this factor (default: 3.0)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  FLORIO INDUSTRIES — Training Data Merger")
    print(f"{'='*60}\n")

    # ── Analyze inputs ──
    print("Analyzing real dataset...")
    real_info = count_dataset(args.real)
    print(f"  Matched pairs: {real_info['matched']}")
    print(f"  Classes: {real_info['class_distribution']}")

    print("\nAnalyzing synthetic dataset...")
    synth_info = count_dataset(args.synthetic)
    print(f"  Matched pairs: {synth_info['matched']}")
    print(f"  Classes: {synth_info['class_distribution']}")

    # ── Merge class maps ──
    real_classes_path = os.path.join(args.real, "classes.txt")
    synth_classes_path = os.path.join(args.synthetic, "classes.txt")
    unified_classes, real_remap, synth_remap = merge_class_maps(
        real_classes_path, synth_classes_path
    )
    print(f"\nUnified classes ({len(unified_classes)}): {unified_classes}")

    # ── Create output structure ──
    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.output, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output, split, "labels"), exist_ok=True)

    # ── Split and copy ──
    random.seed(args.seed)

    # Split real data
    real_stems = sorted(real_info["stems"])
    random.shuffle(real_stems)
    real_val_count = int(len(real_stems) * args.val_ratio)
    real_val = set(real_stems[:real_val_count])
    real_train = set(real_stems[real_val_count:])

    # Split synthetic data
    synth_stems = sorted(synth_info["stems"])
    random.shuffle(synth_stems)
    synth_val_count = int(len(synth_stems) * args.val_ratio)
    synth_val = set(synth_stems[:synth_val_count])
    synth_train = set(synth_stems[synth_val_count:])

    real_copies = max(1, int(args.real_weight))

    print(f"\nCopying data (real weight: {args.real_weight}x)...")

    # Copy training data
    real_train_count = copy_and_remap(
        real_train,
        os.path.join(args.real, "images"), os.path.join(args.real, "labels"),
        os.path.join(args.output, "train", "images"),
        os.path.join(args.output, "train", "labels"),
        real_remap, prefix="real", copies=real_copies,
    )
    synth_train_count = copy_and_remap(
        synth_train,
        os.path.join(args.synthetic, "images"), os.path.join(args.synthetic, "labels"),
        os.path.join(args.output, "train", "images"),
        os.path.join(args.output, "train", "labels"),
        synth_remap, prefix="synth",
    )

    # Copy validation data (no oversampling for val — keep it representative)
    real_val_count = copy_and_remap(
        real_val,
        os.path.join(args.real, "images"), os.path.join(args.real, "labels"),
        os.path.join(args.output, "val", "images"),
        os.path.join(args.output, "val", "labels"),
        real_remap, prefix="real",
    )
    synth_val_count_out = copy_and_remap(
        synth_val,
        os.path.join(args.synthetic, "images"), os.path.join(args.synthetic, "labels"),
        os.path.join(args.output, "val", "images"),
        os.path.join(args.output, "val", "labels"),
        synth_remap, prefix="synth",
    )

    # ── Write dataset.yaml ──
    yaml_path = os.path.join(args.output, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write("# Florio Industries — Merged Agricultural Training Dataset\n")
        f.write(f"# Real data: {len(real_train)} samples × {real_copies} = {real_train_count} train\n")
        f.write(f"# Synthetic data: {synth_train_count} train\n")
        f.write(f"# Total train: {real_train_count + synth_train_count}\n")
        f.write(f"# Total val: {real_val_count + synth_val_count_out}\n\n")
        f.write(f"path: {os.path.abspath(args.output)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n\n")
        f.write(f"nc: {len(unified_classes)}\n")
        f.write(f"names: {unified_classes}\n")

    # Write unified classes
    with open(os.path.join(args.output, "classes.txt"), "w") as f:
        for cls in unified_classes:
            f.write(f"{cls}\n")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Merge complete!")
    print(f"{'='*60}")
    print(f"  Train: {real_train_count} real + {synth_train_count} synthetic = {real_train_count + synth_train_count}")
    print(f"  Val:   {real_val_count} real + {synth_val_count_out} synthetic = {real_val_count + synth_val_count_out}")
    print(f"  Classes: {len(unified_classes)}")
    print(f"  Output: {args.output}")
    print(f"  Config: {yaml_path}")
    print(f"\n  Train command:")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=80 imgsz=640 batch=16")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
