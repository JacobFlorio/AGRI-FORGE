"""
Florio Industries — Agricultural Model Fine-Tuning Pipeline
=============================================================
Run this on your BIG PC (not the Jetson) after transferring
the training export directory from the Jetson via USB.

This script:
  1. Validates the exported dataset
  2. Splits into train/val sets
  3. Fine-tunes YOLOv8-nano on your farmer-verified agricultural data
  4. Exports the updated model for deployment back to Jetson

Prerequisites (on your training PC):
  pip install ultralytics opencv-python

Usage:
  python3 finetune_agricultural.py --data /path/to/training_export
  
After training completes:
  Copy best.pt to Jetson: ~/firmament_models/yolov8n_ag_v{N}.pt
  Update your detection node to load the new weights.

The data flywheel:
  Fly → Detect → Farmer verifies → Export → Train → Deploy → Fly better
  Each cycle makes the model smarter at YOUR specific crops and conditions.
"""

import argparse
import os
import shutil
import random
from pathlib import Path


def validate_dataset(data_dir: str) -> dict:
    """Check that the exported dataset has matching images and labels."""
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images directory at {images_dir}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"No labels directory at {labels_dir}")
    
    images = {Path(f).stem for f in os.listdir(images_dir) if f.endswith(".jpg")}
    labels = {Path(f).stem for f in os.listdir(labels_dir) if f.endswith(".txt")}
    
    matched = images & labels
    images_only = images - labels
    labels_only = labels - images
    
    print(f"\n  Dataset Validation:")
    print(f"  Matched pairs:    {len(matched)}")
    print(f"  Images no label:  {len(images_only)}")
    print(f"  Labels no image:  {len(labels_only)}")
    
    if len(matched) == 0:
        raise ValueError("No matched image/label pairs found!")
    
    return {
        "matched": matched,
        "images_only": images_only,
        "labels_only": labels_only,
    }


def split_dataset(data_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """
    Split the flat images/labels directories into train/val sets.
    Creates the directory structure YOLOv8 expects:
    
        data_dir/
            train/
                images/
                labels/
            val/
                images/
                labels/
    """
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    # Get all matched pairs
    stems = sorted([
        Path(f).stem for f in os.listdir(images_dir) 
        if f.endswith(".jpg") and os.path.exists(os.path.join(labels_dir, Path(f).stem + ".txt"))
    ])
    
    random.seed(seed)
    random.shuffle(stems)
    
    split_idx = int(len(stems) * (1 - val_ratio))
    train_stems = stems[:split_idx]
    val_stems = stems[split_idx:]
    
    # Create split directories
    for split_name, split_stems in [("train", train_stems), ("val", val_stems)]:
        split_img_dir = os.path.join(data_dir, split_name, "images")
        split_lbl_dir = os.path.join(data_dir, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        
        for stem in split_stems:
            shutil.copy2(
                os.path.join(images_dir, f"{stem}.jpg"),
                os.path.join(split_img_dir, f"{stem}.jpg"),
            )
            shutil.copy2(
                os.path.join(labels_dir, f"{stem}.txt"),
                os.path.join(split_lbl_dir, f"{stem}.txt"),
            )
    
    print(f"\n  Train/Val Split:")
    print(f"  Train samples:  {len(train_stems)}")
    print(f"  Val samples:    {len(val_stems)}")
    
    # Update dataset.yaml to point to split directories
    classes_path = os.path.join(data_dir, "classes.txt")
    classes = []
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            classes = [line.strip() for line in f if line.strip()]
    
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# Florio Industries — Agricultural Detection Dataset\n")
        f.write(f"# {len(train_stems)} train + {len(val_stems)} val samples\n\n")
        f.write(f"path: {os.path.abspath(data_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
    
    return len(train_stems), len(val_stems)


def finetune(
    data_dir: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",  # empty string = auto-detect GPU
    project: str = "firmament_training",
    name: str = "ag_finetune",
):
    """
    Fine-tune YOLOv8-nano on the agricultural dataset.
    
    This uses transfer learning: starting from the pretrained YOLOv8n
    weights (trained on COCO) and fine-tuning on your agricultural data.
    The model retains its general object detection ability while learning
    to recognize your specific crop anomalies.
    
    On a decent GPU (RTX 3060+), this takes 15-45 minutes for 50 epochs
    on a few hundred images. On your "big PC" it should be fast.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ERROR: ultralytics not installed.")
        print("  Run: pip install ultralytics")
        return None
    
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No dataset.yaml at {yaml_path}")
    
    print(f"\n{'='*60}")
    print(f"  FLORIO INDUSTRIES — Agricultural Model Training")
    print(f"{'='*60}")
    print(f"  Base model:     {base_model}")
    print(f"  Dataset:        {yaml_path}")
    print(f"  Epochs:         {epochs}")
    print(f"  Image size:     {imgsz}")
    print(f"  Batch size:     {batch}")
    print(f"{'='*60}\n")
    
    # Load base model
    model = YOLO(base_model)
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        # Fine-tuning specific settings
        lr0=0.001,          # Lower learning rate for fine-tuning (default is 0.01)
        lrf=0.01,           # Final learning rate ratio
        warmup_epochs=3,    # Gentle warmup
        patience=10,        # Early stopping if no improvement for 10 epochs
        save=True,
        save_period=10,     # Save checkpoint every 10 epochs
        plots=True,         # Generate training plots
        verbose=True,
    )
    
    # Find best weights
    best_path = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best_path):
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best model:  {best_path}")
        print(f"{'='*60}")
        print(f"\n  NEXT STEPS:")
        print(f"  1. Copy {best_path} to USB drive")
        print(f"  2. On Jetson: cp best.pt ~/firmament_models/yolov8n_ag_v{{N}}.pt")
        print(f"  3. Update your detection node to load the new weights")
        print(f"  4. Fly, detect, collect more farmer annotations, repeat")
        print(f"\n  The model will improve with every training cycle.")
        print(f"  Aim for 200+ verified annotations before your first retrain")
        print(f"  for meaningful improvement.\n")
    else:
        print(f"\n  WARNING: Could not find best.pt at {best_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Florio Industries — Agricultural Model Fine-Tuning"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to training_export directory from Jetson"
    )
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="",
                        help="CUDA device (e.g. '0' or 'cpu')")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate and split, don't train")
    args = parser.parse_args()
    
    # Step 1: Validate
    validate_dataset(args.data)
    
    # Step 2: Split
    split_dataset(args.data, val_ratio=args.val_ratio)
    
    if args.validate_only:
        print("\n  Validation complete. Skipping training (--validate-only).\n")
        return
    
    # Step 3: Train
    finetune(
        data_dir=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
