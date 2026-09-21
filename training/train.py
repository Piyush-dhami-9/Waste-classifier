"""
Waste Detection Model Training
==============================
4 classes: RECYCLABLE, ORGANIC, HAZARDOUS, GENERAL
Dataset: training/dataset/ (9156 train, 2287 val, 1283 test)
Output: models/garbage_detect.pt

Auto-detects GPU and picks best settings:
  RTX 4060 8GB  → YOLOv8m, batch=8, imgsz=640
  RTX 3050 4GB  → YOLOv8s, batch=4, imgsz=640
  Low VRAM/CPU  → YOLOv8n, batch=4, imgsz=416
"""

import sys
import gc
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
DATASET_YAML = str(TRAINING_DIR / 'dataset' / 'data.yaml')
FINAL_MODEL = PROJECT_ROOT / 'models' / 'garbage_detect.pt'


def pick_config():
    """Auto-select model size and batch based on GPU."""
    if not torch.cuda.is_available():
        print("No GPU — training on CPU (slow)")
        return 'yolov8n.pt', 4, 416, 'cpu'

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    torch.cuda.empty_cache()
    gc.collect()

    if vram >= 7:
        return 'yolov8m.pt', 8, 640, 0    # RTX 4060 / RTX 3060+
    elif vram >= 3.5:
        return 'yolov8s.pt', 4, 640, 0    # RTX 3050
    else:
        return 'yolov8n.pt', 4, 416, 0    # Low VRAM


def validate_dataset():
    """Check dataset exists and print counts."""
    dataset = TRAINING_DIR / 'dataset'
    for split in ['train', 'valid', 'test']:
        imgs = dataset / split / 'images'
        lbls = dataset / split / 'labels'
        if not imgs.exists() or not lbls.exists():
            print(f"Missing {split}/ folder")
            return False
        n_img = len(list(imgs.glob('*')))
        n_lbl = len(list(lbls.glob('*.txt')))
        print(f"  {split}: {n_img} images, {n_lbl} labels")
    return True


def train():
    model_name, batch, imgsz, device = pick_config()

    print(f"\nModel: {model_name} | Batch: {batch} | ImgSize: {imgsz}")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Output: {FINAL_MODEL}\n")

    model = YOLO(model_name)
    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        batch=batch,
        imgsz=imgsz,
        patience=15,
        device=device,
        workers=2,
        cache=False,
        amp=True,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        # Output
        project=str(TRAINING_DIR / 'runs'),
        name='garbage_detect',
        exist_ok=True,
        verbose=True,
    )

    # Copy best.pt → models/garbage_detect.pt
    best = TRAINING_DIR / 'runs' / 'garbage_detect' / 'weights' / 'best.pt'
    if best.exists():
        if FINAL_MODEL.exists():
            backup = FINAL_MODEL.with_suffix('.pt.backup')
            shutil.copy(FINAL_MODEL, backup)
            print(f"Old model backed up: {backup}")
        shutil.copy(best, FINAL_MODEL)
        print(f"New model saved: {FINAL_MODEL}")
    else:
        print(f"WARNING: best.pt not found at {best}")

    return results


if __name__ == "__main__":
    print("=" * 50)
    print("WASTE DETECTION MODEL TRAINING")
    print("=" * 50)

    if not validate_dataset():
        print("Dataset validation failed.")
        sys.exit(1)

    train()
    print("\nDone! Model at: models/garbage_detect.pt")
