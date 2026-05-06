#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = CODE_ROOT / "datasets" / "vision" / "yolo_seg"
DEFAULT_PROJECT_ROOT = CODE_ROOT / "artifacts" / "vision" / "runs"
DEFAULT_DEPLOY_WEIGHTS = CODE_ROOT / "datasets" / "vision" / "models" / "best.pt"
DEFAULT_BASE_MODEL = str(DEFAULT_DEPLOY_WEIGHTS) if DEFAULT_DEPLOY_WEIGHTS.exists() else "yolov8n-seg.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(raw: str) -> dict[int, str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--names must contain at least one class name")
    return {index: name for index, name in enumerate(names)}


def write_data_yaml(path: Path, dataset_root: Path, names: dict[int, str]) -> None:
    yaml_lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id, class_name in names.items():
        escaped = class_name.replace('"', '\\"')
        yaml_lines.append(f'  {class_id}: "{escaped}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def init_dataset_layout(dataset_root: Path, names: dict[int, str]) -> Path:
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        write_data_yaml(data_yaml, dataset_root, names)
    return data_yaml


def iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_label_file(path: Path, *, allow_box_labels: bool) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return errors
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except Exception:
            errors.append(f"{path}:{line_number}: label values must be numeric")
            continue

        if len(parts) == 5 and allow_box_labels:
            pass
        elif len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            errors.append(
                f"{path}:{line_number}: segmentation labels need class_id plus polygon coordinates"
            )

        out_of_range = [value for value in coords if value < -0.001 or value > 1.001]
        if out_of_range:
            errors.append(f"{path}:{line_number}: normalized coordinates should be in [0, 1]")
    return errors


def validate_dataset(
    dataset_root: Path,
    *,
    allow_missing_labels: bool,
    allow_empty_val: bool,
    allow_box_labels: bool,
) -> dict[str, object]:
    split_stats: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for split in ("train", "val", "test"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        images = iter_images(images_dir)
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        split_stats[split] = {"images": len(images), "labels": len(label_files)}

        if split == "train" and not images:
            errors.append(f"Missing training images: {images_dir}")
        if split == "val" and not images and not allow_empty_val:
            errors.append(f"Missing validation images: {images_dir}")

        if not labels_dir.exists():
            if images and not allow_missing_labels:
                errors.append(f"Missing labels directory: {labels_dir}")
            continue

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                message = f"Missing label for image: {image_path}"
                if allow_missing_labels:
                    warnings.append(message)
                else:
                    errors.append(message)
                continue
            errors.extend(validate_label_file(label_path, allow_box_labels=allow_box_labels))

    return {"splits": split_stats, "errors": errors, "warnings": warnings}


def resolve_device(raw: str) -> str:
    requested = str(raw).strip().lower()
    if requested != "auto":
        return str(raw).strip()
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_yolo_class():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install the hybrid vision environment first."
        ) from exc
    return YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune the current wooden-block YOLO segmentation profile."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_ROOT, help="YOLO segment dataset root")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Existing Ultralytics data.yaml")
    parser.add_argument("--names", type=str, default="upside of cube", help="Comma-separated class names")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Starting .pt model or YOLO name")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT_ROOT, help="Training output root")
    parser.add_argument("--run-name", type=str, default="", help="Ultralytics run name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for fewer multiprocessing issues")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--init-only", action="store_true", help="Create dataset folders and data.yaml, then exit")
    parser.add_argument("--check-only", action="store_true", help="Validate dataset layout, then exit")
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--allow-empty-val", action="store_true")
    parser.add_argument("--allow-box-labels", action="store_true", help="Allow 5-column box labels as a fallback")
    parser.add_argument("--deploy-to", type=Path, default=None, help="Copy resulting best.pt to this path")
    parser.add_argument("--exist-ok", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset.expanduser().resolve()
    names = parse_names(args.names)

    data_yaml = args.data_yaml.expanduser().resolve() if args.data_yaml else dataset_root / "data.yaml"
    if args.init_only:
        data_yaml = init_dataset_layout(dataset_root, names)
        print(f"Initialized dataset layout: {dataset_root}")
        print(f"Data YAML: {data_yaml}")
        return 0

    if not data_yaml.exists():
        data_yaml = init_dataset_layout(dataset_root, names)
        print(f"Created missing data.yaml: {data_yaml}")

    report = validate_dataset(
        dataset_root,
        allow_missing_labels=bool(args.allow_missing_labels),
        allow_empty_val=bool(args.allow_empty_val),
        allow_box_labels=bool(args.allow_box_labels),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        print("Dataset validation failed. Fix the errors above, or use --init-only to create the layout.", file=sys.stderr)
        return 2
    if args.check_only:
        return 0

    device = resolve_device(args.device)
    run_name = args.run_name.strip() or f"block_yolo_seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    YOLO = load_yolo_class()
    model = YOLO(str(args.base_model))
    results = model.train(
        data=str(data_yaml),
        task="segment",
        epochs=max(1, int(args.epochs)),
        imgsz=max(64, int(args.imgsz)),
        batch=int(args.batch),
        device=device,
        workers=max(0, int(args.workers)),
        patience=max(0, int(args.patience)),
        cache=bool(args.cache),
        project=str(args.project.expanduser().resolve()),
        name=run_name,
        exist_ok=bool(args.exist_ok),
    )

    save_dir = Path(getattr(results, "save_dir", args.project / run_name))
    best_weights = save_dir / "weights" / "best.pt"
    print(f"Training finished: {save_dir}")
    print(f"Best weights: {best_weights}")

    if args.deploy_to is not None:
        deploy_to = args.deploy_to.expanduser().resolve()
        if not best_weights.exists():
            raise FileNotFoundError(f"Cannot deploy because best.pt was not found: {best_weights}")
        deploy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, deploy_to)
        print(f"Deployed best.pt to: {deploy_to}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
