#!/usr/bin/env python
# train_yolov8.py

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
from ultralytics import YOLO


class YOLOv8Trainer:
    """
    A class to handle YOLOv8 model training given a dataset in YOLO format.
    Automatically infers num_classes and class_names from train/annotations.json (COCO).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        Automatically infers num_classes and class_names from annotations.json if possible.
        """
        self.config = config
        self.dataset_root = Path(self.config['dataset_path']).resolve()

        # Optional parameters with defaults
        self.model_size = self.config.get('model_size', 'n')
        self.epochs = self.config.get('epochs', 100)
        self.imgsz = self.config.get('imgsz', 640)
        self.batch = self.config.get('batch', 16)
        self.workers = self.config.get('workers', 8)
        self.project = self.config.get('project', 'runs/train')
        self.name = self.config.get('name', f'yolov8{self.model_size}_train')
        self.exist_ok = self.config.get('exist_ok', True)
        self.pretrained = self.config.get('pretrained')
        self.evaluate_on_test_flag = self.config.get('evaluate_on_test', False)

        # Infer classes from JSON or use config
        self.num_classes, self.class_names = self._infer_classes()

        self.data_yaml_path = None
        self.model = None

    def _infer_classes(self) -> tuple[int, list]:
        """
        Infer num_classes and class_names from train/annotations.json (COCO format).
        Falls back to config values if not found or override is provided.
        """
        json_path = self.dataset_root / 'train' / 'annotations.json'
        config_num_classes = self.config.get('num_classes')
        config_class_names = self.config.get('class_names')

        # If user provided both in config, use them
        if config_num_classes is not None and config_class_names is not None:
            print(f"Using user-provided num_classes={config_num_classes} and class_names.")
            if len(config_class_names) != config_num_classes:
                raise ValueError("len(class_names) must equal num_classes")
            return config_num_classes, config_class_names

        # If only num_classes provided but not names, generate dummy names
        if config_num_classes is not None:
            print(f"Using user-provided num_classes={config_num_classes}, auto-generating class names.")
            class_names = [f"class{i}" for i in range(config_num_classes)]
            return config_num_classes, class_names

        # Try to infer from annotations.json
        if not json_path.exists():
            raise FileNotFoundError(
                f"annotations.json not found at {json_path}\n"
                "Please either:\n"
                "  1. Place COCO annotations.json in train/ directory\n"
                "  2. Specify 'num_classes' in config"
            )

        print(f"Inferring classes from {json_path}...")
        import json
        with open(json_path, 'r') as f:
            coco = json.load(f)

        categories = coco.get('categories', [])
        if not categories:
            raise ValueError("No categories found in annotations.json")

        # Sort by id to preserve order
        sorted_cats = sorted(categories, key=lambda x: x['id'])
        class_names = [cat['name'] for cat in sorted_cats]

        # Ensure contiguous IDs starting at 0
        ids = [cat['id'] for cat in sorted_cats]
        if min(ids) < 0 or max(ids) >= len(ids) or len(set(ids)) != len(ids):
            print("Category IDs are not contiguous. Remapping to 0-based index.")
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(ids))}
            class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
        else:
            class_names = [cat['name'] for cat in sorted_cats]

        num_classes = len(class_names)
        print(f"Inferred {num_classes} classes: {class_names}")
        return num_classes, class_names

    def verify_structure(self) -> None:
        """Verify that train/val/test directories and required subdirs exist."""
        subsets = ['train', 'val', 'test']
        for subset in subsets:
            subset_path = self.dataset_root / subset
            images_path = subset_path / 'images'
            labels_path = subset_path / 'labels'

            if not images_path.exists():
                raise FileNotFoundError(f"Images directory missing: {images_path}")
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels directory missing: {labels_path}")

            image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg")) + \
                          list(images_path.glob("*.jpeg")) + list(images_path.glob("*.bmp"))
            label_files = list(labels_path.glob("*.txt"))

            if len(image_files) == 0:
                raise FileNotFoundError(f"No images found in {images_path}")
            if len(label_files) == 0:
                raise FileNotFoundError(f"No label .txt files found in {labels_path}")

        print("Dataset structure verified.")

    def create_data_yaml(self) -> Path:
        """Generate data.yaml for YOLOv8."""
        data_yaml = {
            'train': str(self.dataset_root / 'train' / 'images'),
            'val': str(self.dataset_root / 'val' / 'images'),
            'test': str(self.dataset_root / 'test' / 'images'),
            'nc': self.num_classes,
            'names': self.class_names
        }

        yaml_path = self.dataset_root / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=None, sort_keys=False)

        print(f"data.yaml saved to {yaml_path}")
        self.data_yaml_path = yaml_path
        return yaml_path

    def train(self) -> Any:
        """Train the YOLOv8 model."""
        print("Verifying dataset structure...")
        self.verify_structure()

        print("Creating data.yaml...")
        self.create_data_yaml()

        # Load model: custom pretrained or default
        if self.pretrained:
            pretrained_path = Path(self.pretrained)
            if not pretrained_path.exists():
                raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
            model_path = str(pretrained_path)
            print(f"Loading custom pretrained model: {model_path}")
        else:
            model_path = f"yolov8{self.model_size}.pt"
            print(f"Loading default YOLOv8 model: {model_path}")

        self.model = YOLO(model_path)

        print("Starting training...")
        results = self.model.train(
            data=str(self.data_yaml_path),
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            workers=self.workers,
            project=self.project,
            name=self.name,
            exist_ok=self.exist_ok
        )

        print("Training completed.")
        return results

    def validate_on_test(self) -> Any:
        """Run final evaluation on the test set."""
        if not self.model:
            raise RuntimeError("Model not trained or loaded. Call train() first.")
        print("Evaluating on test set...")
        results = self.model.val(data=str(self.data_yaml_path), split='test')
        print(f"Test mAP50-95: {results.box.map:.4f}")
        return results


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Instantiate and run trainer
    trainer = YOLOv8Trainer(config=config)
    results = trainer.train()

    # Optionally evaluate on test set
    if trainer.config.get('evaluate_on_test', False):
        trainer.validate_on_test()
