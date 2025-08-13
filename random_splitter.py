#!/usr/bin/env python3
import os
import json
import shutil
import random
import argparse
import yaml
from collections import defaultdict
from typing import Dict, Any

class CocoDatasetSplitter:
    """
    A class to split a COCO dataset into train, val, and test sets.
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize the splitter.

        Args:
            images_dir (str): Path to the directory containing all images.
            annotations_file (str): Path to the COCO annotations JSON file.
            output_dir (str): Path to the output directory.
            train_ratio (float): Proportion for training set.
            val_ratio (float): Proportion for validation set.
            test_ratio (float): Proportion for test set.
            seed (int): Random seed for reproducibility.
        """
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-5:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    def split(self):
        """Perform the dataset split."""
        random.seed(self.seed)

        # Read COCO annotations
        with open(self.annotations_file, 'r') as f:
            coco_data = json.load(f)

        images = coco_data['images']
        annotations = coco_data['annotations']
        categories = coco_data.get('categories', [])
        info = coco_data.get('info', {})
        licenses = coco_data.get('licenses', [])

        # Shuffle images
        random.shuffle(images)
        total_images = len(images)

        # Compute split indices
        train_end = int(self.train_ratio * total_images)
        val_end = train_end + int(self.val_ratio * total_images)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        # Map image_id to annotations
        image_id_to_anns = defaultdict(list)
        for ann in annotations:
            image_id_to_anns[ann['image_id']].append(ann)

        # Define splits
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        # Process each split
        for split_name, split_images in splits.items():
            self._process_split(
                split_name=split_name,
                images=split_images,
                image_id_to_anns=image_id_to_anns,
                categories=categories,
                info=info,
                licenses=licenses
            )

        print(f"Dataset split completed. Output saved to: {self.output_dir}")

    def _process_split(
        self,
        split_name: str,
        images: list,
        image_id_to_anns: dict,
        categories: list,
        info: dict,
        licenses: list
    ):
        """Process a single split (train/val/test)."""
        split_dir = os.path.join(self.output_dir, split_name)
        images_out_dir = os.path.join(split_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)

        new_images = []
        new_annotations = []
        old_to_new_id = {}

        for idx, img_info in enumerate(images):
            old_id = img_info['id']
            new_id = idx
            old_to_new_id[old_id] = new_id

            # Update image info
            new_img_info = img_info.copy()
            new_img_info['id'] = new_id
            new_images.append(new_img_info)

            # Copy image file
            src_image_path = os.path.join(self.images_dir, img_info['file_name'])
            dst_image_path = os.path.join(images_out_dir, img_info['file_name'])

            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
            else:
                print(f"Warning: Image not found: {src_image_path}")

            # Update and add annotations
            for ann in image_id_to_anns.get(old_id, []):
                new_ann = ann.copy()
                new_ann['image_id'] = new_id
                new_annotations.append(new_ann)

        # Create new COCO JSON
        new_coco_data = {
            'info': info,
            'licenses': licenses,
            'images': new_images,
            'annotations': new_annotations,
            'categories': categories
        }

        # Save annotations
        annotations_out_path = os.path.join(split_dir, 'annotations.json')
        with open(annotations_out_path, 'w') as f:
            json.dump(new_coco_data, f, indent=2)

        print(f"{split_name}: {len(new_images)} images, {len(new_annotations)} annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split COCO dataset into train/val/test using a YAML config.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    # Read YAML config
    with open(args.config, 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Required fields
    required_keys = ['images_dir', 'annotations_file', 'output_dir']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Optional fields with defaults
    optional_config = {
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
        'seed': 42
    }
    for key, default in optional_config.items():
        if key not in config:
            config[key] = default

    # Instantiate and run splitter
    splitter = CocoDatasetSplitter(
        images_dir=config['images_dir'],
        annotations_file=config['annotations_file'],
        output_dir=config['output_dir'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        seed=config['seed']
    )
    splitter.split()
