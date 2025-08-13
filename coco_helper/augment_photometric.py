#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import albumentations as A
import yaml
import argparse
from tqdm import tqdm


class COCOPhotometricAugmenter:
    """
    Applies photometric augmentations (e.g., brightness, contrast, noise, blur) to a COCO dataset
    without altering bounding box coordinates.

    The class uses default augmentation settings that can be overridden via config,
    and only requires path arguments in __init__.
    """

    def __init__(
        self,
        original_images_dir,
        original_annotation_file,
        output_images_dir,
        output_annotation_file
    ):
        """
        Initializes the augmenter with required paths and default photometric settings.

        Args:
            original_images_dir (str): Directory containing original .png images.
            original_annotation_file (str): Path to input COCO JSON annotations.
            output_images_dir (str): Directory to save original + augmented images.
            output_annotation_file (str): Path to save updated COCO JSON.
        """
        self.original_images_dir = original_images_dir
        self.original_annotation_file = original_annotation_file
        self.output_images_dir = output_images_dir
        self.output_annotation_file = output_annotation_file

        # Default photometric augmentation settings
        self.num_augmented_copies = 5
        self.brightness_limit = [0.1, 0.3]
        self.contrast_limit = [0.1, 0.3]
        self.hue_shift_limit = 20
        self.sat_shift_limit = 30
        self.val_shift_limit = 20
        self.color_jitter_factors = [0.2, 0.2, 0.2, 0.1]  # [brightness, contrast, saturation, hue]
        self.apply_color_jitter = True
        self.apply_brightness_contrast = True
        self.apply_hsv = True

        # Noise and blur settings
        self.gaussian_noise_enabled = True
        self.gaussian_noise_var_limit = [10.0, 50.0]

        self.blur_enabled = True
        self.blur_limit = 3

        self.motion_blur_enabled = True
        self.motion_blur_limit = 5

        self.iso_noise_enabled = True
        self.iso_noise_color_shift = [0.01, 0.05]
        self.iso_noise_intensity = [0.1, 0.5]

        # Will be set in run()
        self.images = {}
        self.annotations = []
        self.img_id_to_anns = {}
        self.coco_data = {}

    def load_coco_annotations(self):
        """
        Loads the original COCO annotation JSON file.

        Returns:
            dict: Parsed COCO dataset (images, annotations, categories, etc.).
        """
        with open(self.original_annotation_file, 'r') as f:
            return json.load(f)

    def build_transform(self):
        """
        Constructs the photometric transformation pipeline using current settings.

        Returns:
            A.Compose: Albumentations pipeline with bounding box support.
        """
        pipeline = []

        # Random Brightness & Contrast
        if self.apply_brightness_contrast:
            pipeline.append(
                A.RandomBrightnessContrast(
                    brightness_limit=tuple(self.brightness_limit),
                    contrast_limit=tuple(self.contrast_limit),
                    p=0.8
                )
            )

        # Hue & Saturation (HSV)
        if self.apply_hsv:
            pipeline.append(
                A.HueSaturationValue(
                    hue_shift_limit=self.hue_shift_limit,
                    sat_shift_limit=self.sat_shift_limit,
                    val_shift_limit=self.val_shift_limit,
                    p=0.8
                )
            )

        # Color Jitter
        if self.apply_color_jitter:
            cj = self.color_jitter_factors
            pipeline.append(
                A.ColorJitter(
                    brightness=cj[0],
                    contrast=cj[1],
                    saturation=cj[2],
                    hue=cj[3],
                    p=0.8
                )
            )

        # Gaussian Noise
        if self.gaussian_noise_enabled:
            pipeline.append(
                A.GaussNoise(var_limit=tuple(self.gaussian_noise_var_limit), p=0.5)
            )

        # Blur
        if self.blur_enabled:
            pipeline.append(A.Blur(blur_limit=self.blur_limit, p=0.3))

        # Motion Blur
        if self.motion_blur_enabled:
            pipeline.append(A.MotionBlur(blur_limit=self.motion_blur_limit, p=0.2))

        # ISO Noise
        if self.iso_noise_enabled:
            pipeline.append(
                A.ISONoise(
                    color_shift=tuple(self.iso_noise_color_shift),
                    intensity=tuple(self.iso_noise_intensity),
                    p=0.5
                )
            )

        return A.Compose(pipeline, bbox_params=A.BboxParams(format='coco', label_fields=[]))

    def run(self):
        """
        Executes the photometric augmentation pipeline:
        - Loads COCO data
        - Copies original images
        - Applies photometric transforms
        - Saves augmented images
        - Generates updated COCO JSON with new image entries

        Outputs images and annotations to the specified output directories.
        """
        # Load COCO data
        self.coco_data = self.load_coco_annotations()
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

        # Create output directory
        os.makedirs(self.output_images_dir, exist_ok=True)

        # New COCO structure
        new_coco = {
            "images": [],
            "annotations": [],
            "categories": self.coco_data["categories"],
            "info": self.coco_data.get("info", {}),
            "licenses": self.coco_data.get("licenses", [])
        }

        # Track new IDs
        next_img_id = max(self.images.keys()) + 1 if self.images else 1
        next_ann_id = max([ann['id'] for ann in self.annotations], default=0) + 1

        # Build transform pipeline (after settings are loaded)
        transform = self.build_transform()

        print("Applying photometric augmentations...")
        for img_id, img_info in tqdm(self.images.items()):
            file_name = img_info['file_name']
            img_path = os.path.join(self.original_images_dir, file_name)

            # Fallback to basename
            if not os.path.exists(img_path):
                img_path = os.path.join(self.original_images_dir, os.path.basename(file_name))
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            bboxes = [ann['bbox'] for ann in self.img_id_to_anns.get(img_id, [])]

            # === Save original image ===
            orig_file_name = f"original_{file_name}"
            orig_output_path = os.path.join(self.output_images_dir, orig_file_name)
            if not os.path.exists(orig_output_path):
                cv2.imwrite(orig_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            new_coco["images"].append({
                "id": img_id,
                "file_name": orig_file_name,
                "width": w,
                "height": h,
                "license": img_info.get("license", 0),
                "coco_url": img_info.get("coco_url", ""),
                "date_captured": img_info.get("date_captured", "")
            })

            for ann in self.img_id_to_anns.get(img_id, []):
                new_coco["annotations"].append({
                    "id": ann["id"],
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "segmentation": ann.get("segmentation", []),
                    "iscrowd": ann["iscrowd"]
                })

            # === Apply augmentations ===
            for _ in range(self.num_augmented_copies):
                try:
                    result = transform(image=image, bboxes=bboxes)
                    aug_image = result['image']

                    # Save augmented image
                    aug_file_name = f"aug_{next_img_id}.png"
                    aug_output_path = os.path.join(self.output_images_dir, aug_file_name)
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(aug_output_path, aug_image_bgr)

                    # Add new image entry
                    new_coco["images"].append({
                        "id": next_img_id,
                        "file_name": aug_file_name,
                        "width": w,
                        "height": h,
                        "license": img_info.get("license", 0),
                        "coco_url": img_info.get("coco_url", ""),
                        "date_captured": img_info.get("date_captured", ""),
                        "augmentation": "photometric"
                    })

                    # Copy annotations (bbox unchanged)
                    for ann in self.img_id_to_anns.get(img_id, []):
                        new_coco["annotations"].append({
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                            "segmentation": ann.get("segmentation", []),
                            "iscrowd": ann["iscrowd"]
                        })
                        next_ann_id += 1

                    next_img_id += 1

                except Exception as e:
                    print(f"Error applying augmentation to {file_name}: {e}")
                    continue

        # Save updated COCO JSON
        with open(self.output_annotation_file, 'w') as f:
            json.dump(new_coco, f, indent=2)

        print(f"Photometric augmentation completed!")
        print(f"Output images: {self.output_images_dir}")
        print(f"Output annotations: {self.output_annotation_file}")
        print(f"Total images: {len(new_coco['images'])}")
        print(f"Total annotations: {len(new_coco['annotations'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply photometric augmentations to COCO dataset.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file (.yaml or .yml)')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.lower().endswith(('.yaml', '.yml')):
        raise ValueError("Config file must be a YAML file with extension .yaml or .yml")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract paths (required)
    paths = config.get('paths', {})
    required_paths = [
        'original_images_dir',
        'original_annotation_file',
        'output_images_dir',
        'output_annotation_file'
    ]
    for key in required_paths:
        if key not in paths:
            raise KeyError(f"Missing required path in config: paths.{key}")

    # Initialize the augmenter with only path arguments
    augmenter = COCOPhotometricAugmenter(
        original_images_dir=paths['original_images_dir'],
        original_annotation_file=paths['original_annotation_file'],
        output_images_dir=paths['output_images_dir'],
        output_annotation_file=paths['output_annotation_file']
    )

    # Override default settings if provided in config
    aug_config = config.get('augmentation', {})

    # Simple scalar values
    if 'num_augmented_copies' in aug_config:
        augmenter.num_augmented_copies = int(aug_config['num_augmented_copies'])

    # Lists: brightness, contrast, etc.
    if 'brightness_limit' in aug_config:
        augmenter.brightness_limit = aug_config['brightness_limit']
    if 'contrast_limit' in aug_config:
        augmenter.contrast_limit = aug_config['contrast_limit']
    if 'hue_shift_limit' in aug_config:
        augmenter.hue_shift_limit = int(aug_config['hue_shift_limit'])
    if 'sat_shift_limit' in aug_config:
        augmenter.sat_shift_limit = int(aug_config['sat_shift_limit'])
    if 'val_shift_limit' in aug_config:
        augmenter.val_shift_limit = int(aug_config['val_shift_limit'])
    if 'color_jitter' in aug_config:
        augmenter.color_jitter_factors = aug_config['color_jitter']
    if 'apply_color_jitter' in aug_config:
        augmenter.apply_color_jitter = bool(aug_config['apply_color_jitter'])
    if 'apply_brightness_contrast' in aug_config:
        augmenter.apply_brightness_contrast = bool(aug_config['apply_brightness_contrast'])
    if 'apply_hsv' in aug_config:
        augmenter.apply_hsv = bool(aug_config['apply_hsv'])

    # Noise and blur
    if 'gaussian_noise' in aug_config:
        gn = aug_config['gaussian_noise']
        augmenter.gaussian_noise_enabled = bool(gn.get('enabled', True))
        if 'var_limit' in gn:
            augmenter.gaussian_noise_var_limit = gn['var_limit']

    if 'blur' in aug_config:
        b = aug_config['blur']
        augmenter.blur_enabled = bool(b.get('enabled', True))
        if 'blur_limit' in b:
            augmenter.blur_limit = int(b['blur_limit'])

    if 'motion_blur' in aug_config:
        mb = aug_config['motion_blur']
        augmenter.motion_blur_enabled = bool(mb.get('enabled', True))
        if 'blur_limit' in mb:
            augmenter.motion_blur_limit = int(mb['blur_limit'])

    if 'iso_noise' in aug_config:
        iso = aug_config['iso_noise']
        augmenter.iso_noise_enabled = bool(iso.get('enabled', True))
        if 'color_shift' in iso:
            augmenter.iso_noise_color_shift = iso['color_shift']
        if 'intensity' in iso:
            augmenter.iso_noise_intensity = iso['intensity']

    # Run augmentation
    augmenter.run()
