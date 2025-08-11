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
    without altering bounding box coordinates, and generates an updated COCO annotation JSON.

    The class reads configuration from a YAML file and supports flexible augmentation settings.
    Original images are preserved, and augmented versions are saved alongside updated annotations.

    Init Parameters:
        config_path (str): Path to the YAML configuration file containing paths and augmentation settings.
    """

        def __init__(self, config_path):
        """
        Initializes the COCOPhotometricAugmenter with settings from a YAML configuration file.

        Loads paths, augmentation parameters, and builds the photometric transformation pipeline.
        Also loads the original COCO dataset annotations into memory for processing.

        Args:
            config_path (str): Path to the YAML configuration file. The file must contain:
                - paths (dict): Contains file/directory paths:
                    - original_images_dir (str): Directory with original .png images.
                    - original_annotation_file (str): Path to the input COCO JSON annotations.
                    - output_images_dir (str): Directory to save original + augmented images.
                    - output_annotation_file (str): Path to save the updated COCO JSON.
                - augmentation (dict): Photometric augmentation settings:
                    - num_augmented_copies (int): Number of augmented versions to generate per image.
                    - brightness_limit (list[float]): Min and max relative brightness change, e.g., [0.1, 0.3].
                    - contrast_limit (list[float]): Min and max relative contrast change.
                    - hue_shift_limit (int): Max degree shift for hue (e.g., 20).
                    - sat_shift_limit (int): Max percentage shift for saturation.
                    - val_shift_limit (int): Max percentage shift for value (brightness in HSV).
                    - color_jitter (list[float]): Factors for [brightness, contrast, saturation, hue] jitter.
                    - apply_color_jitter (bool): Whether to apply color jitter augmentation.
                    - apply_brightness_contrast (bool): Whether to apply brightness/contrast adjustment.
                    - apply_hsv (bool): Whether to apply HSV-based hue/saturation/value shifts.
                    - gaussian_noise (dict):
                        - enabled (bool): If True, applies Gaussian noise.
                        - var_limit (list[float]): Min and max variance for noise.
                    - blur (dict):
                        - enabled (bool): If True, applies regular blur.
                        - blur_limit (int): Maximum kernel size (e.g., 3).
                    - motion_blur (dict):
                        - enabled (bool): If True, applies motion blur.
                        - blur_limit (int): Maximum kernel size.
                    - iso_noise (dict):
                        - enabled (bool): If True, applies simulated ISO camera noise.
                        - color_shift (list[float]): Min and max color shift range.
                        - intensity (list[float]): Min and max noise intensity.
                - border_mode (str, optional): How to handle borders during transforms (not used in photometric).
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Paths
        paths = self.config['paths']
        self.original_images_dir = paths['original_images_dir']
        self.original_annotation_file = paths['original_annotation_file']
        self.output_images_dir = paths['output_images_dir']
        self.output_annotation_file = paths['output_annotation_file']

        # Augmentation settings
        aug = self.config['augmentation']
        self.num_augmented_copies = aug['num_augmented_copies']

        # Build photometric transform pipeline based on config
        self.transform = self.build_transform(aug)

        # Load COCO data
        self.coco_data = self.load_coco_annotations()
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    def load_coco_annotations(self):
        """
        Loads and returns the original COCO annotation JSON file.

        Returns:
            dict: The parsed COCO dataset (images, annotations, categories, etc.).
        """
        with open(self.original_annotation_file, 'r') as f:
            return json.load(f)

    def build_transform(self, aug):
        """
        Constructs an albumentations photometric transformation pipeline based on the configuration.

        Args:
            aug (dict): The 'augmentation' section of the config, specifying which transforms to apply.

        Returns:
            A.Compose: A composed albumentations transform pipeline with bounding box support.
        """
        pipeline = []

        # Random Brightness & Contrast
        if aug.get('apply_brightness_contrast', True):
            pipeline.append(
                A.RandomBrightnessContrast(
                    brightness_limit=tuple(aug['brightness_limit']),  # Min/max relative brightness change
                    contrast_limit=tuple(aug['contrast_limit']),      # Min/max relative contrast change
                    p=0.8  # Probability of applying this transform
                )
            )

        # Hue & Saturation (HSV)
        if aug.get('apply_hsv', True):
            pipeline.append(
                A.HueSaturationValue(
                    hue_shift_limit=aug['hue_shift_limit'],    # Max degrees to shift hue
                    sat_shift_limit=aug['sat_shift_limit'],    # Max percentage to shift saturation
                    val_shift_limit=aug['val_shift_limit'],    # Max percentage to shift value (brightness)
                    p=0.8
                )
            )

        # Color Jitter (Brightness, Contrast, Saturation, Hue)
        if aug.get('apply_color_jitter', True):
            cj = aug['color_jitter']
            pipeline.append(
                A.ColorJitter(
                    brightness=cj[0],  # Brightness factor range
                    contrast=cj[1],    # Contrast factor range
                    saturation=cj[2],  # Saturation factor range
                    hue=cj[3],         # Hue factor range
                    p=0.8
                )
            )

        # Gaussian Noise
        if aug.get('gaussian_noise', {}).get('enabled', False):
            var_limit = tuple(aug['gaussian_noise']['var_limit'])  # Variance range for noise
            pipeline.append(A.GaussNoise(var_limit=var_limit, p=0.5))

        # Blur
        if aug.get('blur', {}).get('enabled', False):
            pipeline.append(A.Blur(blur_limit=aug['blur']['blur_limit'], p=0.3))  # Max kernel size

        # Motion Blur
        if aug.get('motion_blur', {}).get('enabled', False):
            pipeline.append(A.MotionBlur(blur_limit=aug['motion_blur']['blur_limit'], p=0.2))

        # ISO Noise
        if aug.get('iso_noise', {}).get('enabled', False):
            color_shift = tuple(aug['iso_noise']['color_shift'])  # Range of color shift
            intensity = tuple(aug['iso_noise']['intensity'])      # Range of noise intensity
            pipeline.append(A.ISONoise(color_shift=color_shift, intensity=intensity, p=0.5))

        # Combine all transforms with COCO-format bounding box handling
        return A.Compose(pipeline, bbox_params=A.BboxParams(format='coco', label_fields=[]))

    def run(self):
        """
        Executes the full photometric augmentation pipeline:
        - Copies original images
        - Applies photometric transforms
        - Saves augmented images
        - Updates COCO JSON with new image entries and unchanged annotations

        This method creates the output directory, processes each image, and writes the final JSON.
        """
        os.makedirs(self.output_images_dir, exist_ok=True)

        # New COCO structure to hold original + augmented data
        new_coco = {
            "images": [],
            "annotations": [],
            "categories": self.coco_data["categories"],
            "info": self.coco_data.get("info", {}),
            "licenses": self.coco_data.get("licenses", [])
        }

        # Track new unique IDs for images and annotations
        next_img_id = max(self.images.keys()) + 1 if self.images else 1
        next_ann_id = max([ann['id'] for ann in self.annotations], default=0) + 1

        print("Applying photometric augmentations...")
        for img_id, img_info in tqdm(self.images.items()):
            file_name = img_info['file_name']  # Original filename
            img_path = os.path.join(self.original_images_dir, file_name)

            # Fallback: try using just the basename
            if not os.path.exists(img_path):
                img_path = os.path.join(self.original_images_dir, os.path.basename(file_name))
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            # Read image in RGB format
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]  # Original height and width

            # Get bounding boxes in COCO format [x, y, width, height]
            bboxes = [ann['bbox'] for ann in self.img_id_to_anns.get(img_id, [])]

            # === Save original image ===
            orig_file_name = f"original_{file_name}"
            orig_output_path = os.path.join(self.output_images_dir, orig_file_name)
            if not os.path.exists(orig_output_path):
                cv2.imwrite(orig_output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Add original image to new COCO dataset
            new_coco["images"].append({
                "id": img_id,
                "file_name": orig_file_name,
                "width": w,
                "height": h,
                "license": img_info.get("license", 0),
                "coco_url": img_info.get("coco_url", ""),
                "date_captured": img_info.get("date_captured", "")
            })

            # Copy all annotations for this image (bbox unchanged)
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

            # === Apply photometric augmentations ===
            for _ in range(self.num_augmented_copies):
                try:
                    # Apply transforms (bboxes are passed but not modified)
                    result = self.transform(image=image, bboxes=bboxes)
                    aug_image = result['image']  # Augmented image

                    # Save augmented image
                    aug_file_name = f"aug_{next_img_id}.png"
                    aug_output_path = os.path.join(self.output_images_dir, aug_file_name)
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(aug_output_path, aug_image_bgr)

                    # Add new image entry (same dimensions as original)
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

                    # Copy annotations (same bounding boxes)
                    for ann in self.img_id_to_anns.get(img_id, []):
                        new_coco["annotations"].append({
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],  # Unchanged
                            "area": ann["area"],  # Area remains same
                            "segmentation": ann.get("segmentation", []),
                            "iscrowd": ann["iscrowd"]
                        })
                        next_ann_id += 1

                    next_img_id += 1

                except Exception as e:
                    print(f"Error applying photometric augmentation to {file_name}: {e}")
                    continue

        # Save updated COCO annotation file
        with open(self.output_annotation_file, 'w') as f:
            json.dump(new_coco, f, indent=2)

        print(f"Photometric augmentation completed!")
        print(f"Output images: {self.output_images_dir}")
        print(f"Output annotations: {self.output_annotation_file}")
        print(f"Total images: {len(new_coco['images'])}")
        print(f"Total annotations: {len(new_coco['annotations'])}")


# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply photometric augmentations to COCO dataset.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    augmenter = COCOPhotometricAugmenter(config_path=args.config)
    augmenter.run()