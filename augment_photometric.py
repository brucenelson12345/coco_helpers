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
    A class to apply photometric augmentations to a COCO dataset
    without changing bounding box locations.
    """

    def __init__(self, config_path):
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

        # Build photometric transform pipeline
        self.transform = self.build_transform(aug)

        # Load COCO data
        self.coco_data = self.load_coco_annotations()
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    def load_coco_annotations(self):
        """Load COCO annotation JSON."""
        with open(self.original_annotation_file, 'r') as f:
            return json.load(f)

    def build_transform(self, aug):
        """Build albumentations photometric pipeline from config."""
        pipeline = []

        # Random Brightness & Contrast
        if aug.get('apply_brightness_contrast', True):
            pipeline.append(
                A.RandomBrightnessContrast(
                    brightness_limit=tuple(aug['brightness_limit']),
                    contrast_limit=tuple(aug['contrast_limit']),
                    p=0.8
                )
            )

        # Hue & Saturation (HSV)
        if aug.get('apply_hsv', True):
            pipeline.append(
                A.HueSaturationValue(
                    hue_shift_limit=aug['hue_shift_limit'],
                    sat_shift_limit=aug['sat_shift_limit'],
                    val_shift_limit=aug['val_shift_limit'],
                    p=0.8
                )
            )

        # Color Jitter (Brightness, Contrast, Saturation, Hue)
        if aug.get('apply_color_jitter', True):
            cj = aug['color_jitter']
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
        if aug.get('gaussian_noise', {}).get('enabled', False):
            var_limit = tuple(aug['gaussian_noise']['var_limit'])
            pipeline.append(A.GaussNoise(var_limit=var_limit, p=0.5))

        # Blur
        if aug.get('blur', {}).get('enabled', False):
            pipeline.append(A.Blur(blur_limit=aug['blur']['blur_limit'], p=0.3))

        # Motion Blur
        if aug.get('motion_blur', {}).get('enabled', False):
            pipeline.append(A.MotionBlur(blur_limit=aug['motion_blur']['blur_limit'], p=0.2))

        # ISO Noise
        if aug.get('iso_noise', {}).get('enabled', False):
            color_shift = tuple(aug['iso_noise']['color_shift'])
            intensity = tuple(aug['iso_noise']['intensity'])
            pipeline.append(A.ISONoise(color_shift=color_shift, intensity=intensity, p=0.5))

        # Combine with bbox handling (no geometric changes)
        return A.Compose(pipeline, bbox_params=A.BboxParams(format='coco', label_fields=[]))

    def run(self):
        """Run the photometric augmentation pipeline."""
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

            # Bounding boxes in COCO format
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

            # === Apply photometric augmentations ===
            for _ in range(self.num_augmented_copies):
                try:
                    result = self.transform(image=image, bboxes=bboxes)
                    aug_image = result['image']

                    # Save augmented image
                    aug_file_name = f"aug_{next_img_id}.png"
                    aug_output_path = os.path.join(self.output_images_dir, aug_file_name)
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(aug_output_path, aug_image_bgr)

                    # Add image entry (same size as original)
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

                    # Copy annotations (same bbox, no change)
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
                    print(f"Error applying photometric augmentation to {file_name}: {e}")
                    continue

        # Save updated COCO JSON
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