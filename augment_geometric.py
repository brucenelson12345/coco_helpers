import os
import json
import cv2
import numpy as np
import albumentations as A
import yaml
import argparse
from tqdm import tqdm


class COCOGeometricAugmenter:
    """
    A class to apply geometric augmentations to a COCO dataset
    without altering object semantics, while correctly updating bounding boxes.
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
        aug_config = self.config['augmentation']
        self.num_copies_per_transform = aug_config.get('num_copies_per_transform', 1)
        self.transform_configs = aug_config['transforms']
        self.border_mode_str = self.config.get('border_mode', 'constant')

        # Map string to OpenCV border mode
        border_modes = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
            'wrap': cv2.BORDER_WRAP
        }
        self.border_mode = border_modes.get(self.border_mode_str.lower(), cv2.BORDER_CONSTANT)

        # Load COCO data
        self.coco_data = self.load_coco_annotations()
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.img_id_to_anns = {}
        for ann in self.annotations:
            self.img_id_to_anns.setdefault(ann['image_id'], []).append(ann)

        # Initialize transforms
        self.transforms = self.build_transforms()

    def load_coco_annotations(self):
        """Load COCO annotation JSON."""
        with open(self.original_annotation_file, 'r') as f:
            return json.load(f)

    def build_transforms(self):
        """Build list of albumentations based on config."""
        transforms = []
        for t in self.transform_configs:
            name = t['name']
            actions = t['action']
            pipeline = []

            for act in actions:
                if act == "horizontal_flip":
                    pipeline.append(A.HorizontalFlip(p=1.0))
                elif act == "vertical_flip":
                    pipeline.append(A.VerticalFlip(p=1.0))
                elif act == "rotate_90":
                    pipeline.append(A.Rotate(limit=(90, 90), p=1.0, border_mode=self.border_mode))
                elif act == "rotate_180":
                    pipeline.append(A.Rotate(limit=(180, 180), p=1.0, border_mode=self.border_mode))
                elif act == "rotate_270":
                    pipeline.append(A.Rotate(limit=(270, 270), p=1.0, border_mode=self.border_mode))
                elif isinstance(act, list) and act[0] == "rotate":
                    angle = act[1]
                    pipeline.append(A.Rotate(limit=(angle, angle), p=1.0, border_mode=self.border_mode))
                else:
                    raise ValueError(f"Unknown action: {act}")

            # Compose with bbox handling
            composed = A.Compose(pipeline, bbox_params=A.BboxParams(format='coco', label_fields=[]))
            transforms.append((name, composed))
        return transforms

    def run(self):
        """Run the augmentation pipeline."""
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

        print("Applying geometric augmentations...")
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

            # === Apply each geometric transform ===
            for name, transform in self.transforms:
                for _ in range(self.num_copies_per_transform):
                    try:
                        result = transform(image=image, bboxes=bboxes)
                        aug_image = result['image']
                        aug_bboxes = result['bboxes']

                        # Save augmented image
                        aug_file_name = f"aug_{next_img_id}.png"
                        aug_output_path = os.path.join(self.output_images_dir, aug_file_name)
                        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(aug_output_path, aug_image_bgr)

                        # New dimensions
                        new_h, new_w = aug_image.shape[:2]

                        # Add image entry
                        new_coco["images"].append({
                            "id": next_img_id,
                            "file_name": aug_file_name,
                            "width": new_w,
                            "height": new_h,
                            "license": img_info.get("license", 0),
                            "coco_url": img_info.get("coco_url", ""),
                            "date_captured": img_info.get("date_captured", ""),
                            "augmentation": name
                        })

                        # Add annotations
                        for bbox in aug_bboxes:
                            x, y, bw, bh = bbox
                            area = bw * bh
                            new_coco["annotations"].append({
                                "id": next_ann_id,
                                "image_id": next_img_id,
                                "category_id": ann["category_id"],  # from last ann in loop
                                "bbox": list(bbox),
                                "area": area,
                                "segmentation": [],
                                "iscrowd": 0
                            })
                            next_ann_id += 1

                        next_img_id += 1

                    except Exception as e:
                        print(f"Error applying {name} to {file_name}: {e}")
                        continue

        # Save updated COCO JSON
        with open(self.output_annotation_file, 'w') as f:
            json.dump(new_coco, f, indent=2)

        print(f"✅ Geometric augmentation completed!")
        print(f"📸 Output images: {self.output_images_dir}")
        print(f"📄 Output annotations: {self.output_annotation_file}")
        print(f"📊 Total images: {len(new_coco['images'])}")
        print(f"🔖 Total annotations: {len(new_coco['annotations'])}")


# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply geometric augmentations to COCO dataset.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    augmenter = COCOGeometricAugmenter(config_path=args.config)
    augmenter.run()