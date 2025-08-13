#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
import yaml
import argparse


class COCOAugmenter:
    """
    COCOAugmenter class for geometric augmentation of COCO-formatted datasets.
    Applies rotations, flips, and combinations, updates image files and annotations accordingly.
    """

    def __init__(
        self,
        images_dir,
        annotations_file,
        output_images_dir,
        output_annotations_file,
        # Optional augmentation settings with defaults
        rotations=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotate90_plus_vertical_flip=True,
        rotate90_plus_horizontal_flip=True,
        rotate90_plus_hv_flip=True,
        save_original=True,
        image_extensions='.png'
    ):
        """
        Initialize the COCOAugmenter with required paths and optional augmentation settings.

        Args:
            images_dir (str): Path to the folder containing original images.
            annotations_file (str): Path to the COCO annotations JSON file.
            output_images_dir (str): Directory where augmented images will be saved.
            output_annotations_file (str): Path to save the updated COCO JSON file.

            rotations (list of int, optional): List of rotation angles (e.g., [90, 180, 270]).
                Defaults to None (no rotations).
            horizontal_flip (bool, optional): Apply left-right flip. Default: False.
            vertical_flip (bool, optional): Apply top-bottom flip. Default: False.
            rotate90_plus_vertical_flip (bool, optional): Apply 90 rotation + vertical flip. Default: False.
            rotate90_plus_horizontal_flip (bool, optional): Apply 90 rotation + horizontal flip. Default: False.
            rotate90_plus_hv_flip (bool, optional): Apply 90 rotation + both flips. Default: False.
            save_original (bool, optional): If True, saves the original image as '_orig'. Default: True.
            image_extensions (list of str, optional): List of supported image extensions.
                Defaults to ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'].
        """
        # Required paths
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.output_images_dir = output_images_dir
        self.output_annotations_file = output_annotations_file

        # Optional settings with defaults
        self.rotations = rotations or []
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate90_plus_vertical_flip = rotate90_plus_vertical_flip
        self.rotate90_plus_horizontal_flip = rotate90_plus_horizontal_flip
        self.rotate90_plus_hv_flip = rotate90_plus_hv_flip
        self.save_original = save_original
        self.image_extensions = image_extensions or ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        self.image_extensions = [ext.lower() for ext in self.image_extensions]

        # Create output directory
        os.makedirs(self.output_images_dir, exist_ok=True)

        # Load COCO dataset
        if not os.path.exists(self.annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']

        # Map annotations by image ID
        self.anns_by_image = {}
        for ann in self.annotations:
            self.anns_by_image.setdefault(ann['image_id'], []).append(ann)

        # Output containers
        self.output_images = []
        self.output_annotations = []
        self.output_categories = self.categories

        # Build transforms
        self.transforms = {}
        self.suffix_map = {}
        self._build_transforms()

    def _build_transforms(self):
        """Builds transformation functions and suffix mappings based on enabled options."""
        # Always include original
        self.transforms['orig'] = lambda img: img
        self.suffix_map['orig'] = '_orig'

        # Rotations: 90, 180, 270 degrees
        for angle in self.rotations:
            if angle not in (90, 180, 270):
                print(f"Skipping unsupported rotation angle: {angle}")
                continue
            k = angle // 90
            key = f'r{angle}'
            self.transforms[key] = lambda img, k=k: np.rot90(img, k)
            self.suffix_map[key] = f'_r{angle}'

        # Flips
        if self.horizontal_flip:
            self.transforms['hflip'] = lambda img: cv2.flip(img, 1)
            self.suffix_map['hflip'] = '_hflip'

        if self.vertical_flip:
            self.transforms['vflip'] = lambda img: cv2.flip(img, 0)
            self.suffix_map['vflip'] = '_vflip'

        if self.rotate90_plus_vertical_flip:
            self.transforms['r90-vflip'] = lambda img: cv2.flip(np.rot90(img, 1), 0)
            self.suffix_map['r90-vflip'] = '_r90-vflip'

        if self.rotate90_plus_horizontal_flip:
            self.transforms['r90-hflip'] = lambda img: cv2.flip(np.rot90(img, 1), 1)
            self.suffix_map['r90-hflip'] = '_r90-hflip'

        if self.rotate90_plus_hv_flip:
            self.transforms['r90-hvflip'] = lambda img: cv2.flip(np.rot90(img, 1), -1)
            self.suffix_map['r90-hvflip'] = '_r90-hvflip'

    @staticmethod
    def rotate_points(points, angle, cx, cy):
        """Rotates a list of (x, y) points around center (cx, cy) by angle degrees."""
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        points = np.array(points).reshape(-1, 2)
        rotated = (points - [cx, cy]) @ np.array([[cos_a, sin_a], [-sin_a, cos_a]]) + [cx, cy]
        return rotated.reshape(-1).tolist()

    @staticmethod
    def flip_points(points, axis, img_w, img_h):
        """Flips points horizontally (axis=0) or vertically (axis=1)."""
        points = np.array(points).reshape(-1, 2)
        if axis == 0:
            points[:, 0] = img_w - points[:, 0]
        elif axis == 1:
            points[:, 1] = img_h - points[:, 1]
        return points.reshape(-1).tolist()

    def rotate_bbox(self, bbox, angle, img_w, img_h):
        """Computes new axis-aligned bounding box after rotation."""
        x, y, w, h = bbox
        corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        center = (img_w / 2, img_h / 2)
        rotated_corners = self.rotate_points(corners, angle, center[0], center[1])
        rotated_corners = np.array(rotated_corners).reshape(-1, 2)
        x_min = rotated_corners[:, 0].min()
        y_min = rotated_corners[:, 1].min()
        x_max = rotated_corners[:, 0].max()
        y_max = rotated_corners[:, 1].max()
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    def flip_bbox(self, bbox, axis, img_w, img_h):
        """Flips bounding box horizontally or vertically."""
        x, y, w, h = bbox
        if axis == 0:
            x = img_w - (x + w)
        elif axis == 1:
            y = img_h - (y + h)
        return [x, y, w, h]

    def transform_annotation(self, ann, transform_key, orig_w, orig_h, new_w, new_h):
        """Transforms bbox and segmentation in annotation based on augmentation type."""
        new_ann = deepcopy(ann)
        bbox = new_ann.get('bbox')

        if 'r90' in transform_key or 'r180' in transform_key or 'r270' in transform_key:
            angle = 90 if 'r90' in transform_key else (180 if 'r180' in transform_key else 270)
            if bbox:
                new_ann['bbox'] = self.rotate_bbox(bbox, angle, orig_w, orig_h)
            if 'segmentation' in new_ann:
                segs = [
                    self.rotate_points(seg, angle, orig_w / 2, orig_h / 2)
                    for seg in new_ann['segmentation']
                ]
                new_ann['segmentation'] = [s.tolist() for s in segs]

        elif transform_key == 'hflip':
            if bbox:
                new_ann['bbox'] = self.flip_bbox(bbox, 0, orig_w, orig_h)
            if 'segmentation' in new_ann:
                new_ann['segmentation'] = [
                    self.flip_points(seg, 0, orig_w, orig_h) for seg in new_ann['segmentation']
                ]

        elif transform_key == 'vflip':
            if bbox:
                new_ann['bbox'] = self.flip_bbox(bbox, 1, orig_w, orig_h)
            if 'segmentation' in new_ann:
                new_ann['segmentation'] = [
                    self.flip_points(seg, 1, orig_w, orig_h) for seg in new_ann['segmentation']
                ]

        elif transform_key == 'r90-vflip':
            if bbox:
                bbox = self.rotate_bbox(bbox, 90, orig_w, orig_h)
                new_ann['bbox'] = self.flip_bbox(bbox, 1, new_w, new_h)
            if 'segmentation' in new_ann:
                segs = []
                for seg in new_ann['segmentation']:
                    seg = self.rotate_points(seg, 90, orig_w / 2, orig_h / 2)
                    seg = self.flip_points(seg, 1, new_w, new_h)
                    segs.append(seg)
                new_ann['segmentation'] = [s.tolist() for s in segs]

        elif transform_key == 'r90-hflip':
            if bbox:
                bbox = self.rotate_bbox(bbox, 90, orig_w, orig_h)
                new_ann['bbox'] = self.flip_bbox(bbox, 0, new_w, new_h)
            if 'segmentation' in new_ann:
                segs = []
                for seg in new_ann['segmentation']:
                    seg = self.rotate_points(seg, 90, orig_w / 2, orig_h / 2)
                    seg = self.flip_points(seg, 0, new_w, new_h)
                    segs.append(seg)
                new_ann['segmentation'] = [s.tolist() for s in segs]

        elif transform_key == 'r90-hvflip':
            if bbox:
                bbox = self.rotate_bbox(bbox, 90, orig_w, orig_h)
                bbox = self.flip_bbox(bbox, 0, new_w, new_h)
                new_ann['bbox'] = self.flip_bbox(bbox, 1, new_w, new_h)
            if 'segmentation' in new_ann:
                segs = []
                for seg in new_ann['segmentation']:
                    seg = self.rotate_points(seg, 90, orig_w / 2, orig_h / 2)
                    seg = self.flip_points(seg, 0, new_w, new_h)
                    seg = self.flip_points(seg, 1, new_w, new_h)
                    segs.append(seg)
                new_ann['segmentation'] = [s.tolist() for s in segs]

        return new_ann

    def augment(self):
        """Run full augmentation pipeline: transform images and update annotations."""
        print("Starting COCO dataset augmentation...")

        for img_id, img_info in self.images.items():
            filename = img_info['file_name']
            base_name, ext = os.path.splitext(filename)
            if ext.lower() not in self.image_extensions:
                continue

            img_path = os.path.join(self.images_dir, filename)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Failed to load image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            for key, transform_func in self.transforms.items():
                try:
                    transformed_img = transform_func(img_rgb.copy())
                except Exception as e:
                    print(f"Error applying {key} to {filename}: {e}")
                    continue

                suffix = self.suffix_map[key]
                out_filename = f"{base_name}{suffix}{ext}"
                out_path = os.path.join(self.output_images_dir, out_filename)

                Image.fromarray(transformed_img).save(out_path)

                new_img_id = f"{img_id}{suffix}"
                self.output_images.append({
                    "id": new_img_id,
                    "file_name": out_filename,
                    "width": transformed_img.shape[1],
                    "height": transformed_img.shape[0]
                })

                img_anns = self.anns_by_image.get(img_id, [])
                for ann in img_anns:
                    new_ann = self.transform_annotation(
                        ann, key, orig_w, orig_h,
                        transformed_img.shape[1], transformed_img.shape[0]
                    )
                    new_ann['image_id'] = new_img_id
                    new_ann['id'] = f"{ann['id']}{suffix}"
                    self.output_annotations.append(new_ann)

        # Save updated COCO JSON
        output_coco = {
            "images": self.output_images,
            "annotations": self.output_annotations,
            "categories": self.output_categories
        }

        with open(self.output_annotations_file, 'w') as f:
            json.dump(output_coco, f, indent=2)

        print(f"Augmentation complete!")
        print(f"   - {len(self.output_images)} images saved to '{self.output_images_dir}'")
        print(f"   - Annotations saved to '{self.output_annotations_file}'")


if __name__ == "__main__":
    """Parse command line arguments and run augmentation."""
    parser = argparse.ArgumentParser(description="Augment COCO dataset using a YAML config file.")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    # Load YAML config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract required paths
    input_cfg = config['INPUT']
    output_cfg = config['OUTPUT']

    images_dir = input_cfg['images_dir']
    annotations_file = input_cfg['annotations_file']
    output_images_dir = output_cfg['output_images_dir']
    output_annotations_file = output_cfg['output_annotations_file']

    # Optional settings (with defaults handled in class)
    aug_cfg = config.get('AUGMENTATIONS', {})
    optional_args = {
        key: config.get(key) for key in [
            'rotations',
            'horizontal_flip',
            'vertical_flip',
            'rotate90_plus_vertical_flip',
            'rotate90_plus_horizontal_flip',
            'rotate90_plus_hv_flip',
            'save_original',
            'image_extensions'
        ] if key in config
    }
    # Also allow AUGMENTATIONS to contain the flags
    optional_args.update({
        key: aug_cfg.get(key, False) for key in [
            'horizontal_flip',
            'vertical_flip',
            'rotate90_plus_vertical_flip',
            'rotate90_plus_horizontal_flip',
            'rotate90_plus_hv_flip'
        ]
    })
    if 'rotations' in aug_cfg:
        optional_args['rotations'] = aug_cfg['rotations']

    # Instantiate and run
    augmenter = COCOAugmenter(
        images_dir=images_dir,
        annotations_file=annotations_file,
        output_images_dir=output_images_dir,
        output_annotations_file=output_annotations_file,
        **optional_args
    )
    augmenter.augment()
