#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
import yaml


class COCOAugmenter:
    """
    COCOAugmenter class for geometric augmentation of COCO-formatted datasets.
    Applies rotations, flips, and combinations, updates image files and annotations accordingly.
    """

        def __init__(self, config_path):
        """
        Initialize the COCOAugmenter with settings from a YAML configuration file.

        The configuration file must define input/output paths, augmentation options,
        and supported image formats. This method loads the dataset, sets up transforms,
        and prepares output structures.

        Args:
            config_path (str): Path to the YAML configuration file. The file should contain:
                - INPUT:
                    - images_dir (str): Path to the folder containing original images.
                    - annotations_file (str): Path to the COCO annotations JSON file.
                - OUTPUT:
                    - output_images_dir (str): Directory where augmented images will be saved.
                    - output_annotations_file (str): Path to save the updated COCO JSON file.
                - image_extensions (list of str): List of valid image file extensions (e.g., ['.png', '.jpg']).
                - save_original (bool): If True, saves the original image with '_orig' suffix.
                - AUGMENTATIONS (dict): Flags for which geometric transforms to apply:
                    - rotations (list of int): List of rotation angles (e.g., [90, 180, 270]).
                    - horizontal_flip (bool): Whether to apply left-right flip.
                    - vertical_flip (bool): Whether to apply top-bottom flip.
                    - rotate90_plus_vertical_flip (bool): If True, applies 90° rotation + vertical flip.
                    - rotate90_plus_horizontal_flip (bool): If True, applies 90° rotation + horizontal flip.
                    - rotate90_plus_hv_flip (bool): If True, applies 90° rotation + both horizontal and vertical flips.
        """
        # Load configuration from YAML
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Input paths
        self.images_dir = self.config['INPUT']['images_dir']  # Directory containing original images
        self.annotations_file = self.config['INPUT']['annotations_file']  # Path to COCO annotations JSON

        # Output paths
        self.output_images_dir = self.config['OUTPUT']['output_images_dir']  # Directory to save augmented images
        self.output_annotations_file = self.config['OUTPUT']['output_annotations_file']  # Path to save updated JSON

        # Supported image file extensions (lowercase)
        self.extensions = [ext.lower() for ext in self.config['image_extensions']]

        # Whether to save original image with '_orig' suffix
        self.save_original = self.config.get('save_original', True)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_images_dir, exist_ok=True)

        # Load COCO dataset
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        # Map image ID to image metadata
        self.images = {img['id']: img for img in self.coco_data['images']}

        # List of all annotations
        self.annotations = self.coco_data['annotations']

        # List of categories (preserved as-is)
        self.categories = self.coco_data['categories']

        # Map from image ID to list of its annotations
        self.anns_by_image = {}
        for ann in self.annotations:
            self.anns_by_image.setdefault(ann['image_id'], []).append(ann)

        # Output containers for augmented data
        self.output_images = []  # List of new image entries
        self.output_annotations = []  # List of new annotation entries
        self.output_categories = self.categories  # Categories remain unchanged

        # Dictionaries to define transformation functions and filename suffixes
        self.transforms = {}
        self.suffix_map = {}

        # Build transformation functions based on config
        self._build_transforms()

    def _build_transforms(self):
        """
        Builds a dictionary of transformation functions and corresponding filename suffixes
        based on the settings in the config file.
        """
        cfg = self.config['AUGMENTATIONS']

        # Always include the original (unchanged) image transform
        self.transforms['orig'] = lambda img: img
        self.suffix_map['orig'] = '_orig'

        # Add 90, 180, 270 degree rotations if specified
        if 'rotations' in cfg:
            for angle in cfg['rotations']:
                key = f'r{angle}'
                k = angle // 90  # Number of 90° rotations
                self.transforms[key] = lambda img, k=k: np.rot90(img, k)
                self.suffix_map[key] = f'_r{angle}'

        # Horizontal flip: left-right
        if cfg.get('horizontal_flip', False):
            self.transforms['hflip'] = lambda img: cv2.flip(img, 1)
            self.suffix_map['hflip'] = '_hflip'

        # Vertical flip: top-bottom
        if cfg.get('vertical_flip', False):
            self.transforms['vflip'] = lambda img: cv2.flip(img, 0)
            self.suffix_map['vflip'] = '_vflip'

        # Rotate 90° then flip vertically
        if cfg.get('rotate90_plus_vertical_flip', False):
            self.transforms['r90-vflip'] = lambda img: cv2.flip(np.rot90(img, 1), 0)
            self.suffix_map['r90-vflip'] = '_r90-vflip'

        # Rotate 90° then flip horizontally
        if cfg.get('rotate90_plus_horizontal_flip', False):
            self.transforms['r90-hflip'] = lambda img: cv2.flip(np.rot90(img, 1), 1)
            self.suffix_map['r90-hflip'] = '_r90-hflip'

        # Rotate 90° then flip both ways (horizontal + vertical)
        if cfg.get('rotate90_plus_hv_flip', False):
            self.transforms['r90-hvflip'] = lambda img: cv2.flip(np.rot90(img, 1), -1)
            self.suffix_map['r90-hvflip'] = '_r90-hvflip'

    @staticmethod
    def rotate_points(points, angle, cx, cy):
        """
        Rotates a list of (x, y) points around a given center by a specified angle.

        Args:
            points (list): List of [x1, y1, x2, y2, ...] coordinates.
            angle (float): Rotation angle in degrees (positive = counterclockwise).
            cx (float): X-coordinate of rotation center.
            cy (float): Y-coordinate of rotation center.

        Returns:
            list: Transformed point coordinates after rotation.
        """
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        points = np.array(points).reshape(-1, 2)
        rotated = (points - [cx, cy]) @ np.array([[cos_a, sin_a], [-sin_a, cos_a]]) + [cx, cy]
        return rotated.reshape(-1).tolist()

    @staticmethod
    def flip_points(points, axis, img_w, img_h):
        """
        Flips a list of (x, y) points horizontally or vertically.

        Args:
            points (list): List of [x1, y1, x2, y2, ...] coordinates.
            axis (int): 0 for horizontal flip (left-right), 1 for vertical flip (top-bottom).
            img_w (int): Width of the image (used for mirroring).
            img_h (int): Height of the image (used for mirroring).

        Returns:
            list: Transformed point coordinates after flip.
        """
        points = np.array(points).reshape(-1, 2)
        if axis == 0:  # Horizontal flip
            points[:, 0] = img_w - points[:, 0]
        elif axis == 1:  # Vertical flip
            points[:, 1] = img_h - points[:, 1]
        return points.reshape(-1).tolist()

    def rotate_bbox(self, bbox, angle, img_w, img_h):
        """
        Rotates a bounding box (x, y, w, h) by angle around image center and computes new axis-aligned box.

        Args:
            bbox (list): Original bounding box in [x, y, w, h] format.
            angle (float): Rotation angle in degrees.
            img_w (int): Original image width.
            img_h (int): Original image height.

        Returns:
            list: New bounding box in [x, y, w, h] after rotation and realignment.
        """
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
        """
        Flips a bounding box horizontally or vertically.

        Args:
            bbox (list): Bounding box in [x, y, w, h] format.
            axis (int): 0 for horizontal, 1 for vertical flip.
            img_w (int): Image width.
            img_h (int): Image height.

        Returns:
            list: Flipped bounding box in [x, y, w, h] format.
        """
        x, y, w, h = bbox
        if axis == 0:  # Horizontal flip
            x = img_w - (x + w)
        elif axis == 1:  # Vertical flip
            y = img_h - (y + h)
        return [x, y, w, h]

    def transform_annotation(self, ann, transform_key, orig_w, orig_h, new_w, new_h):
        """
        Transforms an annotation (bbox and segmentation) based on the augmentation type.

        Args:
            ann (dict): Original annotation dictionary.
            transform_key (str): Key indicating the transform (e.g., 'r90', 'hflip').
            orig_w (int): Original image width.
            orig_h (int): Original image height.
            new_w (int): Transformed image width.
            new_h (int): Transformed image height.

        Returns:
            dict: New annotation with updated bbox and segmentation.
        """
        new_ann = deepcopy(ann)
        bbox = new_ann.get('bbox')

        # Handle rotation (90, 180, 270)
        if 'r90' in transform_key or 'r180' in transform_key or 'r270' in transform_key:
            angle = 90 if 'r90' in transform_key else (180 if 'r180' in transform_key else 270)
            if bbox:
                new_ann['bbox'] = self.rotate_bbox(bbox, angle, orig_w, orig_h)
            if 'segmentation' in new_ann:
                segs = []
                for seg in new_ann['segmentation']:
                    segs.append(self.rotate_points(seg, angle, orig_w / 2, orig_h / 2))
                new_ann['segmentation'] = [s.tolist() for s in segs]

        # Pure horizontal flip
        elif transform_key == 'hflip':
            if bbox:
                new_ann['bbox'] = self.flip_bbox(bbox, 0, orig_w, orig_h)
            if 'segmentation' in new_ann:
                new_ann['segmentation'] = [
                    self.flip_points(seg, 0, orig_w, orig_h) for seg in new_ann['segmentation']
                ]

        # Pure vertical flip
        elif transform_key == 'vflip':
            if bbox:
                new_ann['bbox'] = self.flip_bbox(bbox, 1, orig_w, orig_h)
            if 'segmentation' in new_ann:
                new_ann['segmentation'] = [
                    self.flip_points(seg, 1, orig_w, orig_h) for seg in new_ann['segmentation']
                ]

        # Rotate 90° then flip vertically
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

        # Rotate 90° then flip horizontally
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

        # Rotate 90° then flip both ways
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
        """
        Executes the full augmentation pipeline: loads images, applies transforms,
        saves augmented images, and updates COCO annotations with new coordinates.

        This is the main method to run after initialization.
        """
        print("Starting COCO dataset augmentation...")

        for img_id, img_info in self.images.items():
            filename = img_info['file_name']
            base_name, ext = os.path.splitext(filename)
            if ext.lower() not in self.extensions:
                continue  # Skip unsupported formats

            img_path = os.path.join(self.images_dir, filename)
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            # Load image in BGR, convert to RGB
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Failed to load image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            # Apply each transformation
            for key, transform_func in self.transforms.items():
                try:
                    transformed_img = transform_func(img_rgb.copy())
                except Exception as e:
                    print(f"Error applying {key} to {filename}: {e}")
                    continue

                # Generate output filename
                suffix = self.suffix_map[key]
                out_filename = f"{base_name}{suffix}{ext}"
                out_path = os.path.join(self.output_images_dir, out_filename)

                # Save the augmented image
                Image.fromarray(transformed_img).save(out_path)

                # Add new image metadata
                new_img_id = f"{img_id}{suffix}"
                new_img_info = {
                    "id": new_img_id,
                    "file_name": out_filename,
                    "width": transformed_img.shape[1],
                    "height": transformed_img.shape[0]
                }
                self.output_images.append(new_img_info)

                # Transform and add annotations
                img_anns = self.anns_by_image.get(img_id, [])
                for ann in img_anns:
                    new_ann = self.transform_annotation(
                        ann, key, orig_w, orig_h,
                        transformed_img.shape[1], transformed_img.shape[0]
                    )
                    new_ann['image_id'] = new_img_id
                    new_ann['id'] = f"{ann['id']}{suffix}"
                    self.output_annotations.append(new_ann)

        # Save the updated COCO JSON
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


# === CLI Usage ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment COCO dataset with geometric transforms.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file (default: config.yaml)')

    args = parser.parse_args()

    augmenter = COCOAugmenter(config_path=args.config)
    augmenter.augment()