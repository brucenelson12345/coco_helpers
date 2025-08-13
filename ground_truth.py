#!/usr/bin/env python3
import os
import json
import cv2
import yaml
import argparse
import random
from pathlib import Path
from typing import Dict, Tuple, Optional


class COCOBBoxDrawer:
    """
    A class to draw bounding boxes on images based on COCO annotations.
    Each category gets a unique color. Bounding boxes are semi-transparent filled.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        output_dir: str,
        class_names: Optional[Dict[int, str]] = None,
        draw_labels: bool = True,
        thickness: int = 2,
        font_scale: float = 0.6,
        alpha: float = 0.3  # Transparency for filled box
    ):
        """
        Initialize the drawer.

        Args:
            image_dir: Path to image directory.
            annotation_file: Path to COCO JSON annotations.
            output_dir: Output directory for annotated images.
            class_names: Optional mapping of category_id -> name.
            draw_labels: Whether to draw class labels.
            thickness: Border thickness of bounding box.
            font_scale: Font scale for labels.
            alpha: Transparency level for filled boxes (0=transparent, 1=opaque).
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.output_dir = output_dir
        self.class_names = class_names
        self.draw_labels = draw_labels
        self.thickness = thickness
        self.font_scale = font_scale
        self.alpha = alpha

        # Validate inputs
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.isfile(self.annotation_file):
            raise ValueError(f"Annotation file not found: {self.annotation_file}")

        # Color cache: category_id -> (B, G, R)
        self.color_cache: Dict[int, Tuple[int, int, int]] = {}

    def _get_random_color(self) -> Tuple[int, int, int]:
        """Generate a random bright BGR color."""
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        return (b, g, r)

    def _get_color(self, category_id: int) -> Tuple[int, int, int]:
        """Get or create a color for a category."""
        if category_id not in self.color_cache:
            self.color_cache[category_id] = self._get_random_color()
        return self.color_cache[category_id]

    def draw(self):
        """Draw bounding boxes with category-specific colors and transparency."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Load COCO annotations
        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Build mappings
        image_id_to_filename = {img['id']: img['file_name'] for img in coco_data.get('images', [])}

        # Class names mapping
        if self.class_names is None:
            self.class_names = {
                cat['id']: cat['name'] for cat in coco_data.get('categories', [])
            }

        # Group annotations by image ID
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            annotations_by_image.setdefault(ann['image_id'], []).append(ann)

        processed_count = 0

        for img_id, anns in annotations_by_image.items():
            if img_id not in image_id_to_filename:
                print(f"Warning: Image ID {img_id} not found in images list. Skipping.")
                continue

            img_filename = image_id_to_filename[img_id]
            img_path = os.path.join(self.image_dir, img_filename)

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}. Skipping.")
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Failed to load image: {img_path}. Skipping.")
                continue

            overlay = image.copy()  # For transparent fills

            for ann in anns:
                x, y, w, h = map(int, ann['bbox'])
                x1, y1 = x, y
                x2, y2 = x + w, y + h

                category_id = ann['category_id']
                class_name = self.class_names.get(category_id, f"Class {category_id}")
                color = self._get_color(category_id)

                # Draw filled rectangle with transparency
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                # Draw solid border
                cv2.rectangle(image, (x1, y1), (x2, y2), color, self.thickness)

                # Draw label with black background and white text
                if self.draw_labels:
                    label = f"{class_name}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                    )
                    # Black background
                    cv2.rectangle(image, (x1, y1 - text_height - 10),
                                (x1 + text_width, y1), (0, 0, 0), -1)
                    # White text
                    cv2.putText(image, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                              (255, 255, 255), 1)

            # Apply overlay (semi-transparent filled boxes)
            cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0, image)

            # Save image
            output_path = os.path.join(self.output_dir, img_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            success = cv2.imwrite(output_path, image)
            if not success:
                print(f"Warning: Could not save image: {output_path}")
            else:
                processed_count += 1

        print(f"Successfully processed and saved {processed_count} annotated images to: {self.output_dir}")
        print(f"Used {len(self.color_cache)} unique colors for categories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes on COCO dataset images using a YAML config."
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        return

    required_keys = ['image_dir', 'annotation_file', 'output_dir']
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return

    optional_params = {
        'draw_labels': config.get('draw_labels', True),
        'thickness': config.get('thickness', 2),
        'font_scale': config.get('font_scale', 0.6),
        'alpha': config.get('alpha', 0.3),  # Transparency
        'class_names': config.get('class_names', None)
    }

    try:
        drawer = COCOBBoxDrawer(
            image_dir=config['image_dir'],
            annotation_file=config['annotation_file'],
            output_dir=config['output_dir'],
            **optional_params
        )
        drawer.draw()
    except Exception as e:
        print(f"Error during drawing: {e}")
