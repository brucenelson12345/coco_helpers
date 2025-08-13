import os
import json
import cv2
import argparse
from pathlib import Path

class COCO2YOLOConverter:
    """
    A class to convert COCO-style dataset annotations to YOLO format.
    """
    
    def __init__(self, dataset_dir):
        """
        Initialize the converter with the path to the dataset root directory.
        
        Args:
            dataset_dir (str or Path): Path to the dataset root containing train/val/test folders.
        """
        self.dataset_dir = Path(dataset_dir)
        self.subsets = ['train', 'val', 'test']
    
    def convert(self):
        """
        Converts annotations in each subset (train, val, test) from COCO to YOLO format.
        Creates a 'labels' folder in each subset with corresponding .txt files.
        """
        for subset in self.subsets:
            self._convert_subset(subset)
    
    def _convert_subset(self, subset):
        """
        Converts a single subset (e.g., 'train') from COCO to YOLO format.
        
        Args:
            subset (str): One of 'train', 'val', 'test'
        """
        subset_dir = self.dataset_dir / subset
        annotations_file = subset_dir / 'annotations.json'
        images_dir = subset_dir / 'images'
        labels_dir = subset_dir / 'labels'
        
        if not annotations_file.exists():
            print(f"[WARNING] Annotations file not found: {annotations_file}, skipping {subset}")
            return
        
        if not images_dir.exists():
            print(f"[WARNING] Images directory not found: {images_dir}, skipping {subset}")
            return
        
        # Create labels directory
        labels_dir.mkdir(exist_ok=True)
        
        # Load COCO JSON
        print(f"Converting {subset}...")
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Map image id to image info
        images = {img['id']: img for img in coco_data['images']}
        
        # Map category id to continuous YOLO class index (0-based)
        categories = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        print(f"  Found {len(categories)} categories")
        
        # Group annotations by image_id
        annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations:
                annotations[img_id] = []
            annotations[img_id].append(ann)
        
        # Process each image
        converted_count = 0
        for img_id, img_info in images.items():
            file_name = img_info['file_name']
            # Extract filename in case of path (e.g., "folder/img.jpg" -> "img.jpg")
            file_name = Path(file_name).name
            img_path = images_dir / file_name
            
            if not img_path.exists():
                print(f"  [WARNING] Image not found: {img_path}, skipping...")
                continue
            
            # Read image to get dimensions
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [WARNING] Could not read image: {img_path}, skipping...")
                continue
            h, w = image.shape[:2]
            
            # Generate YOLO annotations
            yolo_lines = []
            if img_id in annotations:
                for ann in annotations[img_id]:
                    # COCO bbox: [x_min, y_min, width, height]
                    x_min, y_min, bbox_w, bbox_h = ann['bbox']
                    
                    # Convert to YOLO format: normalized center x, y, width, height
                    x_center = (x_min + bbox_w / 2) / w
                    y_center = (y_min + bbox_h / 2) / h
                    width = bbox_w / w
                    height = bbox_h / h
                    
                    # Get class ID
                    cat_id = ann['category_id']
                    if cat_id not in categories:
                        print(f"  [WARNING] Unknown category ID: {cat_id}")
                        continue
                    class_id = categories[cat_id]
                    
                    # Clip values to [0.0, 1.0] to avoid floating-point boundary issues
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Write to .txt file
            txt_filename = img_path.stem + '.txt'
            txt_path = labels_dir / txt_filename
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            converted_count += 1
        
        print(f"  Converted {converted_count} images in {subset} to YOLO format.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO-style dataset to YOLO format.")
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset root directory containing 'train/', 'val/', 'test/' folders."
    )
    args = parser.parse_args()
    
    converter = COCO2YOLOConverter(args.dataset_dir)
    converter.convert()
