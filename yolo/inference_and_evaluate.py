import os
import cv2
import json
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


class InferenceEvaluator:
    """
    A class to run inference on images, draw bounding boxes with labels,
    and evaluate mAP using COCO-style annotations.
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.class_names = []
        self.colors = config.get("colors", [
            [0, 150, 255], [255, 150, 0], [0, 255, 150],
            [150, 0, 255], [255, 0, 150], [150, 255, 0]
        ])
        self.conf_threshold = config.get("conf_threshold", 0.25)

    def load_model(self):
        """Load YOLO model."""
        print(f"Loading model: {self.config['weights']}")
        self.model = YOLO(self.config["weights"])
        self.model.to(self.config["device"])

    def load_class_names(self):
        """Load class names from COCO annotations."""
        anno_json_path = Path(self.config["data_dir"]) / "annotations.json"
        with open(anno_json_path, 'r') as f:
            coco_data = json.load(f)
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        sorted_ids = sorted(categories.keys())
        self.class_names = [categories[cls_id] for cls_id in sorted_ids]

    def draw_bounding_boxes(self, image, boxes, labels, confs=None):
        """Draw transparent filled boxes with black background + white text."""
        overlay = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        for i, (box, cls_id) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box)
            color = self.colors[int(cls_id) % len(self.colors)]
            label = self.class_names[int(cls_id)]

            # Filled transparent box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Text background (black)
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)
            # White text
            cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return image

    def run_inference_and_save_images(self):
        """Run inference and save annotated images."""
        image_dir = Path(self.config["data_dir"]) / "images"
        output_image_dir = Path(self.config["output_dir"]) / "images"
        output_image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in image_dir.glob("*.png")])
        detections = []

        print(f"Running inference on {len(image_paths)} images...")

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            h, w = img.shape[:2]
            results = self.model(img, device=self.config["device"], conf=self.conf_threshold, verbose=False)
            result = results[0]

            boxes = []
            labels = []
            confs = []

            for det in result.boxes:
                xyxy = det.xyxy[0].cpu().numpy()
                conf = det.conf.cpu().numpy()[0]
                cls = det.cls.cpu().numpy()[0]

                x1, y1, x2, y2 = xyxy
                width = x2 - x1
                height = y2 - y1

                # Try to extract image ID from filename (e.g., '000001.png' -> 1)
                try:
                    image_id = int(img_path.stem.lstrip("0")) if img_path.stem.lstrip("0") else hash(img_path.name) % 10000
                except:
                    image_id = hash(img_path.name) % 10000

                detections.append({
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(conf)
                })

                boxes.append([x1, y1, x2, y2])
                labels.append(cls)
                confs.append(conf)

            # Draw and save
            img_with_boxes = self.draw_bounding_boxes(img, boxes, labels, confs)
            cv2.imwrite(str(output_image_dir / img_path.name), img_with_boxes)

        return detections

    def evaluate_mAP(self, detections):
        """Evaluate mAP using COCO API and write results."""
        anno_json_path = Path(self.config["data_dir"]) / "annotations.json"
        output_txt_path = Path(self.config["output_dir"]) / "mAP_results.txt"

        coco_gt = COCO(anno_json_path)
        coco_dt = coco_gt.loadRes([{
            "image_id": d["image_id"],
            "category_id": d["category_id"],
            "bbox": d["bbox"],
            "score": d["score"]
        } for d in detections])

        # Temp file for COCO API
        temp_file = "temp_detections.json"
        with open(temp_file, 'w') as f:
            json.dump(detections, f)

        # Overall mAP
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        overall_ap = coco_eval.stats[0]  # AP@0.5:0.95

        # Per-class mAP
        per_class_ap = {}
        for cls_name in self.class_names:
            # Find category ID
            cat_ids = coco_gt.getCatIds(catNms=[cls_name])
            if not cat_ids:
                per_class_ap[cls_name] = 0.0
                continue
            cat_id = cat_ids[0]

            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            per_class_ap[cls_name] = coco_eval.stats[0]

        # Restore full categories
        coco_eval.params.catIds = sorted(coco_gt.getCatIds())

        # Save to file
        with open(output_txt_path, 'w') as f:
            f.write("mAP Evaluation Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Overall mAP (AP@0.5:0.95): {overall_ap:.4f}\n\n")
            f.write("Per-Category mAP (AP@0.5:0.95):\n")
            for name, ap in per_class_ap.items():
                f.write(f"{name}: {ap:.4f}\n")

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"mAP results saved to {output_txt_path}")
        return overall_ap, per_class_ap

    def run(self):
        """Main execution pipeline."""
        # Create output directory
        Path(self.config["output_dir"]).mkdir(parents=True, exist_ok=True)

        # Load components
        self.load_model()
        self.load_class_names()

        # Run inference
        detections = self.run_inference_and_save_images()

        # Evaluate mAP
        if detections:
            self.evaluate_mAP(detections)
        else:
            print("No detections found. Skipping mAP evaluation.")

        print(f"Inference completed. Output saved to {self.config['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Run inference and mAP evaluation using a YAML config.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate and run
    evaluator = InferenceEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()