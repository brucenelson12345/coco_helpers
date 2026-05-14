EDGE IMPULSE TO YOLO/ONNX PIPELINE

This repository contains a complete pipeline to convert an Edge Impulse object detection dataset into COCO and YOLO formats, train a YOLO model, export it to ONNX, and run inference & validation.

PREREQUISITES
Ensure you have the required Python libraries installed:
pip install -r requirements.txt


DATA PREPARATION & CONVERSION
1. edge_to_coco.py
Description: Converts a raw Edge Impulse project directory into standard COCO format datasets for training and validation. It reads the info.labels file, extracts bounding boxes, and copies the images into train and val folders while generating annotations.json files.
Required Arguments:
  - input_dir: Path to the Edge Impulse project directory (must contain training/testing folders and info.labels).
Example Usage:
  python edge_to_coco.py ./path/to/edge_impulse_export

2. annotations_splitter.py
Description: Splits a single COCO dataset directory into train, val, and test subsets based on a defined ratio, automatically copying images and creating new JSON annotations for each split.
Required Arguments:
  - --config or -c: Path to the YAML configuration file.
Example YAML (split.yaml):
  image_dir: ./coco_dataset/images
  annotations_file: ./coco_dataset/annotations.json
  output_dir: ./split_dataset
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  seed: 42
Example Usage:
  python annotations_splitter.py -c split.yaml

3. coco_to_yolo.py
Description: Converts a standard COCO dataset (containing train, val, test subsets with annotations.json) into the YOLO format. It generates a labels/ directory for each subset containing .txt files with normalized bounding box coordinates.
Required Arguments:
  - dataset_dir: Path to the root dataset directory containing train/, val/, and test/ folders.
Example Usage:
  python coco_to_yolo.py ./split_dataset


TRAINING & EXPORTING
4. trainer.py
Description: Trains a YOLO model using a YOLO-formatted dataset. It automatically generates a data.yaml file by inferring the class names and counts directly from the COCO annotations.json file located in the train/ directory.
Required Arguments:
  - --config or -c: Path to the YAML configuration file.
Example YAML (train.yaml):
  dataset_path: ./split_dataset
  weights: yolov8n.pt
  name: my_custom_model
  epochs: 50
  imgsz: 640
  batch: 16
  workers: 8
  devices: [0]
Example Usage:
  python trainer.py -c train.yaml

5. export_onnx.py
Description: Exports a trained YOLO .pt model to ONNX format. It includes a critical step that uses onnxruntime tools to fix the dynamic input shapes to a static resolution, which is highly beneficial for edge deployment.
Required Arguments:
  - --input or -i: Path to the input YOLO .pt model.
  - --resolution or -r: Target fixed resolution as Height Width (e.g., 480 640).
  - --output or -o: Path to the output .onnx file.
Example Usage:
  python export_onnx.py -i runs/train/my_custom_model/weights/best.pt -r 640 640 -o best_fixed.onnx


INFERENCE & VISUALIZATION
6. inference.py
Description: Runs YOLO inference on a single image and visually overlays the bounding boxes, class labels, and confidence scores directly onto the image.
Required Arguments:
  - --model or -m: Path to the YOLO model (.pt).
  - --input or -i: Path to the input image file.
  - --output or -o: Path to save the annotated output image.
Example Usage:
  python inference.py -m best.pt -i test_image.jpg -o result.jpg

7. ground_truth.py
Description: A visualization tool that parses a COCO annotations.json file and draws the ground truth bounding boxes directly on the corresponding images. Useful for verifying dataset integrity before training.
Required Arguments:
  - --config or -c: Path to the YAML configuration file.
Example YAML (groundtruth_coco.yaml):
  image_dir: ./split_dataset/train/images
  annotation_file: ./split_dataset/train/annotations.json
  output_dir: ./ground_truth_viz
  draw_labels: true
  thickness: 2
  font_scale: 0.6
  alpha: 0.3
Example Usage:
  python ground_truth.py -c groundtruth_coco.yaml


EVALUATION & VALIDATION
8. inference_evaluate.py
Description: Runs bulk inference over a directory of images, saves visually annotated versions of the results, and calculates overall and per-class mAP by comparing the YOLO outputs against a COCO annotations.json file using the pycocotools API.
Required Arguments:
  - config: Path to the YAML configuration file. (Positional argument)
Example YAML (eval.yaml):
  weights: best.pt
  device: cuda:0
  data_dir: ./split_dataset/val
  output_dir: ./evaluation_results
  conf_threshold: 0.25
Example Usage:
  python inference_evaluate.py eval.yaml

9. validator.py
Description: Runs YOLO validation on a test/val set and automatically groups the resulting mAP scores based on class name prefixes. This is particularly useful if your dataset has sub-variants of a primary object class (e.g., car_red, car_blue grouped under car).
Required Arguments:
  - --model: Path to the YOLO model file (.pt).
Optional Arguments:
  - --output-csv: Path to save the grouped metrics CSV (Defaults to grouped_map_results.csv).
Example Usage:
  python validator.py --model best.pt --output-csv final_metrics.csv
