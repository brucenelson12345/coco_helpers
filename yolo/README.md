# Dataset Preprocessing
A Python script to convert a COCO-style dataset into YOLO format for object detection training.
This tool processes train, val, and test subsets, reads annotations.json files, and generates corresponding .txt label files in YOLO format inside a new labels/ directory for each subset.

## The input dataset should be organized as follows:

dataset_root/
├── train/
│   ├── annotations.json
│   └── images/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── val/
│   ├── annotations.json
│   └── images/
├── test/
    ├── annotations.json
    └── images/

Each annotations.json must follow the standard COCO annotation format.

Output

After conversion, a labels/ folder is created in each subset:

dataset_root/
└── train/
    ├── images/
    ├── labels/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── annotations.json

Each .txt file contains detections in YOLO format:
<class_id> <x_center> <y_center> <width> <height>
All values are normalized relative to the image dimensions (range: 0.0 to 1.0).

Command to run
    python coco_to_yolo.py ./data/coco_dataset

Features
- Converts COCO bbox annotations to YOLO format
- Handles train, val, and test splits automatically
- Creates labels/ directories if they don't exist
- Uses actual image dimensions (via OpenCV) for accurate normalization
- Maps COCO category IDs to continuous 0-based class indices
- Skips missing or unreadable images with warnings
- Command-line interface with argparse
- Robust path handling (supports full paths in file_name)


Notes
- Image file extensions are preserved; .txt files use the same base name.
- Coordinates are clamped to [0.0, 1.0] to avoid floating-point boundary issues.
- If your dataset doesn't have a test or val split, the script will skip it with a warning.

# Trainer
A Python-based training script for YOLOv8 object detection models using custom datasets.
Automatically infers class names and counts from COCO-style annotations.json, supports
configurable training via YAML, and integrates seamlessly with Ultralytics YOLOv8.

Key Features:
  - Automatic class detection from train/annotations.json (COCO format)
  - Minimal config: only dataset path required
  - Optional custom pretrained model support (.pt)
  - Configurable via YAML
  - Supports train, validation, and test splits
  - Evaluation on test set (optional)
  - Compatible with YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x

Dataset Structure:
  Your dataset must follow this layout:

  dataset_root/
  ├── train/
  │   ├── images/           # .png, .jpg, etc.
  │   ├── labels/           # YOLO .txt label files (one per image)
  │   └── annotations.json  # COCO format (for class inference)
  ├── val/
  │   ├── images/
  │   ├── labels/
  │   └── annotations.json
  └── test/
      ├── images/
      ├── labels/
      └── annotations.json

Required Files:
  - Images: .png, .jpg, .jpeg, or .bmp in each images/ folder
  - Labels: .txt files in YOLO format (class_id x_center y_center width height, normalized)
  - annotations.json: COCO format (at least in train/) to auto-detect classes

Installation:
  1. Install dependencies:
     pip install ultralytics

  2. (Optional) For JSON parsing, ensure standard libraries (json) are available (built-in).

Configuration:
  Create a YAML config file (e.g., config.yaml):

    dataset_path: /path/to/your/dataset_root   # REQUIRED

    # Optional overrides (auto-inferred if omitted)
    # num_classes: 3
    # class_names: [person, bicycle, car]

    # Training parameters
    model_size: n           # n, s, m, l, x
    epochs: 100
    imgsz: 640
    batch: 16
    workers: 8              # data loading workers
    pretrained: path/to/model.pt  # optional custom checkpoint
    project: runs/train
    name: yolov8n_exp1
    exist_ok: true
    evaluate_on_test: true  # run final evaluation on test set

Run Training:
  python train_yolov8.py config.yaml

Output:
  - Training logs and models saved to: runs/train/yolov8n_exp1/
  - Best and last checkpoints: weights/best.pt, weights/last.pt
  - Results include mAP, precision, recall
  - Optional test set evaluation printed at the end

Notes:
  - If class_names or num_classes are omitted, they are auto-inferred from
    train/annotations.json (must be in COCO format with 'categories' array).
  - Label .txt files must match image filenames (e.g., image123.png → image123.txt).
  - Ensure class IDs in .txt files are consistent with COCO category indexing.

Tips:
  - Use 'workers' to speed up data loading (set to # of CPU cores).
  - Use 'pretrained' to resume training or fine-tune a custom model.
  - Set 'batch: -1' for auto-batch size (recommended for GPU memory optimization).

# Inference_and_evaluate
A clean, configurable, and easy-to-use Python script to train YOLOv8 object 
detection models on custom datasets. Automatically infers class labels from 
COCO-style annotations and supports full configuration via a single YAML file.

Ideal for researchers, engineers, and ML practitioners who want fast, 
reproducible training with minimal setup.

-------------------------------------------------------------------------------

FEATURES

- Auto-infer classes: Reads train/annotations.json (COCO format) to detect 
  num_classes and class_names
- Minimal config: Only 'dataset_path' is required
- Flexible training: Supports YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- Custom pretrained models: Resume training or fine-tune from a .pt checkpoint
- Test set evaluation: Optional post-training evaluation on test split
- Configurable via YAML: All parameters are optional and well-documented
- Data validation: Checks dataset structure and label files
- Compatible with Ultralytics: Full integration with 'ultralytics' ecosystem

-------------------------------------------------------------------------------

DATASET STRUCTURE

Your dataset must follow this directory layout:

dataset_root/
├── train/
│   ├── images/           # .png, .jpg, .jpeg, .bmp
│   ├── labels/           # YOLO .txt label files (one per image)
│   └── annotations.json  # COCO format (for class inference)
├── val/
│   ├── images/
│   ├── labels/
│   └── annotations.json
└── test/
    ├── images/
    ├── labels/
    └── annotations.json

🔍 The script uses train/annotations.json to auto-detect class names and count. 
   If omitted, you must specify 'num_classes' in the config.

-------------------------------------------------------------------------------

🛠️ INSTALLATION

1. Install the Ultralytics YOLOv8 package:
   pip install ultralytics

2. Place the following files in your project:
   - train_yolov8.py
   - config.yaml (or your chosen config name)

No other dependencies required — yaml, json, and pathlib are built-in.

-------------------------------------------------------------------------------

CONFIGURATION (YAML)

Create a config.yaml file. Only 'dataset_path' is required.

Example: config.yaml

dataset_path: /path/to/your/dataset_root   # REQUIRED

# Optional: override auto-inferred classes
# num_classes: 3
# class_names: [person, bicycle, car]

# Training settings
model_size: n              # n, s, m, l, x → default: 'n'
epochs: 100                # default: 100
imgsz: 640                 # input image size → default: 640
batch: 16                  # batch size (-1 for auto) → default: 16
workers: 8                 # data loader workers → default: 8
pretrained: ./best.pt      # optional custom checkpoint

# Output settings
project: runs/train        # output root → default: 'runs/train'
name: yolov8n_exp1         # experiment name → default: 'yolov8{n}_train'
exist_ok: true             # allow overwriting → default: True

# Evaluation
evaluate_on_test: true     # run test-set eval → default: false

See config_example.yaml for a fully commented version.

-------------------------------------------------------------------------------

USAGE

Run the training script:
  python train_yolov8.py config.yaml

Example Output:
  Verifying dataset structure...
  Dataset structure verified.
  Creating data.yaml...
  data.yaml saved to dataset_root/data.yaml
  Loading YOLOv8 model: yolov8n.pt
  Starting training...
  ...
  Training completed.
  Evaluating on test set...
  Test mAP50-95: 0.7621

-------------------------------------------------------------------------------

OUTPUT

Training results are saved to:

runs/train/yolov8n_exp1/
├── weights/
│   ├── best.pt    # best model
│   └── last.pt    # final model
├── results.csv
├── results.png    # mAP, loss curves
└── train_batch*.jpg  # augmented sample batches

You can load the trained model later:
  from ultralytics import YOLO
  model = YOLO('runs/train/yolov8n_exp1/weights/best.pt')

-------------------------------------------------------------------------------

NOTES

- Label .txt files must be in YOLO format:
    class_id center_x center_y width height  # all normalized 0–1

- Image and label filenames must match:
    image123.jpg ↔ image123.txt

- COCO annotations.json must have a 'categories' array with 'id' and 'name'

- Use batch: -1 to let YOLO auto-detect optimal batch size

-------------------------------------------------------------------------------

ADVANCED TIPS

- Resume training: Set 'pretrained' to runs/train/exp/weights/last.pt
- Fine-tuning: Use a custom .pt model path
- Larger models: Try model_size: m or l for higher accuracy
- Logging: Integrates with TensorBoard (automatic) and Weights & Biases (optional)