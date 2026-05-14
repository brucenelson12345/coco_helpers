YOLO Object Detection Pipeline: Edge Impulse to Deployment
This repository contains a complete pipeline of scripts designed to take a dataset from Edge Impulse, convert it into COCO and YOLO formats, split it, train a YOLO model, evaluate it, and export it for deployment.

Prerequisites
Before running any scripts, you need to set up a Python 3.11 virtual environment on your Linux machine and install the necessary dependencies.

Bash
# Update package list and install Python 3.11 and venv (if not already installed)
sudo apt update
sudo apt install python3.11 python3.11-venv

# Create a virtual environment named 'venv'
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
Pipeline Scripts Overview
1. edge_to_coco.py
Description: Converts an Edge Impulse project directory (containing info.labels and training/testing subdirectories) into separate COCO-formatted datasets (train and val).

Arguments:

input_dir (Positional): Path to the Edge Impulse project directory.

Example Usage:

Bash
python edge_to_coco.py ./my_edge_impulse_project
2. annotations_splitter.py
Description: Splits a single COCO dataset into train, val, and test subsets based on configurable ratios defined in a YAML file. This is useful if your original dataset wasn't fully split.

Arguments:

-c, --config (Required): Path to the YAML configuration file.

Config File (splitter_train-val-test.yml):

YAML
image_dir: ./path_to_images_directory
annotations_file: ./path_to_annotations.json
output_dir: ./path_to_output_directory

# Optional
train_ratio: 0.7
val_ratio: 0.2
test_ratio: 0.1
seed: 42
Example Usage:

Bash
python annotations_splitter.py -c splitter_train-val-test.yml
3. coco_to_yolo.py
Description: Converts annotations from a split COCO dataset (train, val, test subsets) into the YOLO .txt format. It calculates normalized bounding box coordinates and creates labels directories corresponding to the images directories.

Arguments:

dataset_dir (Positional): Path to the dataset root directory containing the train/, val/, and test/ folders.

Example Usage:

Bash
python coco_to_yolo.py ./my_split_dataset
4. YOLO Training Configurations (trainer.yaml)
Description: These files define the settings for training the YOLO model. While the training script itself (e.g., trainer.py or the ultralytics CLI) executes the loop, these configurations dictate the dataset paths, hyperparameters, and target classes.

trainer.yaml (Model & Training Parameters):

YAML
dataset_path: ./Path_to_dataset
weights: ./Path_to_YOLO_pretrained_model
name: "My_Trained_Model"
epochs: 100
rect: True
imgsz: 640
batch: 16
workers: 16
pretrained: False
trainer_s.yaml (Dataset Structure):

Bash
# Assuming a standard generic YOLO training script
python trainer.py -c trainer.yaml

5. export_onnx.py
Description: Exports a trained PyTorch YOLO model (.pt) to an ONNX graph (.onnx). It automatically locks dynamic input shapes to a fixed resolution using ONNX runtime tools for maximum compatibility with edge devices.

Arguments:

-i, --input (Required): Path to the input YOLO .pt model.

-r, --resolution (Required): Image resolution as two integers Height Width (e.g., 480 640).

-o, --output (Required): Path to the output .onnx file.

Example Usage:

Bash
python export_onnx.py -i runs/train/weights/best.pt -r 480 640 -o model_fixed.onn

OPTIONAL
ground_truth.py
Description: Validates your dataset by drawing semi-transparent bounding boxes and category-specific labels onto the raw images using the COCO annotations. This acts as a visual sanity check before training.

Arguments:

-c, --config (Required): Path to the YAML configuration file.

Config File (groundtruth_coco.yml):

YAML
image_dir: ./path_to_images_directory
annotation_file: ./path_to_annotations.json
output_dir: ./path_to_output_directory

# Optional
draw_labels: true
thickness: 2
font_scale: 0.6
alpha: 0.3
Example Usage:

Bash
python ground_truth.py -c groundtruth_coco.yml
inference.py
Description: Runs inference on single images using a trained YOLO .pt model. It visually plots the detected bounding boxes, confidence scores, and class labels onto the image and saves the result.

Arguments:

-m, --model (Required): Path to the trained YOLO model (.pt).

-i, --input (Required): Path to the input image file.

-o, --output (Required): Path to save the resulting annotated image.

Example Usage:

Bash
python inference.py -m best.pt -i sample_image.jpg -o inference_result.jpg
