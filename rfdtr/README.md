# RF-DETR Unified Trainer
This project provides a robust, class-based Python script for fine-tuning RF-DETR (Real-Time Detection Transformer) models. It supports both Object Detection and Instance Segmentation tasks, handling common environmental hurdles like SSL certificate errors and dataset directory naming inconsistencies.

## Config.yaml

dataset_dir: "./drone_data"
task: "detection"
model_size: "small"
weights: null
epochs: 100
batch_size: 4
grad_accum_steps: 4
resolution: 640
output_dir: "./outputs"

The trainer is controlled via a YAML file. Below is an explanation of each parameter:
Parameter,Description,Valid Options
dataset_dir,Path to your COCO-formatted dataset (containing train and valid folders).,Any valid path
task,The computer vision task to perform.,"detection, segmentation"
model_size,The complexity of the model backbone.,"nano, small, medium, base, large, xlarge, 2xlarge"
weights,Path to local .pth weights. Use null to download official pre-trained weights.,Path or null
epochs,Total number of training passes over the dataset.,"Integer (e.g., 50)"
batch_size,Number of images processed per GPU step.,"Integer (e.g., 4)"
grad_accum_steps,Number of steps to accumulate gradients before updating weights.,"Integer (e.g., 4)"
resolution,Input image size for training. The script uses letterboxing to keep the aspect ratio.,"Integer (e.g., 640)"
output_dir,Directory where trained weights and logs will be saved.,Any valid path

Directory Structure:
dataset_dir/
├── train/
│   ├── _annotations.coco.json
│   └── images/
└── valid/ (or 'val')
    ├── _annotations.coco.json
    └── images/

## Installation
Python 3.11
`pip install rfdetr pyyaml`