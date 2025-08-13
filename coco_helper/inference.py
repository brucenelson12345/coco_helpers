from ultralytics import YOLO
import os

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "path/to/your/model.pt"          # Your trained model
DATA_YAML = "path/to/dataset.yaml"            # Dataset YAML with 'test' key
OUTPUT_DIR = "inference_results"              # Directory to save images
CONF_THRESHOLD = 0.25                         # Confidence threshold
IMG_SIZE = 640                                # Inference image size

# ----------------------------
# Create output directory
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load the YOLOv8 model
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# 1. Run validation to get mAP on test set
# ----------------------------
print("Computing mAP on test set...")
metrics = model.val(
    data=DATA_YAML,
    split='test',               # Use test split
    imgsz=IMG_SIZE,
    batch=16,
    conf=CONF_THRESHOLD,
    device=0,                   # Use GPU (remove or set to 'cpu' if not available)
    plots=True                  # Generates some plots (optional)
)

# Print mAP50 and mAP50-95
print(f"mAP@0.5: {metrics.box.map:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map50_95:.4f}")

# ----------------------------
# 2. Run prediction and save images with bounding boxes
# ----------------------------
print("Running inference and saving images with bounding boxes...")

results = model.predict(
    source=DATA_YAML.replace("dataset.yaml", "test"),  # Or better: extract test path from YAML
    imgsz=IMG_SIZE,
    conf=CONF_THRESHOLD,
    save=True,                  # This saves the images
    project=OUTPUT_DIR,         # Directory to save
    name='predicted_images',    # Subdirectory name
    exist_ok=True,
    device=0
)

print(f"Inference completed. Annotated images saved in: {os.path.join(OUTPUT_DIR, 'predicted_images')}")
print(f"mAP@0.5: {metrics.box.map:.4f}, mAP@0.5:0.95: {metrics.box.map50_95:.4f}")