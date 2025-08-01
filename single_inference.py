import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "your_model.pt"           # Path to your .pt model
IMAGE_PATH = "input_image.png"         # Path to your input .png image
OUTPUT_PATH = "output_image.png"       # Where to save the result
CONFIDENCE_THRESHOLD = 0.25            # Minimum confidence to show detection
BOX_COLOR = (0, 255, 0)                # Bounding box and label color (BGR)
FILL_COLOR = (0, 255, 0)               # Fill color for bounding box
FILL_ALPHA = 0.2                       # Transparency for filled box (0=transparent, 1=solid)
TEXT_BG_ALPHA = 0.6                    # Transparency for label background
TEXT_COLOR = (255, 255, 255)           # Label text color
FONT_SCALE = 0.6
FONT_THICKNESS = 1
BORDER_PADDING = 5                     # Padding around text in label box

# -------------------------------
# Load Model and Run Inference
# -------------------------------
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, conf=CONFIDENCE_THRESHOLD)
result = results[0]  # Get the first (and only) result

# Load image
img = cv2.imread(IMAGE_PATH)
img_copy = img.copy()
h, w = img.shape[:2]

# Create overlay for transparent boxes
overlay = img.copy()

# -------------------------------
# Process each detection
# -------------------------------
for box in result.boxes:
    # Get coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    conf = box.conf.item()
    cls_id = int(box.cls.item())
    class_name = model.names[cls_id]

    # Draw filled bounding box with transparency
    cv2.rectangle(overlay, (x1, y1), (x2, y2), FILL_COLOR, -1)  # Filled
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)        # Outline

    # Prepare label text
    label = f"{class_name} {conf:.2f}"

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    text_h += 2 * BORDER_PADDING

    # Position label box above the bounding box (or below if near top)
    label_y = y1 - 10
    label_x = x1

    # Flip below if too close to top
    if label_y < text_h:
        label_y = y2 + 2
    else:
        label_y -= text_h

    # Ensure label box stays within image bounds
    label_x = max(0, label_x)
    label_x_end = min(w, label_x + text_w)
    label_y = max(0, label_y)
    label_y_end = min(h, label_y + text_h)

    # Draw semi-transparent background for label
    cv2.rectangle(overlay, (label_x, label_y), (label_x_end, label_y_end), FILL_COLOR, -1)
    cv2.rectangle(img, (label_x, label_y), (label_x_end, label_y_end), BOX_COLOR, 1)

    # Add text
    cv2.putText(img, label, (label_x + BORDER_PADDING, label_y + text_h - BORDER_PADDING - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

# -------------------------------
# Apply transparency overlay
# -------------------------------
cv2.addWeighted(overlay, FILL_ALPHA, img, 1 - FILL_ALPHA, 0, img)

# -------------------------------
# Save and display result
# -------------------------------
cv2.imwrite(OUTPUT_PATH, img)
print(f"Output saved to {OUTPUT_PATH}")

# Optional: Display image (comment out if running headless)
# cv2.imshow("YOLOv8 Inference", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()