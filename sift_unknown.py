import cv2
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------
KNOWN_CLASSES = ['sports ball', 'baseball', 'tennis ball', 'soccer ball', 'basketball']
CONFIDENCE_THRESHOLD = 0.5
MIN_MATCH_COUNT = 10  # Minimum SIFT/ORB matches to consider "known"
RATIO_TEST_THRESHOLD = 0.7  # Lowe's ratio test
FEATURE_METHOD = 'SIFT'  # or 'ORB'

# Paths
KNOWN_IMAGES_DIR = "known_balls"  # Folder with sample images of known balls
OUTPUT_DIR = "detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Initialize Feature Detector
# -------------------------------
if FEATURE_METHOD.upper() == 'SIFT':
    detector = cv2.SIFT_create()
elif FEATURE_METHOD.upper() == 'ORB':
    detector = cv2.ORB_create(nfeatures=500)  # Adjust as needed
else:
    raise ValueError("FEATURE_METHOD must be 'SIFT' or 'ORB'")

# -------------------------------
# Build Reference Descriptors
# -------------------------------
reference_data = []  # List of dicts: {name, img, keypoints, descriptors}

def build_reference_database():
    """Build reference database from known ball images using SIFT/ORB"""
    if not os.path.exists(KNOWN_IMAGES_DIR):
        print(f"[WARNING] {KNOWN_IMAGES_DIR} not found. Running in fallback mode (all low-confidence = unknown).")
        return False

    for fname in os.listdir(KNOWN_IMAGES_DIR):
        path = os.path.join(KNOWN_IMAGES_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Extract features
        kp, desc = detector.detectAndCompute(img, None)
        if desc is None or len(kp) == 0:
            print(f"No features in {fname}")
            continue

        base_name = os.path.splitext(fname)[0]
        reference_data.append({
            'name': base_name,
            'image': img,
            'keypoints': kp,
            'descriptors': desc
        })
        print(f"Loaded reference: {base_name} ({len(kp)} keypoints)")

    print(f"[INFO] Built reference DB with {len(reference_data)} templates.")
    return True

# Try to load reference images
use_reference_db = build_reference_database()

# -------------------------------
# Match Query Image to References
# -------------------------------
def match_to_known_templates(query_gray):
    """
    Match query image (gray) to reference templates.
    Returns True if matches a known object.
    """
    if not use_reference_db:
        return False, None

    query_kp, query_desc = detector.detectAndCompute(query_gray, None)
    if query_desc is None or len(query_desc) == 0:
        return False, None

    # FLANN parameters
    if FEATURE_METHOD.upper() == 'SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
    else:  # ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    best_match_name = None
    max_good_matches = 0

    for ref in reference_data:
        try:
            matches = flann.knnMatch(query_desc, ref['descriptors'], k=2)
            # Apply Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                    good.append(m)
            if len(good) > max_good_matches:
                max_good_matches = len(good)
                best_match_name = ref['name']
        except cv2.error as e:
            print(f"Matching error: {e}")
            continue

    is_known = max_good_matches >= MIN_MATCH_COUNT
    return is_known, best_match_name if is_known else None

# -------------------------------
# Load YOLO Model
# -------------------------------
from ultralytics import YOLO
yolo_model = YOLO('yolov8n.pt')  # or your custom model

# -------------------------------
# Main Detection + Unknown Classification
# -------------------------------
def detect_and_classify(image):
    results = yolo_model(image)
    detections = results[0].boxes

    output_img = image.copy()

    for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        confidence = conf.item()
        class_id = int(cls.item())
        label_name = yolo_model.names[class_id].lower()

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Convert to grayscale for SIFT/ORB
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # -------------------------------
        # Decision Logic
        # -------------------------------
        final_label = "unknown_ball"
        color = (0, 0, 255)  # Red

        # Rule 1: High confidence + known class → likely known
        if confidence > CONFIDENCE_THRESHOLD and label_name in [k.lower() for k in KNOWN_CLASSES]:
            # But still verify with SIFT/ORB if possible
            is_similar, _ = match_to_known_templates(crop_gray)
            if is_similar:
                final_label = label_name
                color = (0, 255, 0)  # Green
            else:
                final_label = f"unknown_ball ({label_name})"
        else:
            # Rule 2: Low confidence or unknown class → check visual similarity
            is_similar, matched_name = match_to_known_templates(crop_gray)
            if is_similar:
                final_label = matched_name or label_name
                color = (255, 165, 0)  # Orange – possible misclassification
            else:
                final_label = f"unknown_ball ({label_name})"

        # Draw bounding box and label
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        text = f"{final_label} {confidence:.2f}"
        cv2.putText(output_img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_img

# -------------------------------
# Run on Webcam or Image
# -------------------------------
if __name__ == "__main__":
    # Option 1: From webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_and_classify(frame)
        cv2.imshow("SIFT/ORB + YOLO Unknown Detection", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Option 2: From image file
    # image = cv2.imread("test.jpg")
    # if image is None:
    #     print("Image not found!")
    #     exit()
    # result = detect_and_classify(image)
    # cv2.imwrite("detections/result_sift_orb.jpg", result)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()