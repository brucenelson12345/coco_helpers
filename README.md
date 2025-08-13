# Virtual Environments
Python virtual environments are used

Python3.12

    python3.12 -m venv .env
    source .env/bin/activate

- augmentation_requirements.txt

    pip install -r augmentation_requirements.txt

# Augments
## Geometric Augmenter
augment_geometric.py geometrically augments a COCO-formatted dataset using configurable rotations and flips. This script generates augmented images (e.g., rotated, flipped) and **automatically updates the corresponding annotations** (bounding boxes and segmentation polygons) to match the transformed images.

Features
- Preserves original image as `_orig`
- Supports **rotations** (90, 180, 270)
- Supports **horizontal and vertical flips**
- Supports **combined transforms** (e.g., rotate 90 + flip)
- Updates **bounding boxes** and **polygon segmentation** coordinates
- Fully configurable via a **YAML file**

Input Requirements
- `images/`: Directory containing your original images (`.png`, `.jpg`, etc.)
- `annotations.json`: A COCO-format JSON file with `images`, `annotations`, and `categories`

The output includes:
- A new folder of augmented images with descriptive filenames
- An updated COCO `.json` annotation file with correct image entries and transformed annotations

Configuration (YAML)

Create a `geometric.yaml` file to define paths and augmentation settings.

Full YAML Structure

```yaml
INPUT:
  images_dir: "path/to/images"               # Required: input image folder
  annotations_file: "path/to/annotations.json"  # Required: COCO annotations

OUTPUT:
  output_images_dir: "path/to/output/images"     # Required: where to save augmented images
  output_annotations_file: "path/to/output/annotations_aug.json"  # Required: output JSON path
# Optional settings (all have defaults if omitted)
rotations: [90, 180, 270]
horizontal_flip: true
vertical_flip: true
rotate90_plus_vertical_flip: true
rotate90_plus_horizontal_flip: true
rotate90_plus_hv_flip: true
save_original: true
image_extensions:
  - ".png"
# Optional: alternative grouping (same as above, takes lower precedence)
AUGMENTATIONS:
  horizontal_flip: true
  vertical_flip: true
  rotate90_plus_vertical_flip: true
  rotate90_plus_horizontal_flip: true
  rotate90_plus_hv_flip: true
  rotations: [90, 180, 270]

Command to run

  python augment_geometric.py --config configs/geometric.yaml

## Geometric Augmenter
augment_photometric.py photometrically augments a COCO-formatted dataset using photometric transforms (e.g., brightness, contrast, noise, blur) to a COCO-formatted dataset **without altering bounding box coordinates**. This script preserves object annotations while diversifying image appearance.

Features:
- Reads original images and COCO JSON annotations.
- Applies configurable photometric transformations.
- Saves **original + augmented images** to a new folder.
- Generates an **updated COCO JSON** that includes entries for all new augmented images.

Input Requirements
- `images/`: Directory containing your original images (`.png`, `.jpg`, etc.)
- `annotations.json`: A COCO-format JSON file with `images`, `annotations`, and `categories`

The output includes:
- A new folder of augmented images with descriptive filenames
- An updated COCO `.json` annotation file with correct image entries and transformed annotations

Configuration (YAML)

Create a `photometric.yaml` file to define paths and augmentation settings.

Full YAML Structure

```yaml
paths:
  original_images_dir: "path/to/images"                # Directory containing original .png/.jpg images
  original_annotation_file: "path/to/annotations.json" # Input COCO annotation file
  output_images_dir: "path/to/augmented_images"        # Output directory for images (created if not exists)
  output_annotation_file: "path/to/augmented_annotations.json" # Output COCO JSON file

Command to run

  python augment_photometric.py --config configs/photometric.yaml

# Verification
## Ground Truth Bounding Box Visualizer
ground_truth.py is used to visualize bounding boxes by their respective annotations and bboxes by drawing **color-coded, semi-transparent bounding boxes** on images. Each object category is assigned a **unique random color**, and labels.

Features:
- Per-category unique colors** (consistent across all instances)
- Semi-transparent filled bounding boxes** for better visibility
- Solid border** around each box
- Black background + white text** for clear labels
- Handles nested image directories
- Configurable via YAML file
- Supports custom class names

Input Requirements
- `images/`: Directory containing your original images (`.png`, `.jpg`, etc.)
- `annotations.json`: A COCO-format JSON file with `images`, `annotations`, and `categories`

The output includes:
- A new folder of drawn bounding box images

Configuration (YAML)

Create a `ground_truth.yaml` file to define paths and augmentation settings.

Full YAML Structure

```yaml
paths:
  image_dir: "path/to/images"                # Directory containing original .png/.jpg images
  annotation_file: "path/to/annotations.json" # Input COCO annotation file
  output_dir: "path/to/augmented_images"        # Output directory for images (created if not exists)

Command to run

  python ground_truth.py --config configs/ground_truth.yaml

# Splitters
## Random Splitter
random_splitter.py is used to split a COCO-formatted dataset into train, validation, and test subsets with reindexed image IDs and properly mapped annotations.

Features:
- Splits images and annotations into train, val, and test sets using user-defined ratios.
- Copies only the relevant images into each subset.
- Rewrites COCO JSON annotations with new image IDs starting from 0.
- Preserves categories, info, and licenses from the original JSON.
- COCO formatted output directories

The output includes:
- A new folder of images split into train, val, and test
- New annotations.json for each train, val, and test

Configuration (YAML)

Create a `random_splitter.yaml` file to define paths and augmentation settings.

Full YAML Structure

```yaml
paths:
  image_dir: "path/to/images"                # Directory containing original .png/.jpg images
  annotation_file: "path/to/annotations.json" # Input COCO annotation file
  output_dir: "path/to/augmented_images"        # Output directory for images (created if not exists)
# Optional (defaults shown)
  train_ratio: 0.7
  val_ratio: 0.2
  test_ratio: 0.1
  seed: 42

Command to run

  python random_splitter.py --config configs/random_splitter.yaml

## Group Splitter
group_splitter.py is used to split a COCO-formatted dataset into train, validation, and test subsets with reindexed image IDs and properly mapped annotations.

Features:
- Splits images and annotations into train, val, and test sets using groups of original + augmentations of an image.
- Copies only the relevant images of the group into each subset.
- Rewrites COCO JSON annotations with new image IDs starting from 0.
- Preserves categories, info, and licenses from the original JSON.
- COCO formatted output directories

The output includes:
- A new folder of images split into train, val, and test
- New annotations.json for each train, val, and test

Configuration (YAML)

Create a `group_splitter.yaml` file to define paths and augmentation settings.

Full YAML Structure

```yaml
paths:
  image_dir: "path/to/images"                # Directory containing original .png/.jpg images
  annotation_file: "path/to/annotations.json" # Input COCO annotation file
  output_dir: "path/to/augmented_images"        # Output directory for images (created if not exists)

Command to run

  python group_splitter.py --config configs/group_splitter.yaml