import os
import json
import yaml
import random
import shutil
from collections import defaultdict, Counter
from typing import List, Dict, Set

def load_config(config_path: str) -> Dict[str, str]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['image_dir', 'coco_json_path', 'output_dir']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        if not config[key]:
            raise ValueError(f"Config key '{key}' is empty or not set.")
        config[key] = os.path.expanduser(config[key])  # Support ~
    
    return config

# === Load Config from YAML ===
# Update this path to your config file
CONFIG_FILE = "config.yaml"  # Change this if needed

config = load_config(CONFIG_FILE)

image_dir = config['image_dir']
coco_json_path = config['coco_json_path']
output_dir = config['output_dir']

# Validate paths
if not os.path.isdir(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")

if not os.path.isfile(coco_json_path):
    raise FileNotFoundError(f"COCO JSON file not found: {coco_json_path}")

os.makedirs(output_dir, exist_ok=True)
print(f"Using config:\nImage Dir: {image_dir}\nCOCO JSON: {coco_json_path}\nOutput Dir: {output_dir}\n")

random.seed(42)  # For reproducibility

# === Load COCO Data ===
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

coco_images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

# Map: image_id -> image dict
image_id_to_img = {img['id']: img for img in coco_images}

# Map: image_id -> list of annotations
image_id_to_anns = defaultdict(list)
for ann in annotations:
    image_id_to_anns[ann['image_id']].append(ann)

# Map: file_name -> image dict
file_to_img = {img['file_name']: img for img in coco_images}

def extract_base_name(filename: str) -> str:
    """Extract base name by removing augmentation suffix."""
    name, ext = os.path.splitext(filename)
    if '_' in name:
        candidate = name.rsplit('_', 1)[0]
        if os.path.exists(os.path.join(image_dir, candidate + ext)):
            return candidate
    return name

# === Step 1: Group images by base name ===
base_to_images = defaultdict(list)
for img in coco_images:
    fname = img['file_name']
    base = extract_base_name(fname)
    base_to_images[base].append(img)

image_groups = list(base_to_images.values())
print(f"Total image groups (each base + aug): {len(image_groups)}")

# === Step 2: Map each image group to its categories ===
cat_to_groups = defaultdict(set)  # cat_id -> set of group indices
group_to_cats = defaultdict(set)  # group index -> set of cat_ids

group_index_to_group = {i: group for i, group in enumerate(image_groups)}

for idx, group in enumerate(image_groups):
    img_ids_in_group = {img['id'] for img in group}
    cats_in_group = set()
    for img_id in img_ids_in_group:
        for ann in image_id_to_anns[img_id]:
            cat_id = ann['category_id']
            cats_in_group.add(cat_id)
            cat_to_groups[cat_id].add(idx)
    group_to_cats[idx] = cats_in_group

# === Step 3: Stratified assignment using greedy cost minimization ===
train_groups = set()
val_groups = set()
test_groups = set()
assignment = {}

cat_counts = {
    'train': Counter(),
    'val': Counter(),
    'test': Counter()
}

total_cat_counts = {cat: len(groups) for cat, groups in cat_to_groups.items()}

def compute_imbalance_loss(idx: int, split: str) -> float:
    loss = 0.0
    target_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    current = cat_counts[split]
    for cat_id in group_to_cats[idx]:
        total = total_cat_counts[cat_id]
        current_count = current[cat_id]
        new_count = current_count + 1
        target = target_ratios[split] * total
        error_before = (current_count - target) ** 2
        error_after = (new_count - target) ** 2
        loss += (error_after - error_before)
    return loss

# Shuffle group indices
group_indices = list(range(len(image_groups)))
random.shuffle(group_indices)

for idx in group_indices:
    cats = group_to_cats[idx]
    if not cats:
        assignment[idx] = 'train'
        train_groups.add(idx)
        cat_counts['train'].update(cats)
        continue

    costs = {split: compute_imbalance_loss(idx, split) for split in ['train', 'val', 'test']}
    chosen_split = min(costs, key=costs.get)
    assignment[idx] = chosen_split

    if chosen_split == 'train':
        train_groups.add(idx)
    elif chosen_split == 'val':
        val_groups.add(idx)
    else:
        test_groups.add(idx)

    for cat_id in cats:
        cat_counts[chosen_split][cat_id] += 1

# === Step 4: Ensure all categories in all splits ===
def get_cats_in_splits():
    return (
        set().union(*(group_to_cats[i] for i in train_groups)),
        set().union(*(group_to_cats[i] for i in val_groups)),
        set().union(*(group_to_cats[i] for i in test_groups))
    )

train_cats, val_cats, test_cats = get_cats_in_splits()
all_cats = set(cat_id_to_name.keys())
missing_cats = all_cats - (train_cats & val_cats & test_cats)

if missing_cats:
    print(f"🔧 Ensuring all categories are in all splits. Missing: {missing_cats}")
    for cat_id in missing_cats:
        candidates = sorted(cat_to_groups[cat_id], key=lambda i: len(image_groups[i]))
        for idx in candidates:
            if idx in train_groups or idx in val_groups or idx in test_groups:
                continue
            # Assign to smallest split
            sizes = [(len(test_groups), 'test'), (len(val_groups), 'val'), (len(train_groups), 'train')]
            target = min(sizes)[1]
            assignment[idx] = target
            if target == 'train':
                train_groups.add(idx)
            elif target == 'val':
                val_groups.add(idx)
            else:
                test_groups.add(idx)
            for c in group_to_cats[idx]:
                cat_counts[target][c] += 1
            break

# === Step 5: Extract images ===
def get_images_from_indices(indices):
    return [img for idx in indices for img in image_groups[idx]]

train_imgs = get_images_from_indices(train_groups)
val_imgs = get_images_from_indices(val_groups)
test_imgs = get_images_from_indices(test_groups)

# === Step 6: Remap and Save COCO JSONs ===
def remap_coco_json(images: List[Dict], image_id_to_anns: Dict) -> Dict:
    new_images = []
    new_annotations = []
    ann_id_counter = 0

    sorted_imgs = sorted(images, key=lambda x: x['id'])
    img_id_map = {}

    for new_id, img in enumerate(sorted_imgs):
        old_id = img['id']
        img_id_map[old_id] = new_id
        new_img = img.copy()
        new_img['id'] = new_id
        new_images.append(new_img)

    for img in new_images:
        old_id = [i['id'] for i in coco_images if i['file_name'] == img['file_name']][0]
        anns = image_id_to_anns[old_id]
        for ann in anns:
            new_ann = ann.copy()
            new_ann['id'] = ann_id_counter
            ann_id_counter += 1
            new_ann['image_id'] = img['id']
            new_annotations.append(new_ann)

    return {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories
    }

train_coco = remap_coco_json(train_imgs, image_id_to_anns)
val_coco = remap_coco_json(val_imgs, image_id_to_anns)
test_coco = remap_coco_json(test_imgs, image_id_to_anns)

# Save JSONs
for name, coco in [('train', train_coco), ('val', val_coco), ('test', test_coco)]:
    path = os.path.join(output_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Saved {name}.json with {len(coco['images'])} images and {len(coco['annotations'])} annotations")

# === Step 7: Copy images (optional) ===
for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
    split_img_dir = os.path.join(output_dir, split_name, 'images')
    os.makedirs(split_img_dir, exist_ok=True)
    for img in img_list:
        src = os.path.join(image_dir, img['file_name'])
        dst = os.path.join(split_img_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"⚠️ Warning: Source image not found: {src}")

# === Step 8: Print Summary ===
print("\n✅ Final Split Summary:")
print(f"Train: {len(train_imgs)} images ({len(train_groups)} groups)")
print(f"Val:   {len(val_imgs)} images ({len(val_groups)} groups)")
print(f"Test:  {len(test_imgs)} images ({len(test_groups)} groups)")

print("\n📊 Category Distribution per Split:")
print(f"{'Class':<15} {'Train':<8} {'Val':<6} {'Test':<6} {'Total':<6}")
print("-" * 45)
for cat_id, cat_name in sorted(cat_id_to_name.items()):
    tr = cat_counts['train'][cat_id]
    va = cat_counts['val'][cat_id]
    te = cat_counts['test'][cat_id]
    total = tr + va + te
    print(f"{cat_name:<15} {tr:<8} {va:<6} {te:<6} {total:<6}")
