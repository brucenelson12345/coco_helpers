import os
import json
import random
from collections import defaultdict, Counter
from typing import List, Dict, Set

# === CONFIG ===
image_dir = "path/to/your/images"           # Update this
coco_json_path = "path/to/annotations.json" # Update this
output_dir = "output_splits"                # Update this

os.makedirs(output_dir, exist_ok=True)
random.seed(42)  # For reproducibility

# Load COCO data
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
# We want: category -> list of image groups that have at least one annotation in that category
cat_to_groups = defaultdict(set)  # cat_id -> set of group ids (index in image_groups)
group_to_cats = defaultdict(set)  # group index -> set of cat_ids

# Also track group index to actual group
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

# === Step 3: Stratified assignment using greedy balancing ===
# We'll assign each group to train/val/test such that per-category counts are as close to 70/20/10 as possible

train_groups = set()
val_groups = set()
test_groups = set()

# Final assignment: group index -> split
assignment = {}

# Sort categories by number of groups (smallest first to prioritize rare classes)
sorted_cats = sorted(cat_to_groups.keys(), key=lambda c: len(cat_to_groups[c]))

# Initialize counts per category per split
cat_counts = {
    'train': Counter(),
    'val': Counter(),
    'test': Counter()
}

total_cat_counts = {cat: len(groups) for cat, groups in cat_to_groups.items()}

# Function to compute imbalance if we assign group `idx` to `split`
def compute_imbalance_loss(idx, split):
    loss = 0.0
    target_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
    current = cat_counts[split]
    for cat_id in group_to_cats[idx]:
        total = total_cat_counts[cat_id]
        current_count = current[cat_id]
        new_count = current_count + 1
        target = target_ratios[split] * total
        # Squared error before and after
        error_before = (current_count - target) ** 2
        error_after = (new_count - target) ** 2
        loss += error_after - error_before  # Change in error
    return loss

# Greedy assignment: sort groups randomly and assign to least "costly" split
group_indices = list(range(len(image_groups)))
random.shuffle(group_indices)

for idx in group_indices:
    cats = group_to_cats[idx]
    if not cats:
        # No annotations, assign to train
        assignment[idx] = 'train'
        train_groups.add(idx)
        continue

    # Compute cost for each split
    costs = {}
    for split in ['train', 'val', 'test']:
        costs[split] = compute_imbalance_loss(idx, split)

    # Choose split with minimum cost
    chosen_split = min(costs, key=costs.get)
    assignment[idx] = chosen_split
    if chosen_split == 'train':
        train_groups.add(idx)
    elif chosen_split == 'val':
        val_groups.add(idx)
    else:
        test_groups.add(idx)

    # Update counts
    for cat_id in cats:
        cat_counts[chosen_split][cat_id] += 1

# === Step 4: Ensure at least one of each category in each split ===
all_cats = set(cat_id_to_name.keys())

def get_cats_in_splits():
    train_cats = set()
    val_cats = set()
    test_cats = set()
    for idx in train_groups:
        train_cats.update(group_to_cats[idx])
    for idx in val_groups:
        val_cats.update(group_to_cats[idx])
    for idx in test_groups:
        test_cats.update(group_to_cats[idx])
    return train_cats, val_cats, test_cats

train_cats, val_cats, test_cats = get_cats_in_splits()
missing_cats = all_cats - (train_cats & val_cats & test_cats)

if missing_cats:
    print(f"Ensuring all categories are in all splits. Missing: {missing_cats}")
    # For each missing category, find a group that has it and move it
    for cat_id in missing_cats:
        candidate_groups = sorted(cat_to_groups[cat_id], key=lambda idx: sum(len(image_id_to_anns[img['id']]) for img in image_groups[idx]))
        for idx in candidate_groups:
            if idx in train_groups or idx in val_groups or idx in test_groups:
                continue  # Already assigned? Shouldn't happen
            # Assign to the split that has the fewest total images (to balance)
            sizes = [
                (len(test_groups), 'test'),
                (len(val_groups), 'val'),
                (len(train_groups), 'train')
            ]
            target = min(sizes)[1]
            assignment[idx] = target
            if target == 'train':
                train_groups.add(idx)
            elif target == 'val':
                val_groups.add(idx)
            else:
                test_groups.add(idx)
            # Update counts
            for c in group_to_cats[idx]:
                cat_counts[target][c] += 1
            break

# === Step 5: Extract images for each split ===
def get_images_from_group_indices(indices):
    imgs = []
    for idx in indices:
        imgs.extend(image_groups[idx])
    return imgs

train_imgs = get_images_from_group_indices(train_groups)
val_imgs = get_images_from_group_indices(val_groups)
test_imgs = get_images_from_group_indices(test_groups)

# === Step 6: Remap IDs and save COCO JSONs ===
def remap_coco_json(images: List[Dict], annotations: List[Dict], image_id_to_anns: Dict) -> Dict:
    new_images = []
    new_annotations = []
    img_id_map = {}
    ann_id_counter = 0

    # Sort by original ID for determinism
    sorted_imgs = sorted(images, key=lambda x: x['id'])

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

train_coco = remap_coco_json(train_imgs, annotations, image_id_to_anns)
val_coco = remap_coco_json(val_imgs, annotations, image_id_to_anns)
test_coco = remap_coco_json(test_imgs, annotations, image_id_to_anns)

# Save JSONs
with open(os.path.join(output_dir, 'train.json'), 'w') as f:
    json.dump(train_coco, f, indent=2)
with open(os.path.join(output_dir, 'val.json'), 'w') as f:
    json.dump(val_coco, f, indent=2)
with open(os.path.join(output_dir, 'test.json'), 'w') as f:
    json.dump(test_coco, f, indent=2)

# Optional: Copy images
for split_name, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
    split_img_dir = os.path.join(output_dir, split_name, 'images')
    os.makedirs(split_img_dir, exist_ok=True)
    for img in img_list:
        src = os.path.join(image_dir, img['file_name'])
        dst = os.path.join(split_img_dir, img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} not found")

# === Step 7: Print statistics ===
print("\n✅ Split completed:")
print(f"Train: {len(train_imgs)} images ({len(train_groups)} groups)")
print(f"Val:   {len(val_imgs)} images ({len(val_groups)} groups)")
print(f"Test:  {len(test_imgs)} images ({len(test_groups)} groups)")

print("\n📊 Category Distribution (per split):")
for cat_id, cat_name in cat_id_to_name.items():
    tr = cat_counts['train'][cat_id]
    va = cat_counts['val'][cat_id]
    te = cat_counts['test'][cat_id]
    total = tr + va + te
    print(f"{cat_name} ({cat_id}): {tr:3d} train ({tr/total:.1%}), {va:2d} val ({va/total:.1%}), {te:2d} test ({te/total:.1%})")
