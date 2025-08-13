#!/usr/bin/env python3
"""
Core class for splitting COCO datasets with grouped augmentation handling.
"""

import argparse
import os
import sys
import json
import random
import shutil
import yaml
from collections import defaultdict, Counter


class CocoGroupDatasetSplitter:
    """
    Splits a COCO dataset into train, val, and test sets while ensuring:
      - Images with the same base name (e.g., test_a_*.png) are kept together.
      - Each category appears in all three splits.
      - Final split ratios are close to 70% train, 20% val, 10% test.
    """

    def __init__(self, image_dir: str, coco_json_path: str, output_dir: str):
        """
        Initialize the splitter with required paths.

        :param image_dir: Directory containing all images (original + augmentations).
        :param coco_json_path: Path to the COCO annotations JSON file.
        :param output_dir: Directory where split JSONs and images will be saved.
        """
        self.image_dir = image_dir
        self.coco_json_path = coco_json_path
        self.output_dir = output_dir

        # Validate paths
        self._validate_paths()

        # Data containers
        self.coco_data = None
        self.coco_images = []
        self.annotations = []
        self.categories = []
        self.cat_id_to_name = {}

        self.image_id_to_img = {}
        self.image_id_to_anns = defaultdict(list)
        self.file_to_img = {}

        # Grouping
        self.group_idx_to_group = {}  # idx -> list of image dicts
        self.group_to_cats = defaultdict(set)  # idx -> set of category IDs
        self.cat_to_candidate_groups = defaultdict(list)  # cat_id -> list of (idx, group)

        # Assignments
        self.train_groups = set()
        self.val_groups = set()
        self.test_groups = set()
        self.assigned_groups = set()

        # Category counts per split
        self.cat_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    def _validate_paths(self):
        """Check that input paths exist and create output directory."""
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isfile(self.coco_json_path):
            raise FileNotFoundError(f"COCO JSON not found: {self.coco_json_path}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_coco_data(self):
        """Load and parse the COCO annotation JSON."""
        print(f"Loading COCO data from: {self.coco_json_path}")
        with open(self.coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        self.coco_images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}

        self.image_id_to_img = {img['id']: img for img in self.coco_images}
        self.image_id_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.image_id_to_anns[ann['image_id']].append(ann)

        self.file_to_img = {img['file_name']: img for img in self.coco_images}

    def group_images_by_base(self):
        """
        Group images by base name (everything before the last underscore in filename).

        Example: test_a_orig.png, test_a_r90.png to group 'test_a'
        """
        print("Grouping images by base name (prefix before last '_')...")

        def extract_base(filename: str) -> str:
            stem, ext = os.path.splitext(filename)
            if '_' in stem:
                return stem.rsplit('_', 1)[0]
            return stem

        # Map base name to image indices
        base_to_indices = defaultdict(list)
        for idx, img in enumerate(self.coco_images):
            base = extract_base(img['file_name'])
            base_to_indices[base].append(idx)

        # Convert to indexed groups
        self.group_idx_to_group = {}
        for idx, (base, indices) in enumerate(base_to_indices.items()):
            self.group_idx_to_group[idx] = [self.coco_images[i] for i in indices]

        print(f"Grouped into {len(self.group_idx_to_group)} base groups.")

        # Map each group to its categories
        self.group_to_cats = defaultdict(set)
        for group_idx, group in self.group_idx_to_group.items():
            img_ids = {img['id'] for img in group}
            for img_id in img_ids:
                for ann in self.image_id_to_anns[img_id]:
                    self.group_to_cats[group_idx].add(ann['category_id'])

        # Map each category to candidate groups
        self.cat_to_candidate_groups = defaultdict(list)
        for group_idx, group in self.group_idx_to_group.items():
            cats = self.group_to_cats[group_idx]
            for cat_id in cats:
                self.cat_to_candidate_groups[cat_id].append((group_idx, group))

        # Sort candidate groups by size (smaller groups prioritized)
        for cat_id in self.cat_to_candidate_groups:
            self.cat_to_candidate_groups[cat_id].sort(key=lambda x: len(x[1]))

    def ensure_category_coverage(self):
        """
        Ensure each category appears in train, val, and test by assigning one group per split.
        """
        print("Phase 1: Ensuring category coverage in all splits...")
        all_cat_ids = set(self.cat_id_to_name.keys())

        for cat_id in all_cat_ids:
            candidates = self.cat_to_candidate_groups[cat_id]

            # Assign to train
            assigned = False
            for idx, _ in candidates:
                if idx not in self.assigned_groups:
                    self.train_groups.add(idx)
                    self.assigned_groups.add(idx)
                    self.cat_counts['train'][cat_id] += 1
                    assigned = True
                    break
            if not assigned:
                print(f"Could not assign category {cat_id} to train")

            # Assign to val
            assigned = False
            for idx, _ in candidates:
                if idx not in self.assigned_groups:
                    self.val_groups.add(idx)
                    self.assigned_groups.add(idx)
                    self.cat_counts['val'][cat_id] += 1
                    assigned = True
                    break
            if not assigned:
                print(f"Could not assign category {cat_id} to val")

            # Assign to test
            assigned = False
            for idx, _ in candidates:
                if idx not in self.assigned_groups:
                    self.test_groups.add(idx)
                    self.assigned_groups.add(idx)
                    self.cat_counts['test'][cat_id] += 1
                    assigned = True
                    break
            if not assigned:
                print(f"Could not assign category {cat_id} to test")

    def assign_remaining_groups(self):
        """
        Assign remaining groups to minimize deviation from 70/20/10 ratio.
        """
        print("Phase 2: Assigning remaining groups to meet 70/20/10 ratio...")
        target_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        remaining = [i for i in self.group_idx_to_group if i not in self.assigned_groups]
        random.shuffle(remaining)

        # Total group count per category
        total_cat_counts = Counter()
        for idx in self.group_idx_to_group:
            for cat_id in self.group_to_cats[idx]:
                total_cat_counts[cat_id] += 1

        def compute_cost(idx: int, split: str) -> float:
            loss = 0.0
            cats = self.group_to_cats[idx]
            for cat_id in cats:
                total = total_cat_counts[cat_id]
                current = self.cat_counts[split][cat_id]
                new = current + 1
                target = target_ratios[split] * total
                error_before = (current - target) ** 2
                error_after = (new - target) ** 2
                loss += (error_after - error_before)
            return loss

        for idx in remaining:
            costs = {sp: compute_cost(idx, sp) for sp in ['train', 'val', 'test']}
            chosen = min(costs, key=costs.get)

            if chosen == 'train':
                self.train_groups.add(idx)
            elif chosen == 'val':
                self.val_groups.add(idx)
            else:
                self.test_groups.add(idx)

            for cat_id in self.group_to_cats[idx]:
                self.cat_counts[chosen][cat_id] += 1

            self.assigned_groups.add(idx)

    def _remap_coco_json(self, image_list):
        """Remap image and annotation IDs to start from 0."""
        new_images = []
        new_annotations = []
        ann_id = 0

        sorted_imgs = sorted(image_list, key=lambda x: x['id'])
        for new_id, img in enumerate(sorted_imgs):
            new_img = img.copy()
            new_img['id'] = new_id
            new_images.append(new_img)

        img_old_to_new = {img['id']: new_id for new_id, img in enumerate(new_images)}

        for img in new_images:
            old_id = self.file_to_img[img['file_name']]['id']
            for ann in self.image_id_to_anns[old_id]:
                new_ann = ann.copy()
                new_ann['id'] = ann_id
                ann_id += 1
                new_ann['image_id'] = img['id']
                new_annotations.append(new_ann)

        return {
            'images': new_images,
            'annotations': new_annotations,
            'categories': self.categories
        }

    def save_splits(self):
        """Save split JSONs and copy images to output folders."""
        splits = {
            'train': self.train_groups,
            'val': self.val_groups,
            'test': self.test_groups
        }

        for name, group_indices in splits.items():
            imgs = [img for idx in group_indices for img in self.group_idx_to_group[idx]]
            coco = self._remap_coco_json(imgs)

            # Save JSON
            json_path = os.path.join(self.output_dir, f'{name}.json')
            with open(json_path, 'w') as f:
                json.dump(coco, f, indent=2)
            print(f"Saved {name}.json with {len(imgs)} images")

            # Copy images
            img_dir = os.path.join(self.output_dir, name, 'images')
            os.makedirs(img_dir, exist_ok=True)
            for img in imgs:
                src = os.path.join(self.image_dir, img['file_name'])
                dst = os.path.join(img_dir, img['file_name'])
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                else:
                    print(f"Source not found: {src}")

    def print_summary(self):
        """Print final statistics and category distribution."""
        print("\n" + "=" * 50)
        print("FINAL SPLIT SUMMARY")
        print("=" * 50)
        tr_imgs = sum(len(self.group_idx_to_group[i]) for i in self.train_groups)
        val_imgs = sum(len(self.group_idx_to_group[i]) for i in self.val_groups)
        test_imgs = sum(len(self.group_idx_to_group[i]) for i in self.test_groups)

        print(f"Train: {len(self.train_groups)} groups to {tr_imgs} images")
        print(f"Val:   {len(self.val_groups)} groups to {val_imgs} images")
        print(f"Test:  {len(self.test_groups)} groups to {test_imgs} images")

        # Check category coverage
        train_cats = set().union(*(self.group_to_cats[i] for i in self.train_groups))
        val_cats = set().union(*(self.group_to_cats[i] for i in self.val_groups))
        test_cats = set().union(*(self.group_to_cats[i] for i in self.test_groups))
        all_cats = set(self.cat_id_to_name.keys())
        missing = all_cats - (train_cats & val_cats & test_cats)

        if missing:
            names = [self.cat_id_to_name[c] for c in missing]
            print(f"Missing in all splits: {names}")
        else:
            print("All categories present in train, val, and test.")

        print("\nCategory Distribution:")
        print(f"{'Class':<15} {'Train':<6} {'Val':<6} {'Test':<6}")
        print("-" * 35)
        for cat_id, name in sorted(self.cat_id_to_name.items()):
            tr = self.cat_counts['train'][cat_id]
            va = self.cat_counts['val'][cat_id]
            te = self.cat_counts['test'][cat_id]
            print(f"{name:<15} {tr:<6} {va:<6} {te:<6}")

    def run(self):
        """Execute the full splitting pipeline."""
        print("Starting COCO dataset split...")
        self.load_coco_data()
        self.group_images_by_base()
        self.ensure_category_coverage()
        self.assign_remaining_groups()
        self.save_splits()
        self.print_summary()
        print(f"\nSplitOptions saved to: {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split COCO dataset into train/val/test with grouped augmentation handling."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file containing: image_dir, coco_json_path, output_dir"
    )

    args = parser.parse_args()

    # Validate config file
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    with open(args.config, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"YAML parse error: {e}")
            sys.exit(1)

    # Required fields
    required = ["image_dir", "coco_json_path", "output_dir"]
    for key in required:
        if key not in config:
            print(f"Missing required key in config: {key}")
            sys.exit(1)

    # Instantiate and run splitter
    try:
        splitter = CocoGroupDatasetSplitter(
            image_dir=config["image_dir"],
            coco_json_path=config["coco_json_path"],
            output_dir=config["output_dir"]
        )
        splitter.run()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
