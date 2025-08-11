#!/usr/bin/env python3
import os
import json
import yaml
import random
import shutil
from collections import defaultdict, Counter
from typing import Dict, List, Set


class CocoDatasetSplitter:
    """
    Splits a COCO dataset into train, validation, and test sets while preserving image groups
    (e.g., original images and their augmentations) in the same split.

    Ensures:
      - Each group of images (sharing the same base name before the last '_') is kept together.
      - Every category appears in train, val, and test splits.
      - Final split ratios are as close as possible to 70% train, 20% val, 10% test.
    """

    def __init__(self, config_path: str):
        """
        Initialize the splitter using a YAML configuration file.

        :param config_path: Path to a YAML file containing:
                            - image_dir: directory containing all images
                            - coco_json_path: path to the COCO annotations JSON file
                            - output_dir: directory where split JSONs and images will be saved
        """
        self.config = self._load_config(config_path)
        self.image_dir = self.config['image_dir']
        self.coco_json_path = self.config['coco_json_path']
        self.output_dir = self.config['output_dir']

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

        # Grouping: base name -> list of images
        self.base_to_images = defaultdict(list)
        self.group_idx_to_group = {}  # index -> list of image dicts
        self.group_to_cats = defaultdict(set)  # group index -> set of category IDs
        self.cat_to_candidate_groups = defaultdict(list)  # cat_id -> list of (group_idx, group)

        # Final group assignments
        self.train_groups = set()  # set of group indices
        self.val_groups = set()
        self.test_groups = set()
        self.assigned_groups = set()  # tracks assigned group indices

        # Category counts per split (for balancing)
        self.cat_counts = {
            'train': Counter(),
            'val': Counter(),
            'test': Counter()
        }

    def _load_config(self, config_path: str) -> Dict[str, str]:
        """
        Load and validate the configuration from a YAML file.

        :param config_path: Path to the YAML config file.
        :return: Dictionary of configuration values.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required = ['image_dir', 'coco_json_path', 'output_dir']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            config[key] = os.path.expanduser(config[key])  # Expand ~
        return config

    def _validate_paths(self):
        """
        Check that input directories and files exist; create output directory if needed.
        """
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isfile(self.coco_json_path):
            raise FileNotFoundError(f"COCO JSON file not found: {self.coco_json_path}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_coco_data(self):
        """
        Load the COCO annotation JSON and populate internal data structures.

        Sets:
            self.coco_data, self.coco_images, self.annotations, self.categories
            self.image_id_to_img, self.image_id_to_anns, self.file_to_img
        """
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
        Group images by their base name (everything before the last underscore in the filename).

        For example:
            'test_a_orig.png', 'test_a_r90.png' → base 'test_a'
        Groups are stored in self.group_idx_to_group, with each group being a list of image dicts.
        """
        print("Grouping images by base name (prefix before last '_')...")

        def extract_base(filename: str) -> str:
            """Extract base name by splitting on the last underscore."""
            stem, ext = os.path.splitext(filename)
            if '_' in stem:
                return stem.rsplit('_', 1)[0]
            return stem

        # Map base name to list of image indices
        base_to_indices = defaultdict(list)
        for idx, img in enumerate(self.coco_images):
            base = extract_base(img['file_name'])
            base_to_indices[base].append(idx)

        # Convert to indexed groups
        self.group_idx_to_group = {}
        for idx, (base, img_indices) in enumerate(base_to_indices.items()):
            self.group_idx_to_group[idx] = [self.coco_images[i] for i in img_indices]

        print(f"Grouped into {len(self.group_idx_to_group)} base groups.")

        # Map each group to the categories it contains
        self.group_to_cats = defaultdict(set)
        for group_idx, group in self.group_idx_to_group.items():
            img_ids = {img['id'] for img in group}
            for img_id in img_ids:
                for ann in self.image_id_to_anns[img_id]:
                    self.group_to_cats[group_idx].add(ann['category_id'])

        # Map each category to candidate groups that contain it
        self.cat_to_candidate_groups = defaultdict(list)
        for group_idx, group in self.group_idx_to_group.items():
            cats = self.group_to_cats[group_idx]
            for cat_id in cats:
                self.cat_to_candidate_groups[cat_id].append((group_idx, group))

        # Sort candidate groups by size (smaller groups prioritized)
        for cat_id in self.cat_to_candidate_groups:
            self.cat_to_candidate_groups[cat_id].sort(key=lambda x: len(x[1]))

        print(f"Annotated categories mapped across groups.")

    def ensure_category_coverage(self):
        """
        Ensure every category appears in train, val, and test by reserving one group per category per split.

        Assigns the smallest available group for each category to train, then val, then test.
        """
        print("Phase 1: Ensuring each category appears in train, val, and test...")
        all_cat_ids = set(self.cat_id_to_name.keys())

        for cat_id in all_cat_ids:
            candidates = self.cat_to_candidate_groups[cat_id]  # Sorted by size

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
        Assign unassigned groups to minimize deviation from 70% train, 20% val, 10% test ratios.

        Uses a greedy cost function based on per-category imbalance.
        """
        print("Phase 2: Assigning remaining groups to meet 70/20/10 ratio...")
        target_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        remaining_groups = [i for i in self.group_idx_to_group if i not in self.assigned_groups]
        random.shuffle(remaining_groups)

        # Total number of groups each category appears in
        total_cat_counts = Counter()
        for idx in self.group_idx_to_group:
            for cat_id in self.group_to_cats[idx]:
                total_cat_counts[cat_id] += 1

        def compute_cost(idx: int, split: str) -> float:
            """
            Compute cost of assigning group `idx` to `split` based on category imbalance.
            Lower cost = better for ratio.
            """
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

        for idx in remaining_groups:
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

    def _remap_coco_json(self, image_list: List[Dict]) -> Dict:
        """
        Remap image and annotation IDs to be contiguous starting from 0.

        :param image_list: List of image dicts to include in the new JSON.
        :return: New COCO-format dictionary with remapped IDs.
        """
        new_images = []
        new_annotations = []
        ann_id_counter = 0

        # Sort by original ID for determinism
        sorted_images = sorted(image_list, key=lambda x: x['id'])

        # Remap image IDs
        img_id_map = {}
        for new_id, img in enumerate(sorted_images):
            old_id = img['id']
            img_id_map[old_id] = new_id
            new_img = img.copy()
            new_img['id'] = new_id
            new_images.append(new_img)

        # Remap annotation IDs and update image_id references
        for img in new_images:
            old_id = self.file_to_img[img['file_name']]['id']
            for ann in self.image_id_to_anns[old_id]:
                new_ann = ann.copy()
                new_ann['id'] = ann_id_counter
                ann_id_counter += 1
                new_ann['image_id'] = img['id']
                new_annotations.append(new_ann)

        return {
            'images': new_images,
            'annotations': new_annotations,
            'categories': self.categories
        }

    def save_splits(self):
        """
        Save train/val/test splits as COCO JSONs and copy corresponding images to output folders.
        """
        splits = {
            'train': self.train_groups,
            'val': self.val_groups,
            'test': self.test_groups
        }

        # Extract full image lists and save
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
        """
        Print a final summary of the split: group/image counts and category distribution.
        """
        print("\n" + "="*50)
        print("FINAL SPLIT SUMMARY")
        print("="*50)
        total_train_imgs = sum(len(self.group_idx_to_group[i]) for i in self.train_groups)
        total_val_imgs = sum(len(self.group_idx_to_group[i]) for i in self.val_groups)
        total_test_imgs = sum(len(self.group_idx_to_group[i]) for i in self.test_groups)

        print(f"Train: {len(self.train_groups)} groups → {total_train_imgs} images")
        print(f"Val:   {len(self.val_groups)} groups → {total_val_imgs} images")
        print(f"Test:  {len(self.test_groups)} groups → {total_test_imgs} images")

        # Verify all categories are in all splits
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
        """
        Execute the full dataset splitting pipeline.
        """
        print("Starting COCO dataset split...")
        self.load_coco_data()
        self.group_images_by_base()
        self.ensure_category_coverage()
        self.assign_remaining_groups()
        self.save_splits()
        self.print_summary()
        print(f"\nSplitOptions saved to: {self.output_dir}")


# === CLI Usage ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment COCO dataset with geometric transforms.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file (default: config.yaml)')

    args = parser.parse_args()

    augmenter = CocoDatasetSplitter(config_path=args.config)
    augmenter.augment()