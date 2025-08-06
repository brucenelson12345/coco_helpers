# coco_splitter.py

import os
import json
import yaml
import random
import shutil
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any

class CocoDatasetSplitter:
    """
    Splits a COCO dataset into train/val/test sets with:
    - Grouped image handling (original + augmentations stay together)
    - Each category present in all splits
    - Stratified assignment toward 70/20/10 ratio
    """

    def __init__(self, config_path: str):
        """
        Initialize splitter from a YAML config file.
        :param config_path: Path to YAML config
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

        # Grouping
        self.base_to_images = defaultdict(list)
        self.group_idx_to_group = {}
        self.group_to_cats = defaultdict(set)
        self.cat_to_candidate_groups = defaultdict(list)

        # Assignment
        self.train_groups = set()
        self.val_groups = set()
        self.test_groups = set()
        self.assigned_groups = set()
        self.cat_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    def _load_config(self, config_path: str) -> Dict[str, str]:
        """Load and validate config from YAML."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required = ['image_dir', 'coco_json_path', 'output_dir']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
            config[key] = os.path.expanduser(config[key])
        return config

    def _validate_paths(self):
        """Validate input paths."""
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isfile(self.coco_json_path):
            raise FileNotFoundError(f"COCO JSON not found: {self.coco_json_path}")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_coco_data(self):
        """Load and parse COCO JSON."""
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

    @staticmethod
    def extract_base_name(filename: str) -> str:
        """Extract base image name (without augmentation suffix)."""
        name, ext = os.path.splitext(filename)
        if '_' in name:
            candidate = name.rsplit('_', 1)[0]
            if os.path.exists(os.path.join(os.path.dirname(filename), candidate + ext)):
                return candidate
        return name

    def group_images_by_base(self):
        """Group images by base name (original + augmentations)."""
        print("Grouping images by base name...")
        for img in self.coco_images:
            fname = img['file_name']
            base = self.extract_base_name(fname)
            self.base_to_images[base].append(img)

        image_groups = list(self.base_to_images.values())
        self.group_idx_to_group = {i: group for i, group in enumerate(image_groups)}

        # Build group_to_cats and cat_to_candidate_groups
        for idx, group in self.group_idx_to_group.items():
            img_ids = {img['id'] for img in group}
            cats = set()
            for img_id in img_ids:
                for ann in self.image_id_to_anns[img_id]:
                    cats.add(ann['category_id'])
            self.group_to_cats[idx] = cats
            for cat_id in cats:
                self.cat_to_candidate_groups[cat_id].append((idx, group))

        # Sort candidate groups by size (smallest first)
        for cat_id in self.cat_to_candidate_groups:
            self.cat_to_candidate_groups[cat_id].sort(key=lambda x: len(x[1]))

        print(f"Total image groups: {len(image_groups)}")

    def ensure_category_coverage(self):
        """
        Phase 1: Assign one group per category to train, val, and test
        to ensure every category appears in all splits.
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
                print(f"⚠️  Could not assign category {cat_id} to train")

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
                print(f"⚠️  Could not assign category {cat_id} to val")

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
                print(f"⚠️  Could not assign category {cat_id} to test")

    def assign_remaining_groups(self):
        """
        Phase 2: Assign remaining groups to minimize deviation from 70/20/10 ratio.
        """
        print("Phase 2: Assigning remaining groups to meet 70/20/10 ratio...")
        target_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}
        remaining_groups = [i for i in self.group_idx_to_group if i not in self.assigned_groups]
        random.shuffle(remaining_groups)

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
        """Remap image and annotation IDs to start from 0."""
        new_images = []
        new_annotations = []
        ann_id_counter = 0

        sorted_images = sorted(image_list, key=lambda x: x['id'])
        img_id_map = {}

        for new_id, img in enumerate(sorted_images):
            old_id = img['id']
            img_id_map[old_id] = new_id
            new_img = img.copy()
            new_img['id'] = new_id
            new_images.append(new_img)

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
        """Save train/val/test JSONs and copy images."""
        splits = {
            'train': self.train_groups,
            'val': self.val_groups,
            'test': self.test_groups
        }

        # Extract full image lists
        image_lists = {}
        for name, group_indices in splits.items():
            imgs = [img for idx in group_indices for img in self.group_idx_to_group[idx]]
            image_lists[name] = imgs

            # Save JSON
            coco = self._remap_coco_json(imgs)
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
                    print(f"⚠️  Source not found: {src}")

    def print_summary(self):
        """Print final split statistics."""
        print("\n" + "="*50)
        print("✅ FINAL SPLIT SUMMARY")
        print("="*50)
        print(f"Train: {len(self.train_groups)} groups → {sum(len(self.group_idx_to_group[i]) for i in self.train_groups)} images")
        print(f"Val:   {len(self.val_groups)} groups → {sum(len(self.group_idx_to_group[i]) for i in self.val_groups)} images")
        print(f"Test:  {len(self.test_groups)} groups → {sum(len(self.group_idx_to_group[i]) for i in self.test_groups)} images")

        # Verify coverage
        train_cats = set().union(*(self.group_to_cats[i] for i in self.train_groups))
        val_cats = set().union(*(self.group_to_cats[i] for i in self.val_groups))
        test_cats = set().union(*(self.group_to_cats[i] for i in self.test_groups))
        all_cats = set(self.cat_id_to_name.keys())
        missing = all_cats - (train_cats & val_cats & test_cats)

        if missing:
            names = [self.cat_id_to_name[c] for c in missing]
            print(f"❌ Missing in all splits: {names}")
        else:
            print("✅ All categories present in train, val, and test.")

        print("\n📊 Category Distribution:")
        print(f"{'Class':<15} {'Train':<6} {'Val':<6} {'Test':<6}")
        print("-" * 35)
        for cat_id, name in sorted(self.cat_id_to_name.items()):
            tr = self.cat_counts['train'][cat_id]
            va = self.cat_counts['val'][cat_id]
            te = self.cat_counts['test'][cat_id]
            print(f"{name:<15} {tr:<6} {va:<6} {te:<6}")

    def run(self):
        """Execute full pipeline."""
        print("Starting COCO dataset split...")
        self.load_coco_data()
        self.group_images_by_base()
        self.ensure_category_coverage()
        self.assign_remaining_groups()
        self.save_splits()
        self.print_summary()
        print(f"\nSplitOptions saved to: {self.output_dir}")