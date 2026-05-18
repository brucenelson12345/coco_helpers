import ssl
import os
import yaml
from pathlib import Path

# Globally bypass SSL verification for automatic weight downloads
ssl._create_default_https_context = ssl._create_unverified_context

class RFDETRTrainer:
    """
    A unified trainer for RF-DETR Detection and Segmentation models.
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.dataset_dir = Path(self.config.get('dataset_dir'))
        self.task = self.config.get('task', 'detection').lower()
        self.model_size = self.config.get('model_size', 'small').lower()
        self.weights = self.config.get('weights')
        self.epochs = self.config.get('epochs', 50)
        self.batch_size = self.config.get('batch_size', 4)
        self.grad_accum_steps = self.config.get('grad_accum_steps', 4)
        self.output_dir = self.config.get('output_dir', './rfdetr_output')
        self.resolution = self.config.get('resolution', 640)

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def _prepare_dataset(self):
        """
        Ensures the 'valid' directory exists, symlinking 'val' if necessary.
        """
        val_path = self.dataset_dir / 'val'
        valid_path = self.dataset_dir / 'valid'

        if val_path.is_dir() and not valid_path.exists():
            print(f"Mapping 'val' -> 'valid' for compatibility...")
            try:
                # Use relative symlink for portability
                os.symlink('val', valid_path)
            except OSError as e:
                print(f"Warning: Could not create symlink (check permissions): {e}")
        elif not valid_path.exists():
            print(f"Warning: Neither 'valid' nor 'val' found in {self.dataset_dir}")

    def _get_model(self):
        """
        Dynamically imports and initializes the model based on task and size.
        """
        # Dictionary mapping for clean lookups
        model_map = {
            "detection": {
                "nano": "RFDETRNano", "small": "RFDETRSmall", 
                "medium": "RFDETRMedium", "large": "RFDETRLarge", "base": "RFDETRBase"
            },
            "segmentation": {
                "nano": "RFDETRSegNano", "small": "RFDETRSegSmall", 
                "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge",
                "xlarge": "RFDETRSegXLarge", "2xlarge": "RFDETRSeg2XLarge"
            }
        }

        try:
            from rfdetr import __dict__ as rf_dict
            class_name = model_map[self.task].get(self.model_size, "RFDETRBase")
            ModelClass = rf_dict.get(class_name)
            
            if ModelClass is None:
                # Fallback manual import if __dict__ isn't populated as expected
                import rfdetr
                ModelClass = getattr(rfdetr, class_name)

        except (ImportError, AttributeError, KeyError) as e:
            raise ValueError(f"Unsupported task '{self.task}' or size '{self.model_size}': {e}")

        # Initializing without resolution parameter to avoid state_dict mismatch
        if self.weights and os.path.exists(self.weights):
            print(f"Loading custom weights from {self.weights}")
            return ModelClass(pretrain_weights=self.weights)
        
        print(f"Loading pre-trained {class_name}...")
        return ModelClass()

    def train(self):
        """
        Executes the training pipeline.
        """
        self._prepare_dataset()
        model = self._get_model()

        print("-" * 30)
        print(f"TASK:       {self.task.upper()}")
        print(f"MODEL:      {self.model_size}")
        print(f"RESOLUTION: {self.resolution}")
        print(f"BATCH SIZE: {self.batch_size} (Accum: {self.grad_accum_steps})")
        print("-" * 30)

        model.train(
            dataset_dir=str(self.dataset_dir),
            epochs=self.epochs,
            batch_size=self.batch_size,
            grad_accum_steps=self.grad_accum_steps,
            resolution=self.resolution,
            output_dir=self.output_dir
        )
        print(f"Training complete. Results saved to {self.output_dir}")

# --- Execution Block ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RF-DETR Class-based Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    trainer = RFDETRTrainer(args.config)
    trainer.train()