import argparse
import os
import yaml
import cv2
from pathlib import Path

class RFDETRInference:
    """
    Inference engine for RF-DETR Detection and Segmentation models.
    """
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model_path = self.config.get('model_path')
        self.task = self.config.get('task', 'detection').lower()
        self.model_size = self.config.get('model_size', 'small').lower()
        self.image_path = self.config.get('image_path')
        self.confidence = self.config.get('confidence', 0.25)
        self.output_path = self.config.get('output_path', './result.jpg')

        # Load the model upon initialization
        self.model = self._load_model()

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def _load_model(self):
        """
        Initializes the architecture and loads trained weights.
        """
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

        # Dynamic import from the rfdetr package
        import rfdetr
        class_name = model_map[self.task].get(self.model_size, "RFDETRBase")
        ModelClass = getattr(rfdetr, class_name)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained weights not found at: {self.model_path}")

        print(f"Loading {class_name} with weights from {self.model_path}...")
        # We load the weights directly into the constructor
        return ModelClass(pretrain_weights=self.model_path)

    def run(self):
        """
        Runs inference and saves the visualized result.
        """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        print(f"Running inference on: {self.image_path} (Conf: {self.confidence})")

        # The .predict() method handles preprocessing, inference, and visualization
        # It returns a result object that typically includes detections/masks
        results = self.model.predict(
            source=self.image_path,
            conf=self.confidence
        )

        # rfdetr's results object can save the plotted image directly
        # If task is segmentation, it will automatically include masks
        results.save(self.output_path)
        
        print(f"Inference complete. Result saved to: {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR Inference Script")
    parser.add_argument("--config", type=str, required=True, help="Path to inference_config.yaml")
    args = parser.parse_args()

    inferencer = RFDETRInference(args.config)
    inferencer.run()