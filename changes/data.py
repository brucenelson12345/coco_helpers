from cv2 import resize, INTER_NEAREST
from .draw_image import draw_segmentation_mask

# ... [Inside InputMeta class] ...

def draw_segmentations_and_save_image(
    self,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    pred_masks: np.ndarray, # New parameter for the mask tensor
    model_labels: Dict[int, str],
    annotation_labels: Dict[int, str],
    input_image_dir: str,
    model_width: int,
    model_height: int,
    output_dir: str,
    output_file_name: str,
    mask_threshold: float = 0.5
):
    ratio_x = self.size[0] / model_width
    ratio_y = self.size[1] / model_height
    pred_boxes[:, 0] *= ratio_x
    pred_boxes[:, 1] *= ratio_y
    pred_boxes[:, 2] *= ratio_x
    pred_boxes[:, 3] *= ratio_y

    if self.image_name.endswith(".raw"):
        try:
            output_image = np.fromfile(
                os.path.join(input_image_dir, self.image_name), dtype=np.uint8
            ).reshape(list(self.size) + [3])
        except ValueError:
            output_image = np.zeros(list(self.size) + [3], dtype=np.uint8)
    else:
        output_image = imread(os.path.join(input_image_dir, self.image_name))

    for idx in range(len(pred_boxes)):
        if pred_labels[idx] in list(model_labels.keys()):
            # 1. Process and draw the mask
            # Assuming mask shape is (model_height, model_width) or similar after extraction
            mask = pred_masks[idx]
            
            # Resize the mask back to the original image size using Nearest Neighbor to keep binary edges sharp
            mask_resized = resize(mask, (self.size[0], self.size[1]), interpolation=INTER_NEAREST)
            binary_mask = mask_resized > mask_threshold
            
            # Use a static color or generate one based on the label index
            mask_color = (0, 255, 0) 
            draw_segmentation_mask(output_image, binary_mask, color=mask_color, alpha=0.4)

            # 2. Draw the bounding box on top
            draw_box_from_xyxy(
                output_image,
                pred_boxes[idx, 0:2],
                pred_boxes[idx, 2:4],
                color=(0, 255, 0),
                size=2,
                text=f"{model_labels[pred_labels[idx]]} {np.round(float(pred_scores[idx]), decimals=3)}",
            )
            
    out = Image.fromarray(output_image[:, :, ::-1])  # RGB to BGR
    save_image(out, output_dir, output_file_name, "Output Image")