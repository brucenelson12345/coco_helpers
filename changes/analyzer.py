class SegmentationAnalyzer(DetectionAnalyzer):
    def batch_nms_for_segmentation(self):
        """Prep segmentation output for analysis, catching masks alongside boxes."""
        box_tensors_map, score_tensors_map, label_tensors_map, mask_tensors_map = {}, {}, {}, {}
        
        # Logic to extract the mask tensor. This will depend on the exact naming 
        # convention of your RF-DETR ONNX export outputs.
        for i in range(len(self.output_tensors)):
            tensor_name = self.output_tensors[i][NAME]
            if "box" in tensor_name or 4 in self.output_tensors[i][SHAPE]:
                box_tensors_map = self.vectors[tensor_name].copy()
            elif "class" in tensor_name or "label" in tensor_name:
                label_tensors_map = self.vectors[tensor_name].copy()
            elif "mask" in tensor_name:
                mask_tensors_map = self.vectors[tensor_name].copy()
            elif "score" in tensor_name or "logit" in tensor_name:
                score_tensors_map = self.vectors[tensor_name].copy()

        # Execute batched NMS, passing the mask map as the additional arg
        for key in box_tensors_map.keys():
            boxes_out, scores_out, labels_out, masks_out = batched_nms(
                self.nms_iou_threshold,
                self.confidence_min,
                self.data_loader.width,
                self.data_loader.height,
                box_tensors_map[key].copy(),
                score_tensors_map[key].copy(),
                self.box_format,
                label_tensors_map[key].round().copy(),
                mask_tensors_map[key].copy()  # Mask tensor passed here
            )
            
            box_tensors_map[key] = boxes_out
            score_tensors_map[key] = scores_out
            label_tensors_map[key] = labels_out
            mask_tensors_map[key] = masks_out
            
        return box_tensors_map, score_tensors_map, label_tensors_map, mask_tensors_map

    def draw_segmentation_onto_input_images(self):
        if not self.vectors:
            self.load_snpe_output_vectors()
            
        box_tensors, score_tensors, label_tensors, mask_tensors = self.batch_nms_for_segmentation()
        
        if score_tensors:
            for key, image_meta in zip(box_tensors.keys(), self.data_loader.batch_meta):
                image_meta.draw_segmentations_and_save_image(
                    box_tensors[key].pop(),
                    score_tensors[key].pop(),
                    label_tensors[key].pop(),
                    mask_tensors[key].pop(),
                    self.data_loader.model_labels,
                    self.data_loader.categories,
                    self.data_loader.data_dir,
                    self.data_loader.width,
                    self.data_loader.height,
                    self.output_dir,
                    str(key) + ".jpg",
                )
            logger.info("Output segmented images saved to %s", os.path.abspath(self.output_dir))