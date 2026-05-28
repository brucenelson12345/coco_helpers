# Import the new analyzer at the top
from devai.analyzer import SegmentationAnalyzer

# ... inside evaluate_accuracy() ...

        elif self.model_type in [ModelType.SEGMENTATION]:
            analyzer = SegmentationAnalyzer(
                data_loader=data_loader,
                iou_threshold=0.5,
                nms_iou_threshold=self.nms_iou_threshold,
                confidence_min=self.confidence_min,
                box_format=self.box_format,
                output_tensors=self.output_tensors,
                output_dir=self.qairt_output_dir,
                datatype=self.convert_datatype,
            )
            # If standard bounding box AP is sufficient for now:
            if self.input_annotations:
                ap = analyzer.qairt_evaluate() 
                logger.info("Measured Box AP0.5 is %f", ap)
            if self.save_results:
                analyzer.draw_segmentation_onto_input_images()