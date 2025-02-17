

import os
import numpy as np
from PIL import Image
from yolov4 import Detector
from sahi.prediction import ObjectPrediction
from sahi.models.base import DetectionModel
import configparser


class YOLOv4TinyDetectionModel(DetectionModel):
    def load_model(self):
        """
        Loads YOLOv4 Tiny model using the Detector class (avoiding CLI calls).
        """
        self.config_path = self.config_path 
        self.weights_path = self.model_path 

        self.data_path = self._get_config_value('Paths', 'data_path')
        self.lib_darknet_path = self._get_config_value('Paths', 'lib_darknet_path')
        self.detector = Detector(
            config_path=self.config_path,
            weights_path=self.weights_path,
            meta_path=self.data_path,
            lib_darknet_path=self.lib_darknet_path,
        )

    def _get_config_value(self, section, option):
        """
        Reads a value from the config file.
        """
        config = configparser.ConfigParser()
        config.read('config.ini')  # Path to your config file
        return config.get(section, option)
    
    def perform_inference(self, image: np.ndarray):
        """
        Performs inference using the Detector class.
        Args:
            image: np.ndarray
                Input image as a numpy array.
        """
        original_height, original_width = image.shape[:2]
        network_width, network_height = self.detector.network_width(), self.detector.network_height()

        img = Image.fromarray(image)
        img_resized = img.resize((network_width, network_height))
        img_arr = np.array(img_resized)

        detections = self.detector.perform_detect(image_path_or_buf=img_arr, show_image=False,thresh=self.confidence_threshold)
        
        self._original_predictions = [
            {
                "class_name": det.class_name,
                "confidence": det.class_confidence,
                "bbox": {
                    "x_min": (det.left_x / network_width) * original_width,
                    "y_min": (det.top_y / network_height) * original_height,
                    "x_max": ((det.left_x + det.width) / network_width) * original_width,
                    "y_max": ((det.top_y + det.height) / network_height) * original_height,
                },
            }
            for det in detections
        ]

    def _create_object_prediction_list_from_original_predictions(
        self, shift_amount_list=None, full_shape_list=None
    ):
        """
        Converts YOLO predictions to SAHI-compatible ObjectPrediction list.
        """
        object_predictions = []
        for detection in self._original_predictions:
            shift_x,shift_y=shift_amount_list
            bbox = detection["bbox"]
            confidence = detection["confidence"]
            category_name = detection["class_name"]
            bbox=[bbox["x_min"]-shift_x, bbox["y_min"]-shift_y, bbox["x_max"]-shift_x, bbox["y_max"]-shift_y]
            # fix negative box coords
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = max(0, bbox[2])
            bbox[3] = max(0, bbox[3])

            # fix out of image box coords
            if full_shape_list is not None:
                bbox[0] = min(full_shape_list[1], bbox[0])
                bbox[1] = min(full_shape_list[0], bbox[1])
                bbox[2] = min(full_shape_list[1], bbox[2])
                bbox[3] = min(full_shape_list[0], bbox[3])


            object_prediction = ObjectPrediction(
                bbox=bbox,
                score=confidence,
                category_id=0,  # No class_id available, setting default to 0
                category_name=category_name,
            )
            object_predictions.append(object_prediction)

        self._object_prediction_list_per_image = [object_predictions]

