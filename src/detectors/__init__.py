# detectors/__init__.py
from .base_model import BaseRKNNModel
from .detector import create_detector, BaseDetector, ONNXDetector, RKNNDetector
from .yolo_detector import YOLOv8Detector
