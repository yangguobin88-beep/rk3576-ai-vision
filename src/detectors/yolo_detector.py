"""
YOLOv8 检测器 - 继承 BaseRKNNModel
"""
import cv2
import numpy as np
from .base_model import BaseRKNNModel
from ..common.preprocess import preprocess_with_letterbox, restore_coords
from ..common.postprocess import yolov8_postprocess, get_class_name
from ..common.config import MODEL_INPUT_SIZE, OBJ_THRESH, NMS_THRESH


class YOLOv8ModelDetector(BaseRKNNModel):
    """YOLOv8 目标检测器"""
    
    def __init__(self, model_path, core_mask=None, obj_thresh=None, nms_thresh=None):
        """
        初始化 YOLOv8 检测器
        
        Args:
            model_path: RKNN 模型路径
            core_mask: NPU 核心掩码（仅板端有效）
            obj_thresh: 置信度阈值
            nms_thresh: NMS 阈值
        """
        super().__init__(model_path, core_mask)
        self.obj_thresh = obj_thresh if obj_thresh is not None else OBJ_THRESH
        self.nms_thresh = nms_thresh if nms_thresh is not None else NMS_THRESH
        self.input_size = MODEL_INPUT_SIZE
        
        # 缓存 letterbox 参数，用于坐标还原
        self._scale = 1.0
        self._pad = (0, 0)
    
    def preprocess(self, img):
        """预处理：letterbox + BGR→RGB"""
        img_input, self._scale, self._pad = preprocess_with_letterbox(img, self.input_size)
        return img_input
    
    def postprocess(self, outputs):
        """后处理：解析检测结果"""
        boxes, classes, scores = yolov8_postprocess(
            outputs, 
            obj_thresh=self.obj_thresh, 
            nms_thresh=self.nms_thresh,
            img_size=self.input_size
        )
        
        # 坐标还原到原图
        if boxes is not None:
            boxes = restore_coords(boxes, self._scale, self._pad)
        
        return boxes, classes, scores
    
    def detect(self, img):
        """
        检测接口（带可视化信息）
        
        Returns:
            boxes: 检测框 [[x1,y1,x2,y2], ...]
            classes: 类别 ID
            scores: 置信度
            names: 类别名称列表
        """
        boxes, classes, scores = self.infer(img)
        
        if boxes is None:
            return None, None, None, None
        
        names = [get_class_name(int(c)) for c in classes]
        return boxes, classes, scores, names
    
    def draw_results(self, img, boxes, classes, scores, names=None):
        """在图像上绘制检测结果"""
        if boxes is None:
            return img
        
        img_draw = img.copy()
        
        for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
            x1, y1, x2, y2 = map(int, box)
            name = names[i] if names else get_class_name(int(cls))
            
            # 不同类别用不同颜色
            color = self._get_color(int(cls))
            
            # 画框
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # 画标签背景
            label = f"{name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # 画标签文字
            cv2.putText(img_draw, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return img_draw
    
    def _get_color(self, class_id):
        """根据类别 ID 生成颜色"""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
