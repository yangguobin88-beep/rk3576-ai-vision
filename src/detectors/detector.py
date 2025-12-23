"""
统一检测器接口 - 支持 ONNX 和 RKNN

自动识别模型格式，提供统一的 detect() 接口
"""
from __future__ import annotations

import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from ..common.preprocess import preprocess_with_letterbox, restore_coords
from ..common.postprocess import yolov8_postprocess, get_class_name
from ..common.config import MODEL_INPUT_SIZE, OBJ_THRESH, NMS_THRESH
from ..common.logger import zlog


class BaseDetector(ABC):
    """检测器抽象基类"""
    
    def __init__(self, obj_thresh=None, nms_thresh=None):
        self.obj_thresh = obj_thresh if obj_thresh is not None else OBJ_THRESH
        self.nms_thresh = nms_thresh if nms_thresh is not None else NMS_THRESH
        self.input_size = MODEL_INPUT_SIZE
        self._scale = 1.0
        self._pad = (0, 0)
    
    @abstractmethod
    def _inference(self, img_input):
        """子类实现：执行模型推理"""
        pass
    
    @abstractmethod
    def release(self):
        """子类实现：释放资源"""
        pass
    
    def detect(self, img: np.ndarray) -> tuple[
        np.ndarray | None, 
        np.ndarray | None, 
        np.ndarray | None, 
        list[str] | None
    ]:
        """
        统一检测接口
        
        Args:
            img: BGR 图像 (H, W, 3)
        
        Returns:
            boxes: 检测框 [[x1,y1,x2,y2], ...]（原图坐标）
            classes: 类别 ID
            scores: 置信度
            names: 类别名称
        """
        # 预处理
        img_input, self._scale, self._pad = preprocess_with_letterbox(img, self.input_size)
        
        # 推理
        outputs = self._inference(img_input)
        
        # 后处理
        boxes, classes, scores = yolov8_postprocess(
            outputs, self.obj_thresh, self.nms_thresh, self.input_size
        )
        
        # 坐标还原
        if boxes is not None:
            boxes = restore_coords(boxes, self._scale, self._pad)
            names = [get_class_name(int(c)) for c in classes]
            return boxes, classes, scores, names
        
        return None, None, None, None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()


class ONNXDetector(BaseDetector):
    """ONNX 模型检测器（PC 端开发用）"""
    
    def __init__(self, model_path, obj_thresh=None, nms_thresh=None):
        super().__init__(obj_thresh, nms_thresh)
        
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = model_path
        zlog.info(f"[ONNX] 加载模型: {model_path}")
    
    def _inference(self, img_input):
        """ONNX 推理"""
        # ONNX 需要 NCHW + float32 + 归一化
        img = img_input.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape).astype(np.float32) / 255.0
        outputs = self.session.run(None, {self.input_name: img})
        return outputs
    
    def release(self):
        """释放资源"""
        self.session = None


class RKNNDetector(BaseDetector):
    """RKNN 模型检测器（板端部署用）"""
    
    def __init__(self, model_path, obj_thresh=None, nms_thresh=None, core_mask=None):
        super().__init__(obj_thresh, nms_thresh)
        
        # 自动识别 PC 还是板端
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            self.is_lite = True
            zlog.info(f"[RKNN-Lite] 加载模型: {model_path}")
        except ImportError:
            from rknn.api import RKNN
            self.rknn = RKNN()
            self.is_lite = False
            zlog.info(f"[RKNN] 加载模型: {model_path}")
        
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载模型失败: {model_path}")
        
        if self.is_lite and core_mask is not None:
            ret = self.rknn.init_runtime(core_mask=core_mask)
        else:
            ret = self.rknn.init_runtime()
        
        if ret != 0:
            raise RuntimeError("初始化运行时失败")
        
        self.model_path = model_path
    
    def _inference(self, img_input):
        """RKNN 推理"""
        # RKNN 直接用 uint8 HWC 格式
        outputs = self.rknn.inference(inputs=[img_input])
        return outputs
    
    def release(self):
        """释放资源"""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


def create_detector(
    model_path: str, 
    obj_thresh: Optional[float] = None, 
    nms_thresh: Optional[float] = None, 
    core_mask: Optional[int] = None
) -> BaseDetector:
    """
    工厂函数：根据模型后缀自动创建对应的检测器
    
    Args:
        model_path: 模型路径（.onnx 或 .rknn）
        obj_thresh: 置信度阈值
        nms_thresh: NMS 阈值
        core_mask: NPU 核心掩码（仅 RKNN 板端有效）
    
    Returns:
        detector: ONNXDetector 或 RKNNDetector 实例
    """
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext == '.onnx':
        return ONNXDetector(model_path, obj_thresh, nms_thresh)
    elif ext == '.rknn':
        return RKNNDetector(model_path, obj_thresh, nms_thresh, core_mask)
    else:
        raise ValueError(f"不支持的模型格式: {ext}，支持 .onnx 和 .rknn")
