"""
统一预处理模块 - PC/板端通用
"""
import cv2
import numpy as np
from .config import MODEL_INPUT_SIZE


def preprocess(img, target_size=None):
    """
    统一预处理：BGR→RGB + resize
    
    Args:
        img: OpenCV 读取的 BGR 图像
        target_size: 目标尺寸，默认使用 config.MODEL_INPUT_SIZE
    
    Returns:
        处理后的 RGB 图像，uint8 格式
    
    注意：
        - ⚠️ 不做 normalize，RKNN 模型内部处理
        - ⚠️ 保持 uint8，不转 float32
    """
    if target_size is None:
        target_size = MODEL_INPUT_SIZE
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.uint8)
    return img


def preprocess_with_letterbox(img, target_size=None, color=(0, 0, 0)):
    """
    带 letterbox 的预处理（保持宽高比）
    
    Args:
        img: OpenCV 读取的 BGR 图像
        target_size: 目标尺寸，默认使用 config.MODEL_INPUT_SIZE
        color: 填充颜色，默认黑色 (0,0,0)
    
    Returns:
        img_padded: 处理后的 RGB 图像
        scale: 缩放比例
        pad: 填充偏移 (pad_x, pad_y)
    """
    if target_size is None:
        target_size = MODEL_INPUT_SIZE
    
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    img_resized = cv2.resize(img, (new_w, new_h))
    img_padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    
    return img_padded, scale, (pad_x, pad_y)


def restore_coords(boxes, scale, pad):
    """
    将 letterbox 坐标还原到原图坐标
    
    Args:
        boxes: 检测框 [[x1,y1,x2,y2], ...]
        scale: letterbox 缩放比例
        pad: letterbox 填充偏移 (pad_x, pad_y)
    
    Returns:
        还原后的检测框坐标
    """
    if boxes is None:
        return None
    
    pad_x, pad_y = pad
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes
