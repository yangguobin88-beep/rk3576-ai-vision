"""
YOLO 后处理模块 - YOLOv8 完整实现（纯 numpy，无 torch 依赖）
参考：rknn_model_zoo/examples/yolov8/python/yolov8.py
"""
import numpy as np
from .config import MODEL_INPUT_SIZE, OBJ_THRESH, NMS_THRESH, COCO_CLASSES


def _softmax(x, axis=2):
    """Numpy 实现的 softmax"""
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def _dfl(position):
    """
    Distribution Focal Loss (DFL) - 解码边界框
    ✅ 纯 numpy 实现，板端友好
    """
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    
    # softmax (numpy 实现)
    y = _softmax(y, axis=2)
    
    # 加权求和
    acc_metrix = np.arange(mc).astype(np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y * acc_metrix, axis=2)
    return y


def _box_process(position, img_size=None):
    """将模型输出转换为 xyxy 坐标"""
    if img_size is None:
        img_size = MODEL_INPUT_SIZE
    
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size[1] // grid_h, img_size[0] // grid_w]).reshape(1, 2, 1, 1)

    position = _dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def _filter_boxes(boxes, box_confidences, box_class_probs, obj_thresh=None):
    """过滤低置信度的检测框"""
    if obj_thresh is None:
        obj_thresh = OBJ_THRESH
    
    box_confidences = box_confidences.reshape(-1)
    
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= obj_thresh)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms(boxes, scores, iou_threshold=None):
    """非极大值抑制 (NMS)"""
    if iou_threshold is None:
        iou_threshold = NMS_THRESH
    
    if len(boxes) == 0:
        return np.array([])
    
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def yolov8_postprocess(outputs, obj_thresh=None, nms_thresh=None, img_size=None):
    """
    YOLOv8 后处理完整实现（纯 numpy，板端友好）
    
    Args:
        outputs: 模型输出（多个 tensor）
        obj_thresh: 置信度阈值，默认使用 config.OBJ_THRESH
        nms_thresh: NMS 阈值，默认使用 config.NMS_THRESH
        img_size: 输入图像尺寸，默认使用 config.MODEL_INPUT_SIZE
    
    Returns:
        boxes: 检测框坐标 [[x1,y1,x2,y2], ...]
        classes: 类别 ID [0, 5, ...]
        scores: 置信度 [0.88, 0.85, ...]
    """
    if obj_thresh is None:
        obj_thresh = OBJ_THRESH
    if nms_thresh is None:
        nms_thresh = NMS_THRESH
    if img_size is None:
        img_size = MODEL_INPUT_SIZE
    
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(outputs) // default_branch

    # 处理 3 个不同尺度的输出
    for i in range(default_branch):
        boxes.append(_box_process(outputs[pair_per_branch * i], img_size))
        classes_conf.append(outputs[pair_per_branch * i + 1])
        scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 过滤低置信度
    boxes, classes, scores = _filter_boxes(boxes, scores, classes_conf, obj_thresh)

    # 按类别分别进行 NMS
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        cls = classes[inds]
        s = scores[inds]
        keep = nms(b, s, nms_thresh)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(cls[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def yolov5_postprocess(outputs, conf_threshold=0.5, iou_threshold=0.45):
    """YOLOv5 后处理（anchor-based）- 待实现"""
    raise NotImplementedError("YOLOv5 后处理待实现，请参考 rknn_model_zoo 示例")


def get_class_name(class_id):
    """根据类别 ID 获取类别名称"""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"
