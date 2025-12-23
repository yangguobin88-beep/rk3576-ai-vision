"""
全局配置模块 - 统一管理项目参数

⚠️ 本文件为全局配置，修改参数前请确认对整体系统影响
⚠️ 预处理/后处理依赖同一份参数，请勿单独修改
"""

# ==================== 模型输入配置 ====================

# 模型输入尺寸 (width, height)
# ⚠️ 预处理和后处理必须使用相同的尺寸
MODEL_INPUT_SIZE = (640, 640)

# ==================== YOLOv8 检测配置 ====================

# 置信度阈值：低于此值的检测框被丢弃
OBJ_THRESH = 0.25

# NMS 阈值：IoU 高于此值的重叠框被合并
NMS_THRESH = 0.45

# ==================== 摄像头配置 ====================
# ⚠️ 摄像头原始分辨率（≠ 模型输入尺寸）
# 图像会先从摄像头采集，再 resize 到 MODEL_INPUT_SIZE

# 默认摄像头
CAMERA_SOURCE = 0

# 摄像头分辨率
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 帧率
CAMERA_FPS = 30

# ==================== 跌倒检测配置 ====================

# 判断窗口帧数
FALL_THRESHOLD_FRAMES = 15

# 角度阈值（度）
FALL_ANGLE_THRESHOLD = 60

# 确认比例
FALL_CONFIRM_RATIO = 0.8

# ==================== COCO 类别 ====================

COCO_CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
)
