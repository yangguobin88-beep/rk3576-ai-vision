# BaseRKNNModel 基类详解

> 这是一个"模板类"，用于统一管理所有 RKNN 模型的加载、推理、释放流程。

---

## 🎯 为什么需要 BaseRKNNModel？

### 问题：没有基类时的代码重复

```python
# YOLOv8 模型
rknn_yolo = RKNNLite()
rknn_yolo.load_rknn('yolov8.rknn')
rknn_yolo.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
# ... 推理 ...
rknn_yolo.release()

# RetinaFace 模型
rknn_face = RKNNLite()
rknn_face.load_rknn('retinaface.rknn')  # 重复代码！
rknn_face.init_runtime(core_mask=RKNNLite.NPU_CORE_0)  # 重复代码！
# ... 推理 ...
rknn_face.release()  # 重复代码！
```

**问题：** 每个模型都要重复写 `load_rknn`、`init_runtime`、`release`。

---

### 解决方案：用基类封装公共逻辑

```python
class YOLOv8Detector(BaseRKNNModel):
    def preprocess(self, img):
        # 只需要写 YOLO 特有的预处理
        return preprocess(img, (640, 640))
    
    def postprocess(self, outputs):
        # 只需要写 YOLO 特有的后处理
        return yolov8_postprocess(outputs)

# 使用时超级简单
detector = YOLOv8Detector('yolov8.rknn')
boxes, classes, scores = detector.infer(img)  # 自动调用预处理+推理+后处理
detector.release()
```

---

## 📖 逐行代码讲解

### 1️⃣ 初始化：自动识别 PC/板端

```python
def __init__(self, model_path, core_mask=None):
    try:
        from rknnlite.api import RKNNLite  # 板端库
        self.rknn = RKNNLite()
        self.is_lite = True
    except ImportError:
        from rknn.api import RKNN  # PC 端库
        self.rknn = RKNN()
        self.is_lite = False
```

**作用：** 自动判断运行环境
- **板端**：导入成功 → 用 `RKNNLite`
- **PC端**：导入失败 → 用 `RKNN`

**好处：** 同一份代码，PC 和板子都能跑！

---

### 2️⃣ 加载模型 + 初始化运行时

```python
# 1. 加载模型文件
ret = self.rknn.load_rknn(model_path)
if ret != 0:
    raise RuntimeError(f"加载模型失败: {model_path}")

# 2. 初始化运行时（分配 NPU 资源）
if self.is_lite and core_mask is not None:
    ret = self.rknn.init_runtime(core_mask=core_mask)  # 板端可以指定 NPU 核心
else:
    ret = self.rknn.init_runtime()  # PC 端不需要指定

if ret != 0:
    raise RuntimeError("初始化运行时失败")
```

**`core_mask` 参数：**
- `NPU_CORE_0`：使用第 0 个 NPU 核心
- `NPU_CORE_AUTO`：自动分配
- 只在板端有效，PC 端无此概念

---

### 3️⃣ 预处理和后处理：留给子类实现

```python
def preprocess(self, img):
    raise NotImplementedError  # 强制子类实现

def postprocess(self, outputs):
    raise NotImplementedError  # 强制子类实现
```

**为什么用 `NotImplementedError`？**

- **强制约束**：子类必须实现这两个方法，否则会报错
- **不同模型的预处理/后处理不一样**：
  - YOLO：640×640 resize
  - RetinaFace：320×320 resize
  - 后处理更是完全不同

---

### 4️⃣ 推理方法：组合预处理+推理+后处理

```python
def infer(self, img):
    img_input = self.preprocess(img)           # 1. 预处理
    outputs = self.rknn.inference(inputs=[img_input])  # 2. NPU 推理
    return self.postprocess(outputs)           # 3. 后处理
```

**这就是模板方法模式！**

流程固定，但每个步骤的具体实现由子类决定。

---

### 5️⃣ 资源释放

```python
def release(self):
    if self.rknn is not None:
        self.rknn.release()  # 释放 NPU 资源
        self.rknn = None
```

**重要性：** 不释放会导致 NPU 资源泄漏！

---

### 6️⃣ 上下文管理器：支持 `with` 语法

```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.release()  # 自动释放资源
    return False
```

**用法：**

```python
# 方式 1：手动释放
detector = YOLOv8Detector('yolov8.rknn')
result = detector.infer(img)
detector.release()  # 容易忘记！

# 方式 2：with 语法（推荐）
with YOLOv8Detector('yolov8.rknn') as detector:
    result = detector.infer(img)
# 自动释放，不会忘记
```

---

## 🧩 实际使用示例

### 子类实现：YOLOv8Detector

```python
from common import BaseRKNNModel, preprocess
from common.postprocess import yolov8_postprocess

class YOLOv8Detector(BaseRKNNModel):
    """YOLOv8 检测器"""
    
    def preprocess(self, img):
        """YOLO 预处理"""
        return preprocess(img, target_size=(640, 640))
    
    def postprocess(self, outputs):
        """YOLO 后处理"""
        return yolov8_postprocess(outputs)
```

### 使用

```python
# 方式 1：普通用法
detector = YOLOv8Detector('models/yolov8.rknn', core_mask=RKNNLite.NPU_CORE_0)
img = cv2.imread('test.jpg')
boxes, classes, scores = detector.infer(img)
detector.release()

# 方式 2：with 语法（推荐）
with YOLOv8Detector('models/yolov8.rknn') as detector:
    img = cv2.imread('test.jpg')
    boxes, classes, scores = detector.infer(img)
    # 自动释放资源
```

---

## 🎨 设计模式：模板方法模式

```
BaseRKNNModel (模板类)
    ├── __init__()        ← 固定流程：加载+初始化
    ├── infer()           ← 固定流程：预处理→推理→后处理
    ├── preprocess()      ← 抽象方法，子类实现
    ├── postprocess()     ← 抽象方法，子类实现
    └── release()         ← 固定流程：释放资源

YOLOv8Detector (子类)
    ├── preprocess()      ← 实现 YOLO 的预处理
    └── postprocess()     ← 实现 YOLO 的后处理

RetinaFaceDetector (子类)
    ├── preprocess()      ← 实现 RetinaFace 的预处理
    └── postprocess()     ← 实现 RetinaFace 的后处理
```

---

## ✅ 优点总结

| 优点 | 说明 |
|------|------|
| **代码复用** | 加载、初始化、释放逻辑只写一次 |
| **统一接口** | 所有模型用法一致 |
| **自动适配** | PC/板端自动切换 |
| **防止错误** | 强制子类实现必要方法 |
| **资源管理** | 支持 `with` 语法，不会忘记释放 |

---

## 🔑 关键概念

1. **基类（Base Class）**：定义共同逻辑
2. **抽象方法（Abstract Method）**：子类必须实现
3. **模板方法（Template Method）**：固定流程，细节由子类决定
4. **上下文管理器（Context Manager）**：`with` 语法自动管理资源

---

## ✅ 总结

`BaseRKNNModel` 就像一个**"模型加载器模板"**：

- **固定的事情**（加载、初始化、释放）→ 基类搞定
- **变化的事情**（预处理、后处理）→ 子类实现

**结果：** 新增模型只需要写 10 行代码，不用重复写 50 行！
