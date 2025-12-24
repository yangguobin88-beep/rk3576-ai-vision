# RK3576 边缘 AI 视觉检测系统

<p align="center">
  <img src="https://img.shields.io/badge/Platform-RK3576-blue" alt="Platform">
  <img src="https://img.shields.io/badge/NPU-6TOPS-green" alt="NPU">
  <img src="https://img.shields.io/badge/Version-1.0.1-orange" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

基于瑞芯微 RK3576 SoC 的边缘 AI 视觉检测系统，利用 6 TOPS NPU 实现高性能实时目标检测、跌倒判断等 AI 视觉功能。

---

## ✨ 特性

- 🚀 **高性能推理**：利用 RK3576 内置 6 TOPS NPU 加速，实现实时 AI 检测
- 🔄 **跨平台开发**：PC 端使用 ONNX 开发调试，板端使用 RKNN 部署
- 🧩 **模块化架构**：detectors / logic / common 三层分离，易于扩展
- 🛡️ **生产级稳定**：完整的异常处理、信号处理、资源管理
- 📦 **开箱即用**：支持 YOLOv8 目标检测，可扩展人脸/姿态检测

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────┐
│  入口层：main.py                                │
├─────────────────────────────────────────────────┤
│  AI 模型层：detectors/                          │
│  - BaseModelDetector（抽象基类）                │
│  - ONNXModelDetector（PC 端）                   │
│  - RKNNModelDetector（板端）                    │
│  - create_model_detector()（工厂函数）          │
├─────────────────────────────────────────────────┤
│  业务逻辑层：logic/                             │
│  - FallJudge（跌倒判断）                        │
├─────────────────────────────────────────────────┤
│  基础设施层：common/                            │
│  - camera / preprocess / postprocess            │
│  - config / logger                              │
└─────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
rk3576-ai-vision/
├── src/
│   ├── main.py                  # 主程序入口
│   ├── detectors/               # AI 模型检测器
│   │   ├── __init__.py
│   │   ├── detector.py          # BaseModelDetector + 工厂
│   │   ├── base_model.py        # BaseRKNNModel
│   │   └── yolo_detector.py     # YOLOv8ModelDetector
│   ├── logic/                   # 业务逻辑
│   │   ├── __init__.py
│   │   └── fall_judge.py        # 跌倒判断器
│   └── common/                  # 通用模块
│       ├── __init__.py
│       ├── camera.py            # 摄像头封装
│       ├── preprocess.py        # 图像预处理
│       ├── postprocess.py       # 检测后处理
│       ├── config.py            # 全局配置
│       └── logger.py            # 日志系统
├── models/                      # 模型文件 (.onnx/.rknn)
├── tests/                       # 单元测试
├── scripts/                     # 工具脚本
├── docs/                        # 开发文档
├── requirements.txt             # Python 依赖
└── README.md
```

---

## 🚀 快速开始

### 环境要求

| 环境 | 要求 |
|------|------|
| Python | 3.8+ |
| OS | Windows / Linux / RK3576 板端 |

### 安装依赖

```bash
# PC 端（开发）
pip install -r requirements.txt

# 板端（部署）
pip3 install opencv-python numpy rknn-lite2
```

### 运行示例

#### 图片检测

```bash
cd src
python main.py --image ../test.jpg --model ../models/yolov8n.onnx
```

#### 摄像头实时检测

```bash
cd src
python main.py --camera 0 --model ../models/yolov8n.onnx
```

#### 板端运行（RK3576）

```bash
cd src
python3 main.py --camera 0 --model ../models/yolov8n.rknn
```

---

## ⚙️ 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--image` | str | - | 输入图片路径 |
| `--camera` | int | - | 摄像头设备号（0, 1, ...） |
| `--model` | str | `../models/yolov8n.onnx` | 模型文件路径 |
| `--conf` | float | 0.25 | 置信度阈值 |
| `--nms` | float | 0.45 | NMS 阈值 |
| `--width` | int | 1280 | 摄像头宽度 |
| `--height` | int | 720 | 摄像头高度 |
| `--output` | str | `result.jpg` | 输出图片路径 |
| `--show` | flag | False | 是否显示检测窗口 |
| `--version` | flag | - | 显示版本号 |

---

## 📊 支持的模型

| 模型 | 格式 | 用途 | 状态 |
|------|------|------|------|
| YOLOv8n | .onnx / .rknn | 目标检测 | ✅ 已支持 |
| YOLOv8-pose | .onnx / .rknn | 姿态检测 | 🔜 规划中 |
| RetinaFace | .onnx / .rknn | 人脸检测 | 🔜 规划中 |
| MoveNet | .onnx / .rknn | 人体姿态 | 🔜 规划中 |

---

## 🔧 开发指南

### 添加新的 AI 模型

1. 在 `src/detectors/` 下创建新的检测器文件
2. 继承 `BaseModelDetector` 或 `BaseRKNNModel`
3. 实现 `preprocess()` 和 `postprocess()` 方法
4. 在 `detectors/__init__.py` 中导出

```python
# src/detectors/face_detector.py
from .base_model import BaseRKNNModel

class FaceModelDetector(BaseRKNNModel):
    def preprocess(self, img):
        # 人脸检测专用预处理
        pass
    
    def postprocess(self, outputs):
        # 人脸检测专用后处理
        pass
```

### 添加新的业务逻辑

1. 在 `src/logic/` 下创建新的判断器
2. 类名使用 `XXXJudge` 或 `XXXAnalyzer`
3. 主方法使用 `judge()` 或 `analyze()`

```python
# src/logic/intrusion_judge.py
class IntrusionJudge:
    def judge(self, boxes, classes):
        # 入侵检测逻辑
        pass
```

---

## 🧪 运行测试

```bash
# 运行所有单元测试
cd d:\rk3576-ai-vision
python -m unittest discover tests/ -v
```

---

## 📈 性能指标

| 平台 | 模型 | 分辨率 | FPS |
|------|------|--------|-----|
| PC (i7-11800H) | YOLOv8n ONNX | 640x640 | ~50 |
| RK3576 NPU | YOLOv8n RKNN | 640x640 | 待测试 |

---

## 📝 更新日志

### v1.0.1 (2024-12-24)
- ✨ 添加生产级保护（信号处理、异常捕获）
- ✨ 统一资源回收入口
- ✨ 添加版本号和参数验证
- ✨ 添加单元测试

### v1.0.0 (2024-12-23)
- 🎉 初始版本
- 目录结构重构（detectors/logic/common）
- 支持 YOLOv8 目标检测
- 支持 ONNX 和 RKNN 双模式

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 👤 作者

**杨国彬**

- GitHub: [@yangguobin88-beep](https://github.com/yangguobin88-beep)
- Email: yangguobin88@gmail.com

---

<p align="center">
  Made with ❤️ for Edge AI
</p>
