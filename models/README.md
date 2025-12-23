# RK3576 AI Vision - Models

此目录用于存放 RKNN 模型文件。

## 需要下载的模型

从网盘下载：https://console.zbox.filez.com/l/8ufwtG （提取码：rknn）

| 模型 | 文件名 | 用途 |
|------|--------|------|
| YOLOv8n | yolov8n.onnx | 目标检测 |
| RetinaFace | RetinaFace_mobile320.onnx | 人脸检测 |
| YOLOv8n-pose | yolov8n-pose.onnx | 人体姿态 |
| PPOCR-Det | ppocrv4_det.onnx | 文字检测 |
| PPOCR-Rec | ppocrv4_rec.onnx | 文字识别 |
| LPRNet | lprnet.onnx | 车牌识别 |

## 转换为 RKNN

使用 RKNN-Toolkit2 将 ONNX 转换为 RKNN 格式。
