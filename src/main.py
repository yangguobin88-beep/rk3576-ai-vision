"""
RK3576 AI 视觉项目 - 主程序入口

使用方法：
    # 图片检测
    python main.py --image test.jpg --model ../models/yolov8n.onnx
    
    # 摄像头检测
    python main.py --camera 0 --model ../models/yolov8n.onnx
    
    # 板端运行
    python3 main.py --camera 0 --model ../models/yolov8.rknn
"""
import cv2
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors import create_detector
from common.camera import Camera, FPSCounter
from common.config import OBJ_THRESH, NMS_THRESH, CAMERA_WIDTH, CAMERA_HEIGHT
from common.logger import zlog


def draw_results(img, boxes, classes, scores, names):
    """绘制检测结果"""
    if boxes is None:
        return img
    
    img_draw = img.copy()
    for box, cls, score, name in zip(boxes, classes, scores, names):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if name == "person" else (255, 0, 0)
        
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        label = f"{name}: {score:.2f}"
        cv2.putText(img_draw, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return img_draw


def run_image(args):
    """图片检测"""
    zlog.info(f"[图片模式] {args.image}")
    
    img = cv2.imread(args.image)
    if img is None:
        zlog.error(f"无法读取图片: {args.image}")
        return
    
    # 创建检测器
    detector = create_detector(args.model, args.conf, args.nms)
    
    # 检测
    boxes, classes, scores, names = detector.detect(img)
    
    # 打印结果
    if boxes is not None:
        zlog.info(f"检测到 {len(boxes)} 个目标")
        for name, score in zip(names, scores):
            zlog.info(f"  {name}: {score:.2f}")
    else:
        zlog.info("未检测到目标")
    
    # 绘制并保存
    result = draw_results(img, boxes, classes, scores, names)
    output = args.output if args.output else "result.jpg"
    cv2.imwrite(output, result)
    zlog.info(f"结果保存: {output}")
    
    # 显示
    if args.show:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    detector.release()


def run_camera(args):
    """摄像头检测"""
    zlog.info(f"[摄像头模式] 设备 {args.camera}")
    
    # 创建检测器和摄像头
    detector = create_detector(args.model, args.conf, args.nms)
    camera = Camera(args.camera, args.width, args.height)
    fps_counter = FPSCounter()
    
    zlog.info("按 'q' 退出")
    
    try:
        camera.start()
        
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            # 检测
            boxes, classes, scores, names = detector.detect(frame)
            
            # 绘制
            frame = draw_results(frame, boxes, classes, scores, names)
            
            # FPS
            fps_counter.tick()
            fps = fps_counter.get_fps()
            count = len(boxes) if boxes is not None else 0
            cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('RK3576 AI Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        camera.release()
        detector.release()
        cv2.destroyAllWindows()
        zlog.info("程序退出")


def main():
    parser = argparse.ArgumentParser(description='RK3576 AI 视觉演示')
    parser.add_argument('--image', type=str, help='图片路径')
    parser.add_argument('--camera', type=int, help='摄像头设备号')
    parser.add_argument('--model', type=str, default='../models/yolov8n.onnx', help='模型路径')
    parser.add_argument('--conf', type=float, default=OBJ_THRESH, help='置信度阈值')
    parser.add_argument('--nms', type=float, default=NMS_THRESH, help='NMS 阈值')
    parser.add_argument('--width', type=int, default=CAMERA_WIDTH)
    parser.add_argument('--height', type=int, default=CAMERA_HEIGHT)
    parser.add_argument('--output', type=str, help='输出路径')
    parser.add_argument('--show', action='store_true', help='显示窗口')
    
    args = parser.parse_args()
    
    zlog.info("=" * 40)
    zlog.info("RK3576 AI 视觉演示")
    zlog.info(f"模型: {args.model}")
    zlog.info("=" * 40)
    
    if args.image:
        run_image(args)
    elif args.camera is not None:
        run_camera(args)
    else:
        zlog.warn("请指定输入源：--image 或 --camera")


if __name__ == '__main__':
    main()
