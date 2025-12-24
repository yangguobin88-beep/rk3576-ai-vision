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
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detectors import create_model_detector
from common.camera import Camera, FPSCounter
from common.config import OBJ_THRESH, NMS_THRESH, CAMERA_WIDTH, CAMERA_HEIGHT
from common.logger import zlog

# 版本号
__version__ = "1.0.1"

# 全局资源（用于优雅退出）
_global_detector = None
_global_camera = None
_running = True


def _signal_handler(sig, frame):
    """处理退出信号（只改状态，不直接退出）"""
    global _running
    zlog.info(f"收到信号 {sig}，准备退出...")
    _running = False


def _cleanup():
    """清理全局资源（唯一的资源回收入口）"""
    global _global_detector, _global_camera
    
    if _global_camera is not None:
        try:
            _global_camera.release()
        except Exception:
            pass
        _global_camera = None
    
    if _global_detector is not None:
        try:
            _global_detector.release()
        except Exception:
            pass
        _global_detector = None
    
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def _graceful_exit(code: int = 0):
    """优雅退出（唯一的退出入口）"""
    try:
        _cleanup()
    except Exception:
        zlog.exception("cleanup 过程中发生异常")
    finally:
        zlog.info("程序退出")
        sys.exit(code)


def _validate_args(args):
    """验证启动参数"""
    # 检查模型文件
    if not os.path.exists(args.model):
        zlog.error(f"模型文件不存在: {args.model}")
        return False
    
    # 检查图片文件
    if args.image and not os.path.exists(args.image):
        zlog.error(f"图片文件不存在: {args.image}")
        return False
    
    # 检查阈值范围
    if not (0 < args.conf < 1):
        zlog.warn(f"置信度阈值异常: {args.conf}，建议范围 (0, 1)")
    
    if not (0 < args.nms < 1):
        zlog.warn(f"NMS 阈值异常: {args.nms}，建议范围 (0, 1)")
    
    return True


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
    """图片检测（不负责 cleanup，由 _graceful_exit 统一处理）"""
    global _global_detector
    
    zlog.info(f"[图片模式] {args.image}")

    img = cv2.imread(args.image)
    if img is None:
        zlog.error(f"无法读取图片: {args.image}")
        return
    
    # 创建检测器
    _global_detector = create_model_detector(args.model, args.conf, args.nms)
    
    # 检测
    boxes, classes, scores, names = _global_detector.detect(img)
    
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


def run_camera(args):
    """摄像头检测（不负责 cleanup，由 _graceful_exit 统一处理）"""
    global _global_detector, _global_camera, _running
    
    zlog.info(f"[摄像头模式] 设备 {args.camera}")
    
    # 创建检测器和摄像头
    _global_detector = create_model_detector(args.model, args.conf, args.nms)
    _global_camera = Camera(args.camera, args.width, args.height)
    fps_counter = FPSCounter()
    
    zlog.info("按 'q' 或 Ctrl+C 退出")
    
    _global_camera.start()
    
    while _running:
        frame = _global_camera.read()
        if frame is None:
            continue
        
        try:
            # 检测（单帧异常不中断）
            boxes, classes, scores, names = _global_detector.detect(frame)
            
            # 绘制
            frame = draw_results(frame, boxes, classes, scores, names)
        except Exception as e:
            zlog.warn(f"单帧推理异常，跳过: {e}")
            continue
        
        # FPS
        fps_counter.tick()
        fps = fps_counter.get_fps()
        count = len(boxes) if boxes is not None else 0
        cv2.putText(frame, f"FPS: {fps:.1f} | Objects: {count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('RK3576 AI Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            _running = False  # 统一用状态控制退出


def main():
    # 注册信号处理（只改状态，不直接退出）
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
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
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    zlog.info("=" * 40)
    zlog.info(f"RK3576 AI 视觉演示 v{__version__}")
    zlog.info(f"模型: {args.model}")
    zlog.info("=" * 40)
    
    # 参数验证
    if not _validate_args(args):
        return
    
    if args.image:
        run_image(args)
    elif args.camera is not None:
        run_camera(args)
    else:
        zlog.warn("请指定输入源：--image 或 --camera")


if __name__ == '__main__':
    try:
        main()
        _graceful_exit(0)  # 正常结束也走统一出口
    except KeyboardInterrupt:
        zlog.info("用户主动退出 (Ctrl+C)")
        _graceful_exit(0)
    except Exception:
        zlog.exception("程序异常退出")
        _graceful_exit(1)
