"""
摄像头封装模块
"""
import cv2
import threading
import time


class Camera:
    """摄像头封装类 - 支持多线程采集"""
    
    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
    
    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {self.source}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self
    
    def _capture_loop(self):
        """后台采集循环（带断线重连）"""
        fail_count = 0
        max_retries = 5
        
        while self.running:
            ret, frame = self.cap.read()
            
            if ret:
                fail_count = 0  # 成功则重置计数
                with self.lock:
                    self.frame = frame
            else:
                fail_count += 1
                if fail_count >= max_retries:
                    # 连续失败，尝试重连
                    self._reconnect()
                    fail_count = 0
                else:
                    time.sleep(0.1)  # 短暂等待后重试
    
    def _reconnect(self):
        """重新连接摄像头"""
        try:
            if self.cap:
                self.cap.release()
            time.sleep(0.5)
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            time.sleep(1.0)  # 重连失败，等待后继续
    
    def start(self):
        if self.cap is None:
            self.open()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        time.sleep(0.1)
        return self
    
    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def release(self):
        self.stop()
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.release()


class FPSCounter:
    """FPS 计数器"""
    def __init__(self, window=30):
        self.times = []
        self.window = window
    
    def tick(self):
        self.times.append(time.time())
        if len(self.times) > self.window:
            self.times.pop(0)
    
    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])
