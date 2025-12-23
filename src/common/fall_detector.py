"""
跌倒检测状态机
"""
import numpy as np
import time


class FallDetector:
    """
    跌倒检测器 - 基于姿态关键点的状态机
    核心逻辑：跌倒 = 关键点高度变化 + 身体主轴角度 + 持续时间
    """
    
    def __init__(self, threshold_frames=15, angle_threshold=60, confirm_ratio=0.8):
        self.threshold_frames = threshold_frames
        self.angle_threshold = angle_threshold
        self.confirm_ratio = confirm_ratio
        self.history = []
        self.last_fall_time = None
    
    def detect(self, keypoints):
        if keypoints is None or len(keypoints) < 12:
            return False, 0
        
        angle = self._calc_body_angle(keypoints)
        is_falling = angle > self.angle_threshold
        self.history.append(is_falling)
        
        if len(self.history) > self.threshold_frames:
            self.history.pop(0)
        
        if len(self.history) >= self.threshold_frames:
            fall_ratio = sum(self.history) / len(self.history)
            is_fall = fall_ratio >= self.confirm_ratio
            if is_fall:
                self.last_fall_time = time.time()
            return is_fall, angle
        
        return False, angle
    
    def _calc_body_angle(self, keypoints):
        try:
            head = keypoints[0][:2]
            hip = [(keypoints[11][0] + keypoints[12][0]) / 2,
                   (keypoints[11][1] + keypoints[12][1]) / 2]
            dx = head[0] - hip[0]
            dy = head[1] - hip[1]
            angle = np.abs(np.arctan2(dx, -dy) * 180 / np.pi)
            return angle
        except:
            return 0
    
    def reset(self):
        self.history.clear()
        self.last_fall_time = None
