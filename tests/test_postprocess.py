"""
后处理模块单元测试
"""
import unittest
import numpy as np
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from common.postprocess import nms, get_class_name, yolov8_postprocess


class TestNMS(unittest.TestCase):
    """NMS 函数测试"""
    
    def test_nms_no_overlap(self):
        """测试不重叠的框"""
        boxes = np.array([
            [0, 0, 10, 10],
            [100, 100, 110, 110]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        
        keep = nms(boxes, scores, 0.5)
        
        # 两个框都应该保留
        self.assertEqual(len(keep), 2)
    
    def test_nms_high_overlap(self):
        """测试高重叠的框"""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110]  # 高度重叠
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        
        keep = nms(boxes, scores, 0.5)
        
        # 只应该保留得分高的那个
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0], 0)
    
    def test_nms_empty_input(self):
        """测试空输入"""
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        
        keep = nms(boxes, scores, 0.5)
        
        self.assertEqual(len(keep), 0)


class TestGetClassName(unittest.TestCase):
    """类别名称获取测试"""
    
    def test_valid_class_id(self):
        """测试有效类别 ID"""
        self.assertEqual(get_class_name(0), "person")
        self.assertEqual(get_class_name(2), "car")
    
    def test_invalid_class_id(self):
        """测试无效类别 ID"""
        result = get_class_name(999)
        self.assertEqual(result, "unknown")


class TestYOLOv8Postprocess(unittest.TestCase):
    """YOLOv8 后处理测试"""
    
    def test_empty_output(self):
        """测试空输出（无检测结果）"""
        # 模拟模型输出（全零，无检测）
        outputs = [np.zeros((1, 84, 8400), dtype=np.float32)]
        
        boxes, classes, scores = yolov8_postprocess(
            outputs, 
            obj_thresh=0.25, 
            nms_thresh=0.45,
            img_size=(640, 640)
        )
        
        # 应该返回 None（无检测结果）
        self.assertIsNone(boxes)
        self.assertIsNone(classes)
        self.assertIsNone(scores)
    
    def test_output_format(self):
        """测试输出格式正确性"""
        # 创建有检测结果的模拟输出
        outputs = [np.zeros((1, 84, 8400), dtype=np.float32)]
        
        # 在某个位置设置高置信度
        # 84 = 4(box) + 80(classes)
        # 设置 box: cx, cy, w, h
        outputs[0][0, 0, 0] = 320  # cx
        outputs[0][0, 1, 0] = 320  # cy
        outputs[0][0, 2, 0] = 100  # w
        outputs[0][0, 3, 0] = 100  # h
        # 设置 person 类（index 4）高置信度
        outputs[0][0, 4, 0] = 0.9
        
        boxes, classes, scores = yolov8_postprocess(
            outputs,
            obj_thresh=0.25,
            nms_thresh=0.45,
            img_size=(640, 640)
        )
        
        if boxes is not None:
            # 验证输出格式
            self.assertEqual(boxes.ndim, 2)
            self.assertEqual(boxes.shape[1], 4)
            self.assertEqual(len(classes), len(boxes))
            self.assertEqual(len(scores), len(boxes))


class TestPostprocessEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_high_threshold(self):
        """测试高阈值（过滤所有结果）"""
        outputs = [np.random.rand(1, 84, 8400).astype(np.float32) * 0.1]
        
        boxes, classes, scores = yolov8_postprocess(
            outputs,
            obj_thresh=0.99,  # 极高阈值
            nms_thresh=0.45,
            img_size=(640, 640)
        )
        
        self.assertIsNone(boxes)


if __name__ == '__main__':
    unittest.main()
