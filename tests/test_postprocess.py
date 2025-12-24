"""
后处理模块单元测试
"""
import unittest
import numpy as np
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from common.postprocess import nms, get_class_name


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
    
    def test_nms_single_box(self):
        """测试单个框"""
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        scores = np.array([0.9])
        
        keep = nms(boxes, scores, 0.5)
        
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep[0], 0)


class TestGetClassName(unittest.TestCase):
    """类别名称获取测试"""
    
    def test_valid_class_id(self):
        """测试有效类别 ID"""
        self.assertEqual(get_class_name(0), "person")
        self.assertEqual(get_class_name(2), "car")
    
    def test_invalid_class_id_returns_fallback(self):
        """测试无效类别 ID 返回默认值"""
        result = get_class_name(999)
        # 实际返回 "class_999" 格式
        self.assertTrue(result.startswith("class_"))
    
    def test_boundary_class_id(self):
        """测试边界类别 ID"""
        # COCO 有 80 类，ID 0-79
        result_79 = get_class_name(79)
        self.assertIsInstance(result_79, str)
        self.assertNotEqual(result_79, "")


class TestNMSEdgeCases(unittest.TestCase):
    """NMS 边界情况测试"""
    
    def test_nms_same_score(self):
        """测试相同得分"""
        boxes = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.9])
        
        keep = nms(boxes, scores, 0.3)
        
        # 至少保留一个
        self.assertGreaterEqual(len(keep), 1)
    
    def test_nms_low_threshold(self):
        """测试低 NMS 阈值"""
        boxes = np.array([
            [0, 0, 100, 100],
            [90, 90, 190, 190]  # 轻微重叠
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8])
        
        # 这两个框 IoU 很小（约 1%），所以都保留是正确的
        keep = nms(boxes, scores, 0.5)
        self.assertEqual(len(keep), 2)


if __name__ == '__main__':
    unittest.main()
