"""
预处理模块单元测试
"""
import unittest
import numpy as np
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from common.preprocess import preprocess, preprocess_with_letterbox, restore_coords


class TestPreprocess(unittest.TestCase):
    """预处理函数测试"""
    
    def test_preprocess_output_shape(self):
        """测试 preprocess 输出尺寸"""
        # 创建模拟图片 (1080, 1920, 3) BGR
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        result = preprocess(img, (640, 640))
        
        self.assertEqual(result.shape, (640, 640, 3))
        self.assertEqual(result.dtype, np.uint8)
    
    def test_preprocess_with_letterbox_output_shape(self):
        """测试 letterbox 预处理输出尺寸"""
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        result, scale, pad = preprocess_with_letterbox(img, (640, 640))
        
        self.assertEqual(result.shape, (640, 640, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertIsInstance(scale, float)
        self.assertIsInstance(pad, tuple)
        self.assertEqual(len(pad), 2)
    
    def test_letterbox_scale_calculation(self):
        """测试 letterbox 缩放比例计算"""
        # 1920x1080 缩放到 640x640
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        _, scale, (pad_x, pad_y) = preprocess_with_letterbox(img, (640, 640))
        
        # scale = min(640/1920, 640/1080) = min(0.333, 0.593) = 0.333
        expected_scale = 640 / 1920
        self.assertAlmostEqual(scale, expected_scale, places=3)
        
        # 宽度刚好 640，pad_x = 0
        self.assertEqual(pad_x, 0)
        # 高度 = 1080 * 0.333 = 360，pad_y = (640 - 360) / 2 = 140
        self.assertGreater(pad_y, 0)
    
    def test_restore_coords_basic(self):
        """测试坐标还原基本功能"""
        # 模拟 letterbox 参数
        scale = 0.5
        pad = (50, 100)
        
        # 640 坐标系的框
        boxes = np.array([[100, 150, 200, 250]], dtype=np.float32)
        
        restored = restore_coords(boxes, scale, pad)
        
        # x' = (x - pad_x) / scale = (100 - 50) / 0.5 = 100
        self.assertEqual(restored[0, 0], 100)
        # y' = (y - pad_y) / scale = (150 - 100) / 0.5 = 100
        self.assertEqual(restored[0, 1], 100)
    
    def test_restore_coords_none_input(self):
        """测试空输入"""
        result = restore_coords(None, 1.0, (0, 0))
        self.assertIsNone(result)
    
    def test_restore_coords_roundtrip(self):
        """测试坐标还原往返一致性"""
        # 原图尺寸 1920x1080
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        _, scale, pad = preprocess_with_letterbox(img, (640, 640))
        
        # 假设模型在 640 坐标系输出的框
        boxes_640 = np.array([[100, 200, 300, 400]], dtype=np.float32)
        
        # 还原到原图坐标
        boxes_orig = restore_coords(boxes_640, scale, pad)
        
        # 验证还原后的坐标在合理范围内
        self.assertTrue(np.all(boxes_orig[:, [0, 2]] >= 0))
        self.assertTrue(np.all(boxes_orig[:, [1, 3]] >= 0))


class TestPreprocessEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_square_image(self):
        """测试正方形图片"""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        result, scale, (pad_x, pad_y) = preprocess_with_letterbox(img, (640, 640))
        
        self.assertEqual(scale, 1.0)
        self.assertEqual(pad_x, 0)
        self.assertEqual(pad_y, 0)
    
    def test_small_image(self):
        """测试小图片（需要放大）"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, scale, pad = preprocess_with_letterbox(img, (640, 640))
        
        self.assertEqual(result.shape, (640, 640, 3))
        self.assertGreater(scale, 1.0)


if __name__ == '__main__':
    unittest.main()
