"""
RKNN 模型基类 - 统一 load/infer/release
"""


class BaseRKNNModel:
    """所有 RKNN 模型的基类"""
    
    def __init__(self, model_path, core_mask=None):
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            self.is_lite = True
        except ImportError:
            from rknn.api import RKNN
            self.rknn = RKNN()
            self.is_lite = False
        
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载模型失败: {model_path}")
        
        if self.is_lite and core_mask is not None:
            ret = self.rknn.init_runtime(core_mask=core_mask)
        else:
            ret = self.rknn.init_runtime()
        
        if ret != 0:
            raise RuntimeError("初始化运行时失败")
        
        self.model_path = model_path
    
    def preprocess(self, img):
        raise NotImplementedError
    
    def postprocess(self, outputs):
        raise NotImplementedError
    
    def infer(self, img):
        img_input = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[img_input])
        return self.postprocess(outputs)
    
    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
