"""
RKNN 模型基类 - 统一 load/infer/release
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

import numpy as np


class _RKNNRuntime(Protocol):
    """用于类型约束的最小 RKNN 运行时协议"""

    def load_rknn(self, model_path: str) -> int: ...

    def init_runtime(self, core_mask: int | None = None) -> int: ...

    def inference(self, inputs: Iterable[np.ndarray]) -> list[np.ndarray]: ...

    def release(self) -> None: ...


class BaseRKNNModel:
    """所有 RKNN 模型的基类"""

    def __init__(self, model_path: str, core_mask: int | None = None):
        try:
            from rknnlite.api import RKNNLite  # type: ignore
            self.rknn: _RKNNRuntime = RKNNLite()
            self.is_lite = True
        except ImportError:
            from rknn.api import RKNN  # type: ignore
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

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """子类实现：BGR 图像 -> 模型输入格式"""
        raise NotImplementedError

    def postprocess(self, outputs: list[np.ndarray] | tuple[np.ndarray, ...]) -> Any:
        """子类实现：模型输出 -> 业务需要的结构"""
        raise NotImplementedError

    def infer(self, img: np.ndarray) -> Any:
        """执行预处理-推理-后处理的完整链路。"""
        img_input = self.preprocess(img)
        outputs = self.rknn.inference(inputs=[img_input])
        return self.postprocess(outputs)

    def release(self) -> None:
        if getattr(self, "rknn", None) is not None:
            self.rknn.release()
            self.rknn = None

    def __enter__(self) -> "BaseRKNNModel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False
