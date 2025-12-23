"""
统一日志模块 - 支持 PC（logging）和板端（zlog）

使用方式:
    from common.logger import zlog
    zlog.info("模型加载完成")
    zlog.error("加载失败")

PC端: 自动使用 Python logging
板端: 自动使用 zlog（需要 libzlog.so）
"""
import os
import sys
import logging
from datetime import datetime



class ZLogWrapper:
    """
    zlog 封装（板端使用）
    通过 ctypes 调用 libzlog.so
    """
    
    def __init__(self, conf_path="/etc/zlog.conf", category="default"):
        import ctypes
        
        self.zlog = ctypes.CDLL("libzlog.so")
        ret = self.zlog.zlog_init(conf_path.encode())
        if ret != 0:
            raise RuntimeError(f"zlog init failed: {conf_path}")
        
        self.zc = self.zlog.zlog_get_category(category.encode())
        if not self.zc:
            raise RuntimeError(f"zlog get category failed: {category}")
        
        self._category = category
    
    def debug(self, msg):
        self.zlog.zlog_debug(self.zc, str(msg).encode())
    
    def info(self, msg):
        self.zlog.zlog_info(self.zc, str(msg).encode())
    
    def warn(self, msg):
        self.zlog.zlog_warn(self.zc, str(msg).encode())
    
    def warning(self, msg):
        self.warn(msg)
    
    def error(self, msg):
        self.zlog.zlog_error(self.zc, str(msg).encode())
    
    def fatal(self, msg):
        self.zlog.zlog_fatal(self.zc, str(msg).encode())


class ColoredLogger:
    """
    彩色控制台日志（PC 开发阶段使用）
    基于 Python logging，带颜色输出
    """
    
    # 日志级别颜色
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'FATAL': '\033[35m',    # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def __init__(self, name="rk3576", level=logging.INFO, log_file=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # 控制台输出（带颜色）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # 文件输出（可选）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def _colorize(self, level, msg):
        """给消息添加颜色（仅终端）"""
        if sys.stdout.isatty():
            color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            return f"{color}{msg}{reset}"
        return msg
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warn(self, msg):
        self.logger.warning(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def fatal(self, msg):
        self.logger.critical(msg)


def create_logger(name="rk3576", use_zlog=None, zlog_conf="/etc/zlog.conf", 
                  zlog_category="default", log_file=None, level=logging.INFO):
    """
    创建日志器（工厂函数）
    
    Args:
        name: 日志器名称
        use_zlog: 是否使用 zlog，None=自动检测
        zlog_conf: zlog 配置文件路径
        zlog_category: zlog 日志类别
        log_file: 日志文件路径（仅 PC 模式）
        level: 日志级别
    
    Returns:
        logger 实例
    """
    # 自动检测：有 libzlog.so 就用 zlog
    if use_zlog is None:
        try:
            import ctypes
            ctypes.CDLL("libzlog.so")
            use_zlog = os.path.exists(zlog_conf)
        except OSError:
            use_zlog = False
    
    if use_zlog:
        try:
            return ZLogWrapper(zlog_conf, zlog_category)
        except Exception as e:
            print(f"[WARNING] zlog 初始化失败，回退到 logging: {e}")
            return ColoredLogger(name, level, log_file)
    else:
        return ColoredLogger(name, level, log_file)


# 全局默认 zlog 实例
zlog = create_logger("rk3576")


# 快捷函数
def info(msg):
    zlog.info(msg)

def debug(msg):
    zlog.debug(msg)

def warn(msg):
    zlog.warn(msg)

def warning(msg):
    zlog.warning(msg)

def error(msg):
    zlog.error(msg)

def fatal(msg):
    zlog.fatal(msg)
