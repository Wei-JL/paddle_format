"""
通用日志系统模块

提供统一的日志管理功能，支持：
- 按文件名获取日志器实例
- 简单的日志文件命名
- 统一日志格式
- 文件轮转和保留策略
- 多级别日志输出
- 控制台和文件同时输出
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


# 宏定义 - 配置参数
LOG_DIR_NAME = "logs"
LOG_FILE_PREFIX = "log_out"
LOG_FILE_SUFFIX = ".log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s() - %(lineno)d - %(message)s'


class UniversalLogger:
    """通用日志器类 - 可复制到任何项目使用"""
    
    _loggers = {}  # 存储不同文件名的日志器实例
    _log_dir = None
    _initialized = False
    
    @classmethod
    def get_logger(cls, filename: str = None):
        """
        获取指定文件名的日志器实例
        
        Args:
            filename: 调用日志的文件名（不含扩展名）
            
        Returns:
            logging.Logger: 日志器实例
        """
        if filename is None:
            filename = "default"
        
        # 如果该文件名的日志器已存在，直接返回
        if filename in cls._loggers:
            return cls._loggers[filename]
        
        # 首次初始化时设置日志配置
        if not cls._initialized:
            cls._setup_logging_config()
            cls._initialized = True
        
        # 创建新的日志器
        logger = logging.getLogger(f"UniversalLogger.{filename}")
        logger.setLevel(logging.INFO)
        
        # 如果没有处理器，添加处理器
        if not logger.handlers:
            # 统一日志格式
            formatter = logging.Formatter(
                LOG_FORMAT,
                datefmt=LOG_DATE_FORMAT
            )
            
            # 文件处理器 - 轮转日志
            log_file = cls._log_dir / cls._get_log_filename()
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # 控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 缓存日志器
        cls._loggers[filename] = logger
        
        return logger
    
    @classmethod
    def _setup_logging_config(cls):
        """设置日志配置"""
        # 创建日志目录
        cls._log_dir = cls._get_log_directory()
        cls._log_dir.mkdir(exist_ok=True)
    
    @classmethod
    def _get_log_directory(cls):
        """获取日志目录路径"""
        # 获取当前文件所在目录的上级目录的上级目录作为基准（项目根目录）
        current_path = Path(__file__).resolve()
        base_dir = current_path.parent.parent.parent
        return base_dir / LOG_DIR_NAME
    
    @classmethod
    def _get_log_filename(cls):
        """生成日志文件名：log_out + 时间戳 + .log"""
        timestamp = datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
        return f"{LOG_FILE_PREFIX}_{timestamp}{LOG_FILE_SUFFIX}"
    
    @classmethod
    def set_log_level(cls, level):
        """
        设置所有日志器的级别
        
        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        for logger in cls._loggers.values():
            logger.setLevel(level)
    
    @classmethod
    def get_log_info(cls):
        """获取日志系统信息"""
        return {
            'log_directory': str(cls._log_dir) if cls._log_dir else None,
            'active_loggers': list(cls._loggers.keys()),
            'log_filename': cls._get_log_filename(),
            'config': {
                'log_file_prefix': LOG_FILE_PREFIX,
                'log_file_suffix': LOG_FILE_SUFFIX,
                'max_bytes': LOG_MAX_BYTES,
                'backup_count': LOG_BACKUP_COUNT,
                'log_format': LOG_FORMAT
            }
        }


def get_logger(filename: str = None):
    """
    获取日志器实例的便捷函数
    
    Args:
        filename: 调用日志的文件名（不含扩展名）
        
    Returns:
        logging.Logger: 日志器实例
        
    Example:
        logger = get_logger("data_processor")
        logger.info("开始处理数据")
    """
    return UniversalLogger.get_logger(filename)


if __name__ == "__main__":
    # 测试日志系统
    
    # 测试不同文件名的日志器
    logger1 = get_logger("test_module1")
    logger2 = get_logger("test_module2")
    
    logger1.debug("这是模块1的DEBUG消息")
    logger1.info("这是模块1的INFO消息")
    logger1.warning("这是模块1的WARNING消息")
    
    logger2.info("这是模块2的INFO消息")
    logger2.error("这是模块2的ERROR消息")
    
    # 显示日志系统信息
    log_info = UniversalLogger.get_log_info()
    print(f"日志系统信息:")
    for key, value in log_info.items():
        print(f"  {key}: {value}")
    
    # 测试同一文件名获取相同实例
    logger1_again = get_logger("test_module1")
    print(f"相同文件名获取相同实例: {logger1 is logger1_again}")