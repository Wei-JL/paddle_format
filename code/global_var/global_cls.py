"""
全局变量和常量定义

包含项目中使用的通用参数、常量和配置
"""

# VOC数据集相关常量
ANNOTATIONS_DIR = "Annotations"
JPEGS_DIR = "JPEGImages"
IMAGESETS_DIR = "ImageSets"
MAIN_DIR = "Main"
XML_EXTENSION = ".xml"

# 支持的图像文件扩展名
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.svg']

# 数据集划分相关常量 (默认只有训练集和验证集)
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
TEST_RATIO = 0.0

# 数据集划分文件名
TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"
TRAINVAL_TXT = "trainval.txt"

# 日志相关常量
LOG_DIR_NAME = "logs"
LOG_FILE_PREFIX = "log_out"
LOG_FILE_SUFFIX = ".log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s() - %(lineno)d - %(message)s'

# 文件处理相关常量
DEFAULT_ENCODING = 'utf-8'
RANDOM_SEED = 42