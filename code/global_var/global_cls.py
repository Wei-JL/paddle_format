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

# 类别统计相关常量
# 类别统计相关常量
LABELS_TXT = "labels.txt"
COUNT_ALL_CLS_TXT = "count_all_cls.txt"

# COCO格式相关常量
TRAIN_COCO_JSON = "train_coco.json"
VAL_COCO_JSON = "val_coco.json"
TEST_COCO_JSON = "test_coco.json"

# 图像处理相关常量
DEFAULT_IMAGE_CHANNELS = 3
SUPPORTED_IMAGE_CHANNELS = [1, 3, 4]  # 灰度、RGB、RGBA

# 文件夹后缀常量
FILTERED_SUFFIX = "filtered"
NO_DRAGON_FRUIT_SUFFIX = "no_dragon_fruit_processor"

# XML标签常量
XML_SIZE_TAG = "size"
XML_WIDTH_TAG = "width"
XML_HEIGHT_TAG = "height"
XML_DEPTH_TAG = "depth"
XML_OBJECT_TAG = "object"
XML_NAME_TAG = "name"
XML_BNDBOX_TAG = "bndbox"
XML_XMIN_TAG = "xmin"
XML_YMIN_TAG = "ymin"
XML_XMAX_TAG = "xmax"
XML_YMAX_TAG = "ymax"
XML_DIFFICULT_TAG = "difficult"

# 进度条相关常量
PROGRESS_BAR_UNIT = "文件"
PROGRESS_BAR_DESC_CHECK = "检查图像尺寸"
PROGRESS_BAR_DESC_CONVERT = "转换COCO格式"

# 用户输入常量
USER_INPUT_YES = "Y"
USER_INPUT_NO = "N"

# 分隔符常量
SEPARATOR_LINE = "=" * 60
SEPARATOR_SHORT = "-" * 30
TAB_SEPARATOR = "\t"
NEWLINE = "\n"
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

# 类别统计相关常量
LABELS_TXT = "labels.txt"
COUNT_ALL_CLS_TXT = "count_all_cls.txt"
