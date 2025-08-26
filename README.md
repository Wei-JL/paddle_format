# Paddle Format - VOC数据集处理工具

一个功能完整的VOC格式数据集处理工具，支持数据清洗、格式转换和一键处理功能。

## 功能特性

### 🔧 核心功能
- **数据集验证** - 检查VOC数据集基本结构完整性
- **文件匹配** - 自动匹配图像和标注文件
- **空标注清理** - 检测并删除空标注文件
- **类别管理** - 提取和过滤数据集类别
- **图像修正** - 自动修正图像尺寸和通道数
- **数据集划分** - 智能划分训练集、验证集和测试集
- **格式转换** - VOC到COCO格式转换

### ⚡ 一键处理功能
- **安全提醒** - 强制备份确认机制
- **批量处理** - 一键完成所有数据清洗和转换步骤
- **详细日志** - 完整的处理过程记录
- **错误处理** - 完善的异常处理和恢复机制

## 项目结构

```
paddle_format/
├── code/
│   ├── dataset_handler/          # 数据集处理核心模块
│   │   ├── voc_dataset.py       # VOC数据集处理类
│   │   └── one_click_functions.py # 一键处理功能
│   ├── global_var/              # 全局变量配置
│   ├── logger_code/             # 日志系统
│   ├── logs/                    # 日志文件目录
│   └── use_code/                # 使用示例和测试代码
├── dataset/                     # 数据集目录
├── docs/                        # 文档目录
├── logs/                        # 全局日志目录
└── output/                      # 输出目录
```

## 快速开始

### 环境要求
- Python 3.7+
- OpenCV (cv2)
- tqdm
- numpy

### 安装依赖
```bash
pip install opencv-python tqdm numpy
```

### 基本使用

```python
from code.dataset_handler.voc_dataset import VOCDataset
from code.dataset_handler.one_click_functions import one_click_process_dataset

# 初始化数据集
dataset_path = "path/to/your/voc/dataset"
voc_dataset = VOCDataset(dataset_path)

# 查看数据集信息
voc_dataset.print_summary()

# 一键处理数据集
result = one_click_process_dataset(
    voc_dataset,
    exclude_classes=None,        # 可选：排除的类别列表
    auto_fix_images=True,        # 自动修正图像
    new_annotations_suffix="processed"
)
```

### 高级功能

#### 类别过滤
```python
# 过滤指定类别
exclude_classes = ['unwanted_class1', 'unwanted_class2']
result = one_click_process_dataset(
    voc_dataset,
    exclude_classes=exclude_classes,
    auto_fix_images=True,
    new_annotations_suffix="filtered"
)
```

#### 图像尺寸检查和修正
```python
# 检查图像尺寸（不修正）
stats = voc_dataset.check_and_fix_image_dimensions(auto_fix=False)

# 自动修正图像尺寸和XML文件
stats = voc_dataset.check_and_fix_image_dimensions(auto_fix=True)
```

#### COCO格式转换
```python
# 转换为COCO格式
coco_files = voc_dataset.convert_to_coco_format()
print("生成的COCO文件:", coco_files)
```

## 数据集格式要求

### VOC格式目录结构
```
your_dataset/
├── Annotations/          # XML标注文件
├── JPEGImages/          # 图像文件
└── ImageSets/
    └── Main/            # 数据集划分文件（自动生成）
        ├── train.txt
        ├── val.txt
        ├── test.txt
        ├── trainval.txt
        └── labels.txt
```

### 支持的图像格式
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

## 安全特性

- **备份提醒** - 处理前强制确认数据备份
- **用户确认** - 关键操作需要用户明确确认
- **详细日志** - 完整记录所有处理步骤
- **错误恢复** - 完善的异常处理机制

## 测试

运行测试脚本：
```bash
cd code/use_code
python test_one_click_process.py
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License

## 作者

Wei-JL

---

如果您觉得这个项目有用，请给它一个⭐️！