# Paddle Format - 多格式数据集处理工具

一个功能全面、高度自动化的数据集处理工具，支持VOC到COCO/YOLO格式转换、数据清洗、标签过滤和一键处理，专为深度学习计算机视觉任务设计。

## 🚀 功能特性

### 核心功能
- **数据集验证**: 检查VOC数据集的结构完整性。
- **文件匹配与清洗**: 自动匹配图像和标注文件，并清理无效或空的标注。
- **标签过滤 (高级)**:
  - **`exclude_labels`**: 排除指定的标签类别。
  - **`include_labels`**: 只保留指定的标签类别。
- **线程池并发**: 使用多线程加速XML文件的读写和处理。
- **XML格式保持**: 在修改XML文件后，完美保留其原有的缩进和换行格式。
- **图像修正**: 自动修正图像尺寸不匹配和通道数问题。
- **数据集划分**: 智能划分训练集、验证集和测试集。
- **多格式转换**: 
  - **VOC → COCO**: 支持PaddleDetection训练格式
  - **VOC → YOLO**: 支持YOLOv6-YOLOv13系列训练格式

## 🔧 支持的深度学习框架

### PaddleDetection (百度飞桨)
- **GitHub**: [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- 支持完整的COCO格式转换
- 兼容PaddleDetection的数据加载要求
- 提供训练集/验证集/测试集划分

### YOLO系列 (PyTorch)
- **支持版本**: YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv13
- **YOLOv13 GitHub**: [https://github.com/WongKinYiu/yolov13](https://github.com/WongKinYiu/yolov13)
- **Ultralytics GitHub**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- 自动生成YAML配置文件
- 支持标准YOLO标注格式 (归一化坐标)
- 兼容所有主流YOLO实现

### ⚡ 一键化处理
- **`one_click_complete_conversion()`**: 一键完成从数据验证、清洗、过滤、划分到格式转换的全过程。
- **安全提醒**: 在执行任何可能修改数据的操作前，强制要求用户确认备份。
- **详细日志**: 为所有操作提供完整的日志记录，便于追踪和调试。

## 📂 项目结构

```
paddle_format/
├── code/
│   ├── dataset_handler/          # 数据集处理核心模块
│   │   ├── voc_dataset.py       # VOC数据集核心处理类 (COCO格式转换)
│   │   └── yolo_series_dataset.py # YOLO系列数据集处理类
│   ├── global_var/              # 全局变量配置
│   ├── logger_code/             # 日志系统
│   ├── use_code/                           # 使用示例代码
│   │   ├── simple_process_example.py       # 简单一键清洗+转换示例
│   │   └── label_filtering_example.py      # 类别筛选使用示例
│   └── logs/                    # 代码日志
├── dataset/                     # 示例数据集目录
├── docs/                        # 项目文档
├── logs/                        # 全局日志目录
└── output/                      # 输出文件目录
```

## 🛠️ 快速开始

### 环境要求
- Python 3.7+
- OpenCV (`opencv-python`)
- `tqdm`
- `numpy`

### 安装
```bash
pip install opencv-python tqdm numpy
```

### 使用示例

#### 1. 基础一键处理
```python
from code.dataset_handler.voc_dataset import VOCDataset

# 初始化VOCDataset对象，指定数据集路径
voc_dataset = VOCDataset(dataset_path="dataset/Fruit")

# 执行一键转换，完成所有处理步骤
voc_dataset.one_click_complete_conversion()
```

#### 2. 高级标签过滤

**只保留 `pineapple` 和 `snake fruit` 两个类别:**
```python
from code.dataset_handler.voc_dataset import VOCDataset

dataset = VOCDataset(
    dataset_path="dataset/Fruit",
    # 只保留这两个标签
    include_labels=["pineapple", "snake fruit"],
    # 指定过滤后的新标注文件夹名称
    output_annotations_name="Annotations_filtered"
)

# 执行处理
dataset.one_click_complete_conversion()
```

**排除 `banana` 类别:**
```python
from code.dataset_handler.voc_dataset import VOCDataset

dataset = VOCDataset(
    dataset_path="dataset/Fruit",
    # 排除这个标签
    exclude_labels=["banana"],
    # 指定过滤后的新标注文件夹名称
    output_annotations_name="Annotations_no_banana"
)

# 执行处理
dataset.one_click_complete_conversion()
```

## 📖 文档

- **[使用指南.md](docs/使用指南.md)**: 详细的功能介绍和使用示例。
- **[文件树文档.md](docs/文件树文档.md)**: 项目文件结构和核心模块的详细说明。

## 🧪 测试

项目提供了丰富的测试脚本，位于 `code/use_code/` 目录下，覆盖了所有核心功能。

```bash
# 运行标签过滤功能测试
python code/use_code/test_label_filter.py
```

## 🤝 贡献

欢迎通过提交Issue和Pull Request来改进此项目。

## 📄 许可证

MIT License

## 👨‍💻 作者

Wei-JL

---

**如果您觉得这个项目对您有帮助，请给它一个 ⭐️！**