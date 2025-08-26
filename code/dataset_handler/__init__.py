"""
数据集处理模块

提供各种数据集格式的处理类：
- VOC格式数据集处理
- COCO格式数据集处理（待实现）
- YOLO格式数据集处理（待实现）
"""

from .voc_dataset import VOCDataset

__all__ = ['VOCDataset']