#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VOC数据集处理器

专门用于对已清洗和划分好的VOC数据集进行高级处理操作，
包括格式转换、类别过滤、一键处理等功能。

Author: CodeBuddy
Date: 2025-08-28
"""

import os
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set
import re
from xml.etree import ElementTree as ET

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from global_var.global_cls import *
from logger_code.logger_sys import get_logger
from dataset_handler.voc_dataset import VOCDataset

# 设置日志
logger = get_logger(Path(__file__).stem)


class VOCDatasetProcessor:
    """
    VOC数据集处理器
    
    专门用于对已清洗和划分好的VOC数据集进行高级处理操作。
    该类假设输入的数据集已经通过VOCDataset类进行了基础清洗和划分。
    """
    
    def __init__(self, dataset_path: str):
        """
        初始化VOC数据集处理器
        
        Args:
            dataset_path (str): 数据集根目录路径，应包含Annotations和JPEGImages目录
        """
        self.dataset_path = Path(dataset_path)
        self.annotations_dir = self.dataset_path / ANNOTATIONS_DIR
        self.images_dir = self.dataset_path / JPEGS_DIR
        self.imagesets_dir = self.dataset_path / IMAGESETS_DIR / MAIN_DIR
        
        # 验证基本结构
        self._validate_structure()
        
        # 内部VOCDataset实例，用于调用基础功能
        self._voc_dataset = None
        
        logger.info(f"VOC数据集处理器初始化完成: {self.dataset_path}")
    
    def _validate_structure(self):
        """验证数据集基本结构"""
        required_dirs = [self.annotations_dir, self.images_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"必需目录不存在: {dir_path}")
        
        # 检查是否有XML和图像文件
        xml_files = list(self.annotations_dir.glob("*.xml"))
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
        
        if not xml_files:
            raise ValueError(f"在{self.annotations_dir}中未找到XML标注文件")
        if not image_files:
            raise ValueError(f"在{self.images_dir}中未找到图像文件")
        
        logger.info(f"数据集结构验证通过 - XML文件: {len(xml_files)}个, 图像文件: {len(image_files)}个")
    
    def _get_voc_dataset(self) -> VOCDataset:
        """获取内部VOCDataset实例，延迟初始化"""
        if self._voc_dataset is None:
            self._voc_dataset = VOCDataset(
                dataset_path=str(self.dataset_path),
                train_ratio=0.85,
                val_ratio=0.15,
                test_ratio=0.0
            )
        return self._voc_dataset
    
    
    def convert_to_coco_format(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        将VOC格式数据集转换为COCO格式
        
        Args:
            output_dir (str, optional): 输出目录，默认为数据集根目录
            
        Returns:
            Dict[str, str]: 生成的COCO JSON文件路径
        """
        logger.info("开始转换VOC格式到COCO格式")
        
        # 使用内部VOCDataset实例进行转换
        voc_dataset = self._get_voc_dataset()
        result = voc_dataset.convert_to_coco_format(output_dir)
        
        logger.info("COCO格式转换完成")
        return result
    
    def get_dataset_info(self) -> Dict:
        """获取数据集基本信息"""
        xml_files = list(self.annotations_dir.glob("*.xml"))
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
        
        # 提取类别信息
        classes = set()
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        classes.add(name_elem.text)
            except Exception:
                continue
        
        return {
            'dataset_path': str(self.dataset_path.absolute()),
            'annotations_dir': str(self.annotations_dir),
            'images_dir': str(self.images_dir),
            'total_xml_files': len(xml_files),
            'total_image_files': len(image_files),
            'total_classes': len(classes),
            'classes': sorted(list(classes))
        }
    
    def print_summary(self):
        """打印数据集处理器摘要信息"""
        info = self.get_dataset_info()
        
        logger.info("=== VOC数据集处理器摘要 ===")
        logger.info(f"数据集路径: {info['dataset_path']}")
        logger.info(f"标注目录: {info['annotations_dir']}")
        logger.info(f"图像目录: {info['images_dir']}")
        logger.info(f"XML文件数: {info['total_xml_files']} 个")
        logger.info(f"图像文件数: {info['total_image_files']} 个")
        logger.info(f"类别数量: {info['total_classes']} 个")
        logger.info(f"类别列表: {', '.join(info['classes'])}")
        
        print("=== VOC数据集处理器摘要 ===")
        print(f"📁 数据集路径: {info['dataset_path']}")
        print(f"📋 XML文件数: {info['total_xml_files']} 个")
        print(f"🖼️  图像文件数: {info['total_image_files']} 个")
        print(f"🏷️  类别数量: {info['total_classes']} 个")
        print(f"📝 类别列表: {', '.join(info['classes'])}")


if __name__ == "__main__":
    # 测试VOC数据集处理器
    dataset_path = "../../dataset/Fruit"
    
    try:
        processor = VOCDatasetProcessor(dataset_path)
        processor.print_summary()
        
        # 测试类别统计功能
        print(f"\n=== 测试类别统计功能 ===")
        result = processor.count_and_sort_classes()
        print(f"统计结果: {result['success']}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误: {e}")