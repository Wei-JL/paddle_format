#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO系列数据格式转换器

用于将经过VOCDataset处理后的数据集转换为YOLO格式
输入：已处理的VOC格式数据集
输出：YOLO格式数据集 (适用于YOLOv6-YOLOv13)
"""

import os
import shutil
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict
from tqdm import tqdm
import sys
from pathlib import Path

# 导入日志系统 - 使用全局变量
sys.path.append(str(Path(__file__).parent.parent))
from logger_code.logger_sys import get_logger
from global_var.global_cls import *

# 获取当前文件名作为日志标识
current_filename = Path(__file__).stem
logger = get_logger(current_filename)


class YOLOSeriesDataset:
    """
    YOLO系列数据格式转换器
    
    功能：将经过VOCDataset处理后的数据集转换为YOLO格式
    输入：已处理的VOC格式数据集
    输出：YOLO格式数据集 (适用于YOLOv6-YOLOv13)
    """
    
    def __init__(self, processed_dataset_path: str, annotations_folder_name: str = ANNOTATIONS_OUTPUT_DIR):
        """
        初始化YOLO转换器
        
        Args:
            processed_dataset_path: 已处理的数据集路径
            annotations_folder_name: 标签文件夹名称 (默认: Annotations_clear)
        """
        self.dataset_path = os.path.abspath(processed_dataset_path)
        self.dataset_name = os.path.basename(os.path.normpath(processed_dataset_path))
        self.annotations_folder_name = annotations_folder_name
        
        # 输入路径 - 使用全局常量
        self.annotations_dir = os.path.join(self.dataset_path, annotations_folder_name)
        self.images_dir = os.path.join(self.dataset_path, JPEGS_DIR)
        self.imagesets_dir = os.path.join(self.dataset_path, IMAGESETS_DIR, MAIN_DIR)
        
        # 输出路径
        self.output_dir = os.path.join("output", f"{self.dataset_name}_yolo")
        self.output_images_dir = os.path.join(self.output_dir, "images")
        self.output_labels_dir = os.path.join(self.output_dir, "labels")
        
        # 类别映射
        self.class_to_id = {}
        
        logger.info(f"初始化YOLO转换器: {self.dataset_name}")
        logger.info(f"输入路径: {self.dataset_path}")
        logger.info(f"输出路径: {self.output_dir}")
    
    def _create_output_directories(self):
        """创建输出目录结构"""
        directories = [
            self.output_dir,
            os.path.join(self.output_images_dir, "train"),
            os.path.join(self.output_images_dir, "val"),
            os.path.join(self.output_labels_dir, "train"),
            os.path.join(self.output_labels_dir, "val"),
        ]
        
        # 检查是否有测试集
        test_file = os.path.join(self.imagesets_dir, TEST_TXT)
        if os.path.exists(test_file):
            directories.extend([
                os.path.join(self.output_images_dir, "test"),
                os.path.join(self.output_labels_dir, "test"),
            ])
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _build_class_mapping(self):
        """构建类别映射表"""
        all_classes = set()
        
        # 遍历所有XML文件收集类别
        if os.path.exists(self.annotations_dir):
            for xml_file in os.listdir(self.annotations_dir):
                if xml_file.lower().endswith(XML_EXTENSION):
                    xml_path = os.path.join(self.annotations_dir, xml_file)
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        for obj in root.findall(XML_OBJECT_TAG):
                            class_name = obj.find(XML_NAME_TAG).text
                            all_classes.add(class_name)
                    except Exception as e:
                        logger.warning(f"解析XML文件失败: {xml_file}, 错误: {str(e)}")
        
        # 构建映射表
        sorted_classes = sorted(list(all_classes))
        self.class_to_id = {cls: idx for idx, cls in enumerate(sorted_classes)}
        
        logger.info(f"类别映射表: {self.class_to_id}")
    
    def _convert_xml_to_yolo(self, xml_file: str) -> Optional[List[str]]:
        """
        将XML标注转换为YOLO格式
        
        Args:
            xml_file: XML文件名
            
        Returns:
            YOLO格式标注行列表
        """
        xml_path = os.path.join(self.annotations_dir, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像尺寸
            size = root.find(XML_SIZE_TAG)
            if size is None:
                return None
                
            img_width = int(size.find(XML_WIDTH_TAG).text)
            img_height = int(size.find(XML_HEIGHT_TAG).text)
            
            yolo_lines = []
            
            # 转换每个目标
            for obj in root.findall(XML_OBJECT_TAG):
                class_name = obj.find(XML_NAME_TAG).text
                
                if class_name not in self.class_to_id:
                    continue
                
                class_id = self.class_to_id[class_name]
                
                # 获取边界框
                bbox = obj.find(XML_BNDBOX_TAG)
                if bbox is None:
                    continue
                
                xmin = float(bbox.find(XML_XMIN_TAG).text)
                ymin = float(bbox.find(XML_YMIN_TAG).text)
                xmax = float(bbox.find(XML_XMAX_TAG).text)
                ymax = float(bbox.find(XML_YMAX_TAG).text)
                
                # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                center_x = (xmin + xmax) / 2.0 / img_width
                center_y = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
            
            return yolo_lines
            
        except Exception as e:
            logger.error(f"转换XML文件失败: {xml_file}, 错误: {str(e)}")
            return None
    
    def _parse_split_file(self, split_file_path: str) -> List[str]:
        """
        解析数据集划分文件，提取文件名
        
        Args:
            split_file_path: 划分文件路径
            
        Returns:
            文件名列表
        """
        file_names = []
        
        try:
            with open(split_file_path, 'r', encoding=DEFAULT_ENCODING) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 处理包含路径的格式: JPEGImages\fruit0.png	Annotations_clear\fruit0.xml
                    if TAB_SEPARATOR in line:
                        parts = line.split(TAB_SEPARATOR)
                        if len(parts) >= 1:
                            # 从图像路径中提取文件名
                            image_path = parts[0]
                            # 提取文件名（不含扩展名）
                            filename = os.path.splitext(os.path.basename(image_path))[0]
                            file_names.append(filename)
                    else:
                        # 处理简单文件名格式
                        filename = os.path.splitext(line)[0]
                        file_names.append(filename)
                        
        except Exception as e:
            logger.error(f"解析划分文件失败: {split_file_path}, 错误: {str(e)}")
        
        return file_names
    
    def _process_split(self, split_name: str):
        """
        处理单个数据集划分
        
        Args:
            split_name: 划分名称 (train/val/test)
        """
        # 构建划分文件路径
        if split_name == "train":
            split_file = os.path.join(self.imagesets_dir, TRAIN_TXT)
        elif split_name == "val":
            split_file = os.path.join(self.imagesets_dir, VAL_TXT)
        elif split_name == "test":
            split_file = os.path.join(self.imagesets_dir, TEST_TXT)
        else:
            logger.error(f"不支持的划分类型: {split_name}")
            return
        
        if not os.path.exists(split_file):
            logger.warning(f"划分文件不存在: {split_file}")
            return
        
        # 解析文件列表
        file_names = self._parse_split_file(split_file)
        
        logger.info(f"处理 {split_name} 数据集: {len(file_names)} 个文件")
        
        success_count = 0
        
        for file_name in tqdm(file_names, desc=f"转换{split_name}数据"):
            # 转换XML标注
            xml_file = f"{file_name}{XML_EXTENSION}"
            yolo_lines = self._convert_xml_to_yolo(xml_file)
            
            if yolo_lines is None or len(yolo_lines) == 0:
                logger.warning(f"跳过无效文件: {file_name}")
                continue
            
            # 查找对应的图像文件
            source_image_path = None
            
            for ext in IMAGE_EXTENSIONS:
                potential_path = os.path.join(self.images_dir, f"{file_name}{ext}")
                if os.path.exists(potential_path):
                    source_image_path = potential_path
                    break
            
            if not source_image_path:
                logger.warning(f"未找到图像文件: {file_name}")
                continue
            
            # 复制图像文件
            image_ext = os.path.splitext(source_image_path)[1]
            target_image_path = os.path.join(self.output_images_dir, split_name, f"{file_name}{image_ext}")
            shutil.copy2(source_image_path, target_image_path)
            
            # 保存YOLO标注文件
            target_label_path = os.path.join(self.output_labels_dir, split_name, f"{file_name}.txt")
            with open(target_label_path, 'w', encoding=DEFAULT_ENCODING) as f:
                f.write(NEWLINE.join(yolo_lines))
            
            success_count += 1
        
        logger.info(f"{split_name} 数据集处理完成: {success_count}/{len(file_names)} 个文件成功")
    
    def _create_yaml_config(self):
        """创建YOLO数据集配置文件"""
        yaml_content = {
            'path': os.path.abspath(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_to_id),
            'names': list(self.class_to_id.keys())
        }
        
        # 检查是否有测试集
        test_images_dir = os.path.join(self.output_images_dir, "test")
        if os.path.exists(test_images_dir):
            yaml_content['test'] = 'images/test'
        
        yaml_path = os.path.join(self.output_dir, f"{self.dataset_name}.yaml")
        
        # 写入YAML文件
        with open(yaml_path, 'w', encoding=DEFAULT_ENCODING) as f:
            f.write(f"# YOLO数据集配置文件{NEWLINE}")
            f.write(f"# 数据集: {self.dataset_name}{NEWLINE}")
            f.write(f"# 生成时间: 2025-09-01{NEWLINE}{NEWLINE}")
            
            f.write(f"path: {yaml_content['path']}{NEWLINE}")
            f.write(f"train: {yaml_content['train']}{NEWLINE}")
            f.write(f"val: {yaml_content['val']}{NEWLINE}")
            if 'test' in yaml_content:
                f.write(f"test: {yaml_content['test']}{NEWLINE}")
            f.write(f"{NEWLINE}")
            f.write(f"nc: {yaml_content['nc']}{NEWLINE}")
            f.write(f"names: {yaml_content['names']}{NEWLINE}")
        
        logger.info(f"YOLO配置文件已保存: {yaml_path}")
    
    def _create_label_mapping(self):
        """创建标签映射文件"""
        mapping_path = os.path.join(self.output_dir, "label_mapping.txt")
        
        with open(mapping_path, 'w', encoding=DEFAULT_ENCODING) as f:
            for class_name, class_id in self.class_to_id.items():
                f.write(f'{class_id}: "{class_name}"{NEWLINE}')
        
        logger.info(f"标签映射文件已保存: {mapping_path}")
    
    def convert_to_yolo(self) -> bool:
        """
        转换为YOLO格式
        
        Returns:
            转换是否成功
        """
        try:
            logger.info("开始YOLO格式转换...")
            
            # 检查输入路径
            if not os.path.exists(self.annotations_dir):
                logger.error(f"标注文件夹不存在: {self.annotations_dir}")
                return False
            
            if not os.path.exists(self.images_dir):
                logger.error(f"图像文件夹不存在: {self.images_dir}")
                return False
            
            if not os.path.exists(self.imagesets_dir):
                logger.error(f"数据集划分文件夹不存在: {self.imagesets_dir}")
                return False
            
            # 创建输出目录
            self._create_output_directories()
            
            # 构建类别映射
            self._build_class_mapping()
            
            if not self.class_to_id:
                logger.error("未找到有效类别")
                return False
            
            # 处理各个数据集划分
            splits = ['train', 'val']
            test_file = os.path.join(self.imagesets_dir, TEST_TXT)
            if os.path.exists(test_file):
                splits.append('test')
            
            for split in splits:
                self._process_split(split)
            
            # 创建配置文件
            self._create_yaml_config()
            
            # 创建标签映射文件
            self._create_label_mapping()
            
            logger.info("YOLO格式转换完成!")
            self._print_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"YOLO格式转换失败: {str(e)}")
            return False
    
    def _print_summary(self):
        """打印转换摘要"""
        try:
            logger.info("转换摘要:")
            logger.info(f"  数据集名称: {self.dataset_name}")
            logger.info(f"  输出目录: {self.output_dir}")
            logger.info(f"  类别数量: {len(self.class_to_id)}")
            logger.info(f"  类别列表: {list(self.class_to_id.keys())}")
            
            # 统计文件数量
            for split in ['train', 'val', 'test']:
                images_dir = os.path.join(self.output_images_dir, split)
                labels_dir = os.path.join(self.output_labels_dir, split)
                
                if os.path.exists(images_dir):
                    image_count = len([f for f in os.listdir(images_dir) 
                                     if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)])
                    label_count = len([f for f in os.listdir(labels_dir) 
                                     if f.lower().endswith('.txt')]) if os.path.exists(labels_dir) else 0
                    
                    logger.info(f"  {split}集: {image_count} 张图片, {label_count} 个标签文件")
            
        except Exception as e:
            logger.error(f"打印摘要失败: {str(e)}")