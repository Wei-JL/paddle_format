"""
YOLO系列数据集处理器 (YOLOv6-YOLOv13通用格式)
支持从VOC格式转换为YOLO系列训练格式
作者: CodeBuddy
版本: 1.0.0
"""

import os
import json
import shutil
import yaml
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from PIL import Image
import logging

from ..global_var.global_cls import GlobalConfig
from ..logger_code.logger_config import setup_logger


class YOLOSeriesDataset:
    """
    YOLO系列数据集处理器
    支持YOLOv6-YOLOv13通用格式转换
    
    主要功能:
    1. VOC格式转YOLO格式
    2. 数据集划分 (train/val/test)
    3. 类别筛选 (include_labels/exclude_labels)
    4. 数据清洗和验证
    5. 生成YOLO配置文件
    """
    
    def __init__(self, dataset_path: str, train_ratio: float = 0.8, 
                 val_ratio: float = 0.2, test_ratio: float = 0.0):
        """
        初始化YOLO数据集处理器
        
        Args:
            dataset_path: 数据集根目录路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.dataset_name = os.path.basename(self.dataset_path)
        
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("训练集、验证集和测试集比例之和必须等于1")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 初始化路径
        self._init_paths()
        
        # 初始化日志
        self.logger = setup_logger(
            name=f"YOLODataset_{self.dataset_name}",
            log_file=os.path.join(GlobalConfig.LOG_DIR, f"yolo_dataset_{self.dataset_name}.log")
        )
        
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        
        # 类别信息
        self.class_names = []
        self.class_to_id = {}
        self.id_to_class = {}
        
        self.logger.info(f"初始化YOLO数据集处理器: {self.dataset_name}")
        self.logger.info(f"数据集划分比例 - 训练集: {train_ratio}, 验证集: {val_ratio}, 测试集: {test_ratio}")
    
    def _init_paths(self):
        """初始化所有路径"""
        # 输入路径
        self.annotations_dir = os.path.join(self.dataset_path, GlobalConfig.ANNOTATIONS_DIR)
        self.images_dir = os.path.join(self.dataset_path, GlobalConfig.IMAGES_DIR)
        self.imagesets_dir = os.path.join(self.dataset_path, GlobalConfig.IMAGESETS_DIR, GlobalConfig.MAIN_DIR)
        
        # YOLO输出路径
        self.yolo_output_dir = os.path.join(self.dataset_path, "yolo_format")
        self.yolo_images_dir = os.path.join(self.yolo_output_dir, "images")
        self.yolo_labels_dir = os.path.join(self.yolo_output_dir, "labels")
        
        # 创建输出目录
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.yolo_images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_labels_dir, split), exist_ok=True)
    
    def _scan_classes_from_annotations(self) -> Set[str]:
        """扫描所有XML文件获取类别信息"""
        classes = set()
        
        if not os.path.exists(self.annotations_dir):
            self.logger.warning(f"标注目录不存在: {self.annotations_dir}")
            return classes
            
        xml_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(self.annotations_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        classes.add(name_elem.text.strip())
                        
            except Exception as e:
                self.logger.error(f"解析XML文件失败 {xml_file}: {e}")
                
        self.logger.info(f"发现 {len(classes)} 个类别: {sorted(classes)}")
        return classes
    
    def _init_class_mapping(self, include_labels: Optional[List[str]] = None,
                          exclude_labels: Optional[List[str]] = None):
        """初始化类别映射"""
        all_classes = self._scan_classes_from_annotations()
        
        # 应用类别筛选
        if include_labels:
            filtered_classes = set(include_labels) & all_classes
            self.logger.info(f"包含指定类别: {include_labels}")
        elif exclude_labels:
            filtered_classes = all_classes - set(exclude_labels)
            self.logger.info(f"排除指定类别: {exclude_labels}")
        else:
            filtered_classes = all_classes
            
        # 创建类别映射
        self.class_names = sorted(list(filtered_classes))
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        
        self.logger.info(f"最终使用 {len(self.class_names)} 个类别: {self.class_names}")
    
    def _parse_xml_annotation(self, xml_path: str) -> Optional[Dict]:
        """解析XML标注文件"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像信息
            filename = root.find('filename').text if root.find('filename') is not None else None
            size_elem = root.find('size')
            
            if size_elem is None:
                return None
                
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
            
            # 解析目标框
            objects = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None or name_elem.text.strip() not in self.class_to_id:
                    continue
                    
                class_name = name_elem.text.strip()
                class_id = self.class_to_id[class_name]
                
                bbox_elem = obj.find('bndbox')
                if bbox_elem is None:
                    continue
                    
                xmin = float(bbox_elem.find('xmin').text)
                ymin = float(bbox_elem.find('ymin').text)
                xmax = float(bbox_elem.find('xmax').text)
                ymax = float(bbox_elem.find('ymax').text)
                
                # 转换为YOLO格式 (中心点坐标 + 宽高，归一化)
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                objects.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': bbox_width,
                    'height': bbox_height
                })
            
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects
            }
            
        except Exception as e:
            self.logger.error(f"解析XML文件失败 {xml_path}: {e}")
            return None
    
    def _convert_single_annotation(self, xml_file: str, split: str) -> bool:
        """转换单个标注文件"""
        try:
            xml_path = os.path.join(self.annotations_dir, xml_file)
            annotation_data = self._parse_xml_annotation(xml_path)
            
            if not annotation_data or not annotation_data['objects']:
                return False
                
            # 查找对应的图像文件
            base_name = os.path.splitext(xml_file)[0]
            image_file = None
            
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image = base_name + ext
                if os.path.exists(os.path.join(self.images_dir, potential_image)):
                    image_file = potential_image
                    break
                    
            if not image_file:
                self.logger.warning(f"找不到对应的图像文件: {base_name}")
                return False
            
            # 复制图像文件
            src_image_path = os.path.join(self.images_dir, image_file)
            dst_image_path = os.path.join(self.yolo_images_dir, split, image_file)
            shutil.copy2(src_image_path, dst_image_path)
            
            # 生成YOLO标注文件
            label_file = base_name + '.txt'
            label_path = os.path.join(self.yolo_labels_dir, split, label_file)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for obj in annotation_data['objects']:
                    line = f"{obj['class_id']} {obj['x_center']:.6f} {obj['y_center']:.6f} {obj['width']:.6f} {obj['height']:.6f}\n"
                    f.write(line)
            
            return True
            
        except Exception as e:
            self.logger.error(f"转换标注文件失败 {xml_file}: {e}")
            return False
    
    def _read_split_file(self, split: str) -> List[str]:
        """读取数据集划分文件"""
        split_file = os.path.join(self.imagesets_dir, f"{split}.txt")
        
        if not os.path.exists(split_file):
            self.logger.warning(f"划分文件不存在: {split_file}")
            return []
            
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                return [line.strip() + '.xml' for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"读取划分文件失败 {split_file}: {e}")
            return []
    
    def _convert_split_to_yolo(self, xml_files: List[str], split: str) -> int:
        """转换指定划分的数据到YOLO格式"""
        if not xml_files:
            return 0
            
        self.logger.info(f"开始转换 {split} 集，共 {len(xml_files)} 个文件")
        
        # 使用线程池并发处理
        futures = []
        for xml_file in xml_files:
            future = self.executor.submit(self._convert_single_annotation, xml_file, split)
            futures.append(future)
        
        success_count = 0
        for future in as_completed(futures):
            if future.result():
                success_count += 1
        
        self.logger.info(f"{split} 集转换完成，成功转换 {success_count}/{len(xml_files)} 个文件")
        return success_count
    
    def _generate_yaml_config(self) -> str:
        """生成YOLO配置文件"""
        config = {
            'path': self.yolo_output_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        if self.test_ratio > 0:
            config['test'] = 'images/test'
        
        yaml_path = os.path.join(self.yolo_output_dir, f"{self.dataset_name}.yaml")
        
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"生成YOLO配置文件: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            self.logger.error(f"生成YAML配置文件失败: {e}")
            return ""
    
    def _generate_statistics(self) -> Dict:
        """生成转换统计信息"""
        stats = {
            'dataset_name': self.dataset_name,
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'splits': {}
        }
        
        for split in ['train', 'val', 'test']:
            images_dir = os.path.join(self.yolo_images_dir, split)
            labels_dir = os.path.join(self.yolo_labels_dir, split)
            
            if os.path.exists(images_dir):
                image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                
                stats['splits'][split] = {
                    'images': image_count,
                    'labels': label_count
                }
        
        return stats
    
    def convert_to_yolo(self, include_labels: Optional[List[str]] = None,
                       exclude_labels: Optional[List[str]] = None) -> bool:
        """
        转换VOC格式数据集为YOLO格式
        
        Args:
            include_labels: 包含的类别列表
            exclude_labels: 排除的类别列表
            
        Returns:
            bool: 转换是否成功
        """
        try:
            self.logger.info("开始VOC到YOLO格式转换")
            
            # 初始化类别映射
            self._init_class_mapping(include_labels, exclude_labels)
            
            if not self.class_names:
                self.logger.error("没有找到有效的类别")
                return False
            
            # 转换各个数据集划分
            total_converted = 0
            
            # 训练集
            train_files = self._read_split_file('train')
            if train_files:
                total_converted += self._convert_split_to_yolo(train_files, 'train')
            
            # 验证集
            val_files = self._read_split_file('val')
            if val_files:
                total_converted += self._convert_split_to_yolo(val_files, 'val')
            
            # 测试集
            if self.test_ratio > 0:
                test_files = self._read_split_file('test')
                if test_files:
                    total_converted += self._convert_split_to_yolo(test_files, 'test')
            
            # 生成配置文件
            yaml_path = self._generate_yaml_config()
            
            # 生成统计信息
            stats = self._generate_statistics()
            stats_path = os.path.join(self.yolo_output_dir, "conversion_stats.json")
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"YOLO格式转换完成！")
            self.logger.info(f"总共转换 {total_converted} 个样本")
            self.logger.info(f"输出目录: {self.yolo_output_dir}")
            self.logger.info(f"配置文件: {yaml_path}")
            self.logger.info(f"统计信息: {stats_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"YOLO格式转换失败: {e}")
            return False
    
    def one_click_complete_conversion(self, include_labels: Optional[List[str]] = None,
                                    exclude_labels: Optional[List[str]] = None) -> bool:
        """
        一键完成完整的YOLO格式转换流程
        
        Args:
            include_labels: 包含的类别列表
            exclude_labels: 排除的类别列表
            
        Returns:
            bool: 转换是否成功
        """
        try:
            self.logger.info("=" * 50)
            self.logger.info("开始一键YOLO格式转换流程")
            self.logger.info("=" * 50)
            
            # 执行转换
            success = self.convert_to_yolo(include_labels, exclude_labels)
            
            if success:
                self.logger.info("=" * 50)
                self.logger.info("YOLO格式转换流程完成！")
                self.logger.info("=" * 50)
                
                # 打印统计信息
                stats = self._generate_statistics()
                print(f"\n转换统计:")
                print(f"数据集: {stats['dataset_name']}")
                print(f"类别数: {stats['total_classes']}")
                print(f"类别列表: {stats['class_names']}")
                
                for split, info in stats['splits'].items():
                    if info['images'] > 0:
                        print(f"{split}集: {info['images']} 张图片, {info['labels']} 个标注文件")
                
                print(f"\n输出目录: {self.yolo_output_dir}")
                print(f"配置文件: {os.path.join(self.yolo_output_dir, f'{self.dataset_name}.yaml')}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"一键转换流程失败: {e}")
            return False
    
    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)