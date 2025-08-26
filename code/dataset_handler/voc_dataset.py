"""
VOC数据集处理类

用于处理Pascal VOC格式的数据集，提供：
- 数据集基本验证
- 图像和标注文件匹配
- 空标注文件检测和删除
- 数据集划分功能
- 类别提取功能
- 类别过滤功能
- 图像尺寸检查和修正功能
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Set, Dict
import sys
import random
import shutil
import cv2
from tqdm import tqdm
from tqdm import tqdm
import numpy as np

# 导入日志系统 - 使用全局变量
sys.path.append(str(Path(__file__).parent.parent))
from logger_code.logger_sys import get_logger
from global_var.global_cls import *

# 获取当前文件名作为日志标识
current_filename = Path(__file__).stem
logger = get_logger(current_filename)


class VOCDataset:
    """VOC数据集处理类"""
    
    def __init__(self, dataset_path: str, train_ratio: float = TRAIN_RATIO, 
                 val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO):
        """
        初始化VOC数据集
        
        Args:
            dataset_path: 数据集根目录路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        
        # 数据集划分比例
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 标准VOC目录结构
        self.annotations_dir = self.dataset_path / ANNOTATIONS_DIR
        self.images_dir = self.dataset_path / JPEGS_DIR
        self.imagesets_dir = self.dataset_path / IMAGESETS_DIR / MAIN_DIR
        
        # 记录缺失文件的列表
        self.missing_xml_files = []  # 缺少XML文件的图像
        self.missing_image_files = []  # 缺少图像文件的XML
        
        # 有效的文件对列表
        self.valid_pairs = []
        
        # 类别集合
        self.classes = set()
        
        logger.info(f"初始化VOC数据集: {self.dataset_name}")
        logger.info(f"数据集路径: {self.dataset_path.absolute()}")
        logger.info(f"划分比例 - 训练集: {self.train_ratio}, 验证集: {self.val_ratio}, 测试集: {self.test_ratio}")
        
        # 验证数据集基本结构
        self._validate_basic_structure()
        
        # 匹配图像和标注文件
        self._match_files()
        
        # 删除空标注文件
        self._remove_empty_annotations()
        
        # 提取类别信息
        self._extract_classes()
        
        # 数据集划分
        self._split_dataset()
    
    def _validate_basic_structure(self):
        """验证数据集基本结构"""
        logger.info("验证数据集基本结构...")
        
        # 检查必要目录是否存在
        if not self.annotations_dir.exists():
            error_msg = f"标注目录不存在: {self.annotations_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if not self.images_dir.exists():
            error_msg = f"图像目录不存在: {self.images_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.debug(f"目录检查通过: {ANNOTATIONS_DIR}, {JPEGS_DIR}")
        
        # 检查是否至少有一张图片
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            error_msg = f"图像目录中没有找到图片文件: {self.images_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 检查是否至少有一个XML文件
        xml_files = list(self.annotations_dir.glob(f"*{XML_EXTENSION}"))
        if not xml_files:
            error_msg = f"标注目录中没有找到XML文件: {self.annotations_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"基本结构验证通过 - 图像文件: {len(image_files)} 个, XML文件: {len(xml_files)} 个")
    
    def _match_files(self):
        """匹配图像和标注文件"""
        logger.info("开始匹配图像和标注文件...")
        
        # 获取所有图像文件
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        # 获取所有XML文件
        xml_files = list(self.annotations_dir.glob(f"*{XML_EXTENSION}"))
        
        # 创建文件名映射（不含扩展名）
        image_stems = {f.stem: f for f in image_files}
        xml_stems = {f.stem: f for f in xml_files}
        
        # 检查缺少XML的图像文件
        for stem, image_file in image_stems.items():
            if stem not in xml_stems:
                self.missing_xml_files.append(image_file)
                logger.warning(f"图像文件缺少对应的XML标注: {image_file.name}")
        
        # 检查缺少图像的XML文件
        for stem, xml_file in xml_stems.items():
            if stem not in image_stems:
                self.missing_image_files.append(xml_file)
                logger.error(f"XML文件缺少对应的图像文件: {xml_file.name}")
        
        # 如果有XML没有对应图片，报错
        if self.missing_image_files:
            error_msg = f"发现 {len(self.missing_image_files)} 个XML文件没有对应的图像文件"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 收集有效的文件对
        for stem in image_stems:
            if stem in xml_stems:
                self.valid_pairs.append((image_stems[stem], xml_stems[stem]))
        
        # 统计匹配结果
        matched_count = len(self.valid_pairs)
        logger.info(f"文件匹配完成 - 匹配对数: {matched_count}, 缺少XML: {len(self.missing_xml_files)}")
    
    def _remove_empty_annotations(self):
        """删除空标注文件"""
        logger.info("检查并删除空标注文件...")
        
        empty_files = []
        valid_pairs_after_cleanup = []
        
        # 使用进度条显示检查进度
        with tqdm(total=len(self.valid_pairs), desc="检查图像尺寸", unit="文件", leave=False) as pbar:
            for image_file, xml_file in self.valid_pairs:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # 检查是否有object标签
                    objects = root.findall('object')
                    if not objects:
                        empty_files.append(xml_file)
                        logger.warning(f"发现空标注文件: {xml_file.name}")
                        
                        # 删除空标注文件
                        xml_file.unlink()
                        logger.info(f"已删除空标注文件: {xml_file.name}")
                    else:
                        # 保留有效的文件对
                        valid_pairs_after_cleanup.append((image_file, xml_file))
                        
                except Exception as e:
                    logger.error(f"解析XML文件失败: {xml_file.name} - {e}")
                
                pbar.update(1)
        
        # 更新有效文件对列表
        self.valid_pairs = valid_pairs_after_cleanup
        
        if empty_files:
            logger.info(f"共删除 {len(empty_files)} 个空标注文件")
        else:
            logger.info("未发现空标注文件")
    
    def _extract_classes(self):
        """提取所有类别信息"""
        logger.info("开始提取类别信息...")
        
        self.classes = set()
        
        # 使用进度条显示类别提取进度
        with tqdm(total=len(self.valid_pairs), desc="提取类别信息", unit="文件", leave=False) as pbar:
            for _, xml_file in self.valid_pairs:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # 查找所有object标签中的name
                    for obj in root.findall('object'):
                        name_elem = obj.find('name')
                        if name_elem is not None and name_elem.text:
                            class_name = name_elem.text.strip()
                            self.classes.add(class_name)
                            
                except Exception as e:
                    logger.error(f"解析XML文件失败: {xml_file.name} - {e}")
                
                pbar.update(1)
        
        logger.info(f"类别提取完成 - 发现 {len(self.classes)} 个类别: {sorted(self.classes)}")
        
        # 写入labels.txt文件
        self._write_labels_file()
    
    def _write_labels_file(self):
        """写入类别标签文件"""
        logger.info("写入类别标签文件...")
        
        # 创建ImageSets/Main目录
        self.imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # 对标签进行排序（数字优先，然后字母）
        sorted_labels = sorted(self.classes, key=lambda x: (x.isdigit(), x))
        
        labels_file_path = self.imagesets_dir / "labels.txt"
        
        try:
            with open(labels_file_path, 'w', encoding=DEFAULT_ENCODING) as f:
                for i, label in enumerate(sorted_labels):
                    if i == len(sorted_labels) - 1:
                        # 最后一行不换行
                        f.write(label)
                    else:
                        f.write(f"{label}\n")
            
            logger.info(f"类别标签文件写入完成: {labels_file_path}")
            logger.debug(f"写入 {len(sorted_labels)} 个类别")
            
        except Exception as e:
            logger.error(f"写入类别标签文件失败: {e}")
    
    def extract_classes_only(self):
        """
        单独提取类别信息的公共方法
        可以在不进行数据集划分的情况下单独调用
        """
        logger.info("单独提取类别信息...")
        
        if not self.valid_pairs:
            logger.warning("没有有效的文件对，无法提取类别")
            return set()
        
        self._extract_classes()
        return self.classes.copy()
    
    def filter_classes_and_regenerate(self, exclude_classes: List[str], new_annotations_suffix: str = "filtered"):
        """
        过滤指定类别并重新生成数据集
        
        Args:
            exclude_classes: 要剔除的类别列表
            new_annotations_suffix: 新标注目录的后缀名
        
        Returns:
            dict: 过滤结果统计信息
        """
        logger.info(f"开始类别过滤 - 剔除类别: {exclude_classes}")
        logger.info(f"新标注目录后缀: {new_annotations_suffix}")
        
        # 创建新的标注目录
        new_annotations_dir = self.dataset_path / f"{ANNOTATIONS_DIR}_{new_annotations_suffix}"
        new_annotations_dir.mkdir(exist_ok=True)
        logger.info(f"创建新标注目录: {new_annotations_dir}")
        
        # 统计信息
        stats = {
            'total_files': len(self.valid_pairs),
            'filtered_files': 0,
            'empty_after_filter': 0,
            'valid_after_filter': 0,
            'excluded_classes': exclude_classes,
            'remaining_classes': set()
        }
        
        filtered_pairs = []
        
        # 处理每个XML文件
        for image_file, xml_file in self.valid_pairs:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 查找所有object标签
                objects = root.findall('object')
                remaining_objects = []
                
                # 过滤object
                for obj in objects:
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        class_name = name_elem.text.strip()
                        
                        # 如果不在排除列表中，保留该object
                        if class_name not in exclude_classes:
                            remaining_objects.append(obj)
                            stats['remaining_classes'].add(class_name)
                
                # 移除所有原有的object标签
                for obj in objects:
                    root.remove(obj)
                
                # 添加过滤后的object标签
                for obj in remaining_objects:
                    root.append(obj)
                
                # 如果还有有效的object，保存文件
                if remaining_objects:
                    new_xml_path = new_annotations_dir / xml_file.name
                    tree.write(new_xml_path, encoding=DEFAULT_ENCODING, xml_declaration=True)
                    filtered_pairs.append((image_file, new_xml_path))
                    stats['valid_after_filter'] += 1
                    logger.debug(f"过滤后保留文件: {xml_file.name} (剩余 {len(remaining_objects)} 个对象)")
                else:
                    stats['empty_after_filter'] += 1
                    logger.debug(f"过滤后为空，跳过文件: {xml_file.name}")
                
                stats['filtered_files'] += 1
                
            except Exception as e:
                logger.error(f"处理XML文件失败: {xml_file.name} - {e}")
        
        logger.info(f"类别过滤完成:")
        logger.info(f"  处理文件: {stats['filtered_files']} 个")
        logger.info(f"  有效文件: {stats['valid_after_filter']} 个")
        logger.info(f"  空文件: {stats['empty_after_filter']} 个")
        logger.info(f"  剩余类别: {sorted(stats['remaining_classes'])}")
        
        # 更新当前实例的数据
        self.valid_pairs = filtered_pairs
        self.classes = stats['remaining_classes']
        self.annotations_dir = new_annotations_dir
        
        # 重新生成labels.txt
        self._write_labels_file()
        
        # 重新划分数据集
        self._split_dataset()
        
        return stats
    
    def check_and_fix_image_dimensions(self, auto_fix: bool = False):
        """
        检查XML中记录的图像尺寸信息是否与实际图像匹配
        
        Args:
            auto_fix: 是否自动修正不匹配的信息
                     False: 只打印警告日志
                     True: 打印警告并修正XML数据，强制通道数为3
        
        Returns:
            dict: 检查结果统计信息
        """
        logger.info(f"开始检查图像尺寸信息 - 自动修正: {auto_fix}")
        
        stats = {
            'total_checked': 0,
            'dimension_mismatches': 0,
            'channel_mismatches': 0,
            'read_errors': 0,
            'fixed_xmls': 0,
            'converted_images': 0,
            'mismatch_details': []
        }
        
        # 创建进度条
        pbar = tqdm(self.valid_pairs, desc="检查图像尺寸", unit="文件")
        
        for image_file, xml_file in pbar:
            try:
                # 读取XML中的尺寸信息
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                size_elem = root.find('size')
                if size_elem is None:
                    logger.warning(f"XML文件缺少size标签: {xml_file.name}")
                    continue
                
                # 获取XML中记录的尺寸
                xml_width = size_elem.find('width')
                xml_height = size_elem.find('height')
                xml_depth = size_elem.find('depth')
                
                if xml_width is None or xml_height is None or xml_depth is None:
                    logger.warning(f"XML文件size标签不完整: {xml_file.name}")
                    continue
                
                xml_w = int(xml_width.text)
                xml_h = int(xml_height.text)
                xml_d = int(xml_depth.text)
                
                # 读取实际图像
                img = cv2.imread(str(image_file))
                if img is None:
                    logger.error(f"无法读取图像文件: {image_file.name}")
                    stats['read_errors'] += 1
                    continue
                
                # 获取实际图像尺寸
                actual_h, actual_w, actual_d = img.shape
                
                stats['total_checked'] += 1
                
                # 检查尺寸是否匹配
                dimension_mismatch = (xml_w != actual_w or xml_h != actual_h)
                channel_mismatch = (xml_d != actual_d or actual_d != 3)
                
                if dimension_mismatch or channel_mismatch:
                    mismatch_info = {
                        'file': xml_file.name,
                        'xml_size': (xml_w, xml_h, xml_d),
                        'actual_size': (actual_w, actual_h, actual_d),
                        'dimension_mismatch': dimension_mismatch,
                        'channel_mismatch': channel_mismatch
                    }
                    stats['mismatch_details'].append(mismatch_info)
                    
                    if dimension_mismatch:
                        stats['dimension_mismatches'] += 1
                        logger.warning(f"尺寸不匹配 {xml_file.name}: XML({xml_w}x{xml_h}) vs 实际({actual_w}x{actual_h})")
                    
                    if channel_mismatch:
                        stats['channel_mismatches'] += 1
                        logger.warning(f"通道数不匹配 {xml_file.name}: XML({xml_d}) vs 实际({actual_d})")
                    
                    # 如果启用自动修正
                    if auto_fix:
                        # 处理图像通道数
                        if actual_d != 3:
                            if actual_d == 1:
                                # 灰度图转RGB
                                img_fixed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            elif actual_d == 4:
                                # RGBA转RGB
                                img_fixed = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                            else:
                                logger.error(f"不支持的通道数: {actual_d} - {image_file.name}")
                                continue
                            
                            # 覆盖原图像
                            cv2.imwrite(str(image_file), img_fixed)
                            logger.info(f"已转换图像为3通道: {image_file.name}")
                            stats['converted_images'] += 1
                            
                            # 更新实际尺寸信息
                            actual_h, actual_w, actual_d = img_fixed.shape
                        
                        # 修正XML中的尺寸信息
                        xml_width.text = str(actual_w)
                        xml_height.text = str(actual_h)
                        xml_depth.text = "3"  # 强制设为3通道
                        
                        # 保存修正后的XML
                        tree.write(xml_file, encoding=DEFAULT_ENCODING, xml_declaration=True)
                        logger.info(f"已修正XML尺寸信息: {xml_file.name} -> ({actual_w}x{actual_h}x3)")
                        stats['fixed_xmls'] += 1
                
            except Exception as e:
                logger.error(f"处理图像文件失败: {image_file.name} - {e}")
                stats['read_errors'] += 1
        
        # 关闭进度条
        pbar.close()
        
        # 输出统计结果
        logger.info(f"图像尺寸检查完成:")
        logger.info(f"  检查文件: {stats['total_checked']} 个")
        logger.info(f"  尺寸不匹配: {stats['dimension_mismatches']} 个")
        logger.info(f"  通道数不匹配: {stats['channel_mismatches']} 个")
        logger.info(f"  读取错误: {stats['read_errors']} 个")
        
        if auto_fix:
            logger.info(f"  修正XML: {stats['fixed_xmls']} 个")
            logger.info(f"  转换图像: {stats['converted_images']} 个")
        
        return stats
    
    def _split_dataset(self):
        """数据集划分功能"""
        logger.info("开始数据集划分...")
        
        if not self.valid_pairs:
            logger.warning("没有有效的文件对，跳过数据集划分")
            return
        
        # 创建ImageSets/Main目录
        self.imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有有效文件的基础名称
        file_names = [pair[0].stem for pair in self.valid_pairs]
        
        # 设置随机种子确保可重复性
        random.seed(RANDOM_SEED)
        random.shuffle(file_names)
        
        total_count = len(file_names)
        train_count = int(total_count * self.train_ratio)
        val_count = int(total_count * self.val_ratio)
        
        # 划分数据集
        train_files = file_names[:train_count]
        val_files = file_names[train_count:train_count + val_count]
        test_files = file_names[train_count + val_count:]
        
        # 创建trainval集合（训练集+验证集）
        trainval_files = train_files + val_files
        
        # 写入文件
        self._write_split_file(TRAIN_TXT, train_files)
        self._write_split_file(VAL_TXT, val_files)
        self._write_split_file(TEST_TXT, test_files)
        self._write_split_file(TRAINVAL_TXT, trainval_files)
        
        logger.info(f"数据集划分完成:")
        logger.info(f"  训练集: {len(train_files)} 个文件")
        logger.info(f"  验证集: {len(val_files)} 个文件")
        logger.info(f"  测试集: {len(test_files)} 个文件")
        logger.info(f"  训练验证集: {len(trainval_files)} 个文件")
    
    def _write_split_file(self, filename: str, file_list: List[str]):
        """写入划分文件"""
        file_path = self.imagesets_dir / filename
        
        try:
            with open(file_path, 'w', encoding=DEFAULT_ENCODING) as f:
                for file_name in file_list:
                    # 查找对应的图像文件（支持多种格式）
                    image_file = None
                    for img_file, _ in self.valid_pairs:
                        if img_file.stem == file_name:
                            image_file = img_file
                            break
                    
                    if image_file:
                        # 写入格式: 图像路径\t标注路径
                        image_path = f"{JPEGS_DIR}/{image_file.name}"
                        annotation_path = f"{ANNOTATIONS_DIR}/{file_name}.xml"
                        f.write(f"{image_path}\t{annotation_path}\n")
            
            logger.debug(f"写入划分文件: {filename} ({len(file_list)} 个文件)")
            
        except Exception as e:
            logger.error(f"写入划分文件失败: {filename} - {e}")
    
    def get_classes(self) -> Set[str]:
        """获取所有类别"""
        return self.classes.copy()
    
    def get_dataset_info(self):
        """获取数据集基本信息"""
        return {
            'dataset_name': self.dataset_name,
            'dataset_path': str(self.dataset_path.absolute()),
            'annotations_dir': str(self.annotations_dir),
            'total_valid_pairs': len(self.valid_pairs),
            'total_classes': len(self.classes),
            'classes': sorted(list(self.classes)),
            'missing_xml_count': len(self.missing_xml_files),
            'missing_image_count': len(self.missing_image_files),
            'missing_xml_files': [f.name for f in self.missing_xml_files],
            'missing_image_files': [f.name for f in self.missing_image_files],
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            }
        }
    
    def print_summary(self):
        """打印数据集摘要信息"""
        logger.info("=== VOC数据集摘要 ===")
        logger.info(f"数据集名称: {self.dataset_name}")
        logger.info(f"数据集路径: {self.dataset_path}")
        logger.info(f"标注目录: {self.annotations_dir.name}")
        logger.info(f"有效文件对: {len(self.valid_pairs)} 个")
        logger.info(f"类别数量: {len(self.classes)} 个")
        logger.info(f"类别列表: {', '.join(sorted(self.classes))}")
        logger.info(f"缺少XML文件的图像: {len(self.missing_xml_files)} 个")
        logger.info(f"缺少图像文件的XML: {len(self.missing_image_files)} 个")
        logger.info(f"划分比例: 训练集{self.train_ratio}, 验证集{self.val_ratio}, 测试集{self.test_ratio}")
        
        if self.missing_xml_files:
            logger.info("缺少XML的图像文件:")
            for img_file in self.missing_xml_files[:5]:  # 只显示前5个
                logger.info(f"  {img_file.name}")
            if len(self.missing_xml_files) > 5:
                logger.info(f"  ... 还有 {len(self.missing_xml_files) - 5} 个")

    def convert_to_coco_format(self, output_dir: str = None) -> Dict[str, str]:
        """
        将VOC格式数据集转换为COCO格式
        
        Args:
            output_dir: 输出目录，默认为数据集根目录
            
        Returns:
            Dict[str, str]: 生成的COCO JSON文件路径
            
        Raises:
            FileNotFoundError: 当必需的文件不存在时
            ValueError: 当数据集未正确划分时
        """
        import json
        from xml.etree import ElementTree as ET
        
        logger.info("开始转换VOC格式到COCO格式")
        
        # 检查必需文件是否存在
        imagesets_main_dir = self.dataset_path / IMAGESETS_DIR / MAIN_DIR
        labels_file = imagesets_main_dir / "labels.txt"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"labels.txt文件不存在: {labels_file}")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.dataset_path
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取类别信息
        with open(labels_file, 'r', encoding=DEFAULT_ENCODING) as f:
            label_lines = f.readlines()
            categories = [{'id': i + 1, 'name': label.strip()} for i, label in enumerate(label_lines)]
        
        logger.info(f"加载了 {len(categories)} 个类别")
        
        # 只转换有内容的数据集划分
        result_files = {}
        
        for split in ['train', 'val']:
            list_file = imagesets_main_dir / f"{split}.txt"
            if not list_file.exists():
                logger.warning(f"{split}.txt 不存在，跳过 {split} 集转换")
                continue
            
            # 检查文件是否有内容
            with open(list_file, 'r', encoding=DEFAULT_ENCODING) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if not lines:
                logger.warning(f"{split}.txt 文件为空，跳过 {split} 集转换")
                continue
                
            output_json = output_dir / f"{split}_coco.json"
            self._convert_split_to_coco(list_file, categories, output_json)
            result_files[split] = str(output_json)
            logger.info(f"{split} 集转换完成: {output_json}")
        
        # 检查测试集是否存在且有内容
        test_list_file = imagesets_main_dir / "test.txt"
        if test_list_file.exists():
            with open(test_list_file, 'r', encoding=DEFAULT_ENCODING) as f:
                test_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if test_lines:
                output_json = output_dir / "test_coco.json"
                self._convert_split_to_coco(test_list_file, categories, output_json)
                result_files['test'] = str(output_json)
                logger.info(f"test 集转换完成: {output_json}")
            else:
                logger.info("test.txt 文件为空，跳过 test 集转换")
        
        logger.info("VOC到COCO格式转换完成")
        return result_files
    
    def _convert_split_to_coco(self, list_file: Path, categories: List[Dict], output_json: Path):
        """
        转换单个数据集划分到COCO格式
        
        Args:
            list_file: 数据集列表文件路径
            categories: 类别信息列表
            output_json: 输出JSON文件路径
        """
        import json
        from xml.etree import ElementTree as ET
        
        images = []
        annotations = []
        annotation_id = 1
        
        # 读取图像列表
        with open(list_file, 'r', encoding=DEFAULT_ENCODING) as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 解析图像路径和标注路径
            parts = line.split('\t')
            if len(parts) != 2:
                logger.warning(f"跳过格式错误的行: {line}")
                continue
                
            image_path, annotation_path = parts
            image_id = i + 1
            image_filename = Path(image_path).name
            
            # 从XML文件提取图像尺寸
            xml_path = self.dataset_path / annotation_path
            if not xml_path.exists():
                logger.warning(f"XML文件不存在: {xml_path}")
                continue
                
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                size_elem = root.find('size')
                if size_elem is None:
                    logger.warning(f"XML文件中没有size信息: {xml_path}")
                    continue
                    
                image_height = int(size_elem.find('height').text)
                image_width = int(size_elem.find('width').text)
                
                # 添加图像信息
                images.append({
                    'id': image_id,
                    'file_name': image_filename,
                    'height': image_height,
                    'width': image_width,
                    'license': None,
                    'flickr_url': None,
                    'coco_url': None,
                    'date_captured': None
                })
                
                # 解析标注信息
                objects = root.findall('object')
                for obj in objects:
                    # 获取类别名称
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    label = name_elem.text
                    
                    # 查找类别ID
                    category_id = None
                    for cat in categories:
                        if cat['name'] == label:
                            category_id = cat['id']
                            break
                    
                    if category_id is None:
                        logger.warning(f"未知类别 '{label}' 在文件 {xml_path}")
                        continue
                    
                    # 获取边界框
                    bbox_elem = obj.find('bndbox')
                    if bbox_elem is None:
                        continue
                        
                    xmin = int(bbox_elem.find('xmin').text)
                    ymin = int(bbox_elem.find('ymin').text)
                    xmax = int(bbox_elem.find('xmax').text)
                    ymax = int(bbox_elem.find('ymax').text)
                    
                    # 计算COCO格式的边界框 [x, y, width, height]
                    width = xmax - xmin
                    height = ymax - ymin
                    area = width * height
                    
                    # 添加标注信息
                    annotations.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [xmin, ymin, width, height],
                        'area': area,
                        'segmentation': [],
                        'iscrowd': 0
                    })
                    annotation_id += 1
                    
            except Exception as e:
                logger.error(f"处理XML文件时出错 {xml_path}: {e}")
                continue
        
        # 构建COCO格式数据
        coco_data = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        
        # 保存JSON文件
        with open(output_json, 'w', encoding=DEFAULT_ENCODING) as f:
            json.dump(coco_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"转换完成: {len(images)} 张图像, {len(annotations)} 个标注")


if __name__ == "__main__":
    # 测试VOC数据集类
    dataset_path = "../../dataset/Fruit"
    
    try:
        voc_dataset = VOCDataset(dataset_path)
        voc_dataset.print_summary()
        
        # 测试图像尺寸检查功能
        print(f"\n=== 测试图像尺寸检查功能 ===")
        check_stats = voc_dataset.check_and_fix_image_dimensions(auto_fix=False)
        print(f"检查统计: {check_stats}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误: {e}")