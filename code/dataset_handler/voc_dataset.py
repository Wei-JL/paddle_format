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
- 多线程并行处理
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
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from xml.dom import minidom

# 导入日志系统 - 使用全局变量
sys.path.append(str(Path(__file__).parent.parent))
from logger_code.logger_sys import get_logger
from global_var.global_cls import *

# 获取当前文件名作为日志标识
current_filename = Path(__file__).stem
logger = get_logger(current_filename)

class VOCDataset:
    """VOC数据集处理类"""
    
    def __init__(self, dataset_path: str, user_labels_file: str = None, 
                 train_ratio: float = TRAIN_RATIO, 
                 val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO,
                 max_workers: int = 4, annotations_folder_name: str = ANNOTATIONS_DIR):
        """
        初始化VOC数据集
        
        Args:
            dataset_path: 数据集根目录路径
            user_labels_file: 用户提供的正确标签文件路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            max_workers: 线程池最大工作线程数
            annotations_folder_name: 标注文件夹名称，默认为"Annotations"
        """
        self.dataset_path = Path(dataset_path)
        # 自动获取数据集名称（文件夹最后一个名称）
        self.dataset_name = self.dataset_path.name
        
        # 标注文件夹名称（可自定义）
        self.annotations_folder_name = annotations_folder_name
        
        # 数据集划分比例
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 线程池配置
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()  # 线程安全锁
        
        # 标准VOC目录结构 - 使用os.path.join拼接路径
        self.annotations_dir = Path(os.path.join(str(self.dataset_path), self.annotations_folder_name))
        self.images_dir = Path(os.path.join(str(self.dataset_path), JPEGS_DIR))
        self.imagesets_dir = Path(os.path.join(str(self.dataset_path), IMAGESETS_DIR, MAIN_DIR))
        
        # 清洗后的输出目录 - 使用os.path.join拼接路径
        self.annotations_output_dir = Path(os.path.join(str(self.dataset_path), "Annotations_clear"))
        
        # 用户标签文件
        self.user_labels_file = user_labels_file
        self.user_labels = set()  # 用户提供的正确标签集合
        
        # 记录缺失文件的列表
        self.missing_xml_files = []  # 缺少XML文件的图像
        self.missing_image_files = []  # 缺少图像文件的XML
        self.images_without_xml = []  # 有图像但没有XML的文件列表
        
        # 有效的文件对列表
        self.valid_pairs = []
        
        # 类别集合
        self.classes = set()
        
        # 尺寸不匹配记录
        self.dimension_mismatches = []
        self.channel_mismatches = []
        
        logger.info(f"初始化VOC数据集: {self.dataset_name}")
        logger.info(f"数据集路径: {self.dataset_path.absolute()}")
        logger.info(f"划分比例 - 训练集: {self.train_ratio}, 验证集: {self.val_ratio}, 测试集: {self.test_ratio}")
        logger.info(f"线程池配置 - 最大工作线程: {self.max_workers}")
        
        # 验证用户标签文件
        if self.user_labels_file:
            self._validate_user_labels()
        
        print(f"✅ VOC数据集初始化完成: {self.dataset_name}")
        print(f"📁 数据集路径: {self.dataset_path}")
        print(f"🧵 线程池配置: {self.max_workers} 个工作线程")
        print("💡 请调用 one_click_complete_conversion() 方法开始数据处理")
    
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
    
    def _validate_user_labels(self):
        """验证用户提供的标签文件"""
        logger.info("验证用户标签文件...")
        
        if not self.user_labels_file:
            logger.warning("未提供用户标签文件")
            return
        
        labels_path = Path(self.user_labels_file)
        if not labels_path.exists():
            logger.error(f"用户标签文件不存在: {labels_path}")
            raise FileNotFoundError(f"用户标签文件不存在: {labels_path}")
        
        try:
            with open(labels_path, 'r', encoding=DEFAULT_ENCODING) as f:
                for line in f:
                    label = line.strip()
                    if label:
                        self.user_labels.add(label)
            
            logger.info(f"加载用户标签: {len(self.user_labels)} 个")
            logger.info(f"标签列表: {sorted(self.user_labels)}")
            
        except Exception as e:
            logger.error(f"读取用户标签文件失败: {e}")
            raise
    
    def _validate_basic_structure(self):
        """验证数据集基本结构"""
        logger.info("验证数据集基本结构...")
        
        # 检查必需的目录
        required_dirs = [
            (self.annotations_dir, "标注目录"),
            (self.images_dir, "图像目录")
        ]
        
        for dir_path, dir_name in required_dirs:
            if not dir_path.exists():
                error_msg = f"{dir_name}不存在: {dir_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.debug(f"目录检查通过: {dir_path}")
        
        # 创建ImageSets目录（如果不存在）
        self.imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建清洗后的输出目录
        self.annotations_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建清洗输出目录: {self.annotations_output_dir}")
        
        logger.info("数据集基本结构验证通过")
    
    def _match_files_parallel(self):
        """并行匹配图像和标注文件"""
        logger.info("开始并行匹配图像和标注文件...")
        
        print("🔍 正在扫描文件...")
        
        # 获取所有图像文件
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))
            image_files.extend(list(self.images_dir.glob(f"*{ext.upper()}")))
        
        # 获取所有XML文件
        xml_files = list(self.annotations_dir.glob(f"*{XML_EXTENSION}"))
        
        print(f"📊 发现图像文件: {len(image_files)} 个")
        print(f"📊 发现XML文件: {len(xml_files)} 个")
        
        # 记录详细的文件统计信息
        logger.info(f"📊 文件扫描结果:")
        logger.info(f"   图像文件总数: {len(image_files)} 个")
        logger.info(f"   XML标注文件: {len(xml_files)} 个")
        
        # 统计图像文件类型分布
        image_type_count = {}
        for img in image_files:
            ext = img.suffix.lower()
            image_type_count[ext] = image_type_count.get(ext, 0) + 1
        
        logger.info(f"📈 图像文件类型分布:")
        for ext, count in sorted(image_type_count.items()):
            logger.info(f"   {ext}: {count} 个")
        
        # 创建文件名映射（不含扩展名）
        print("🔗 创建文件名映射...")
        image_stems = {f.stem: f for f in image_files}
        xml_stems = {f.stem: f for f in xml_files}
        
        # 检查每个XML文件对应的图像是否存在
        print("✅ 验证XML文件对应的图像...")
        missing_count = 0
        for stem, xml_file in tqdm(xml_stems.items(), desc="检查XML->图像匹配", unit="文件"):
            if stem not in image_stems:
                error_msg = f"XML文件没有对应的图像文件: {xml_file.absolute()} -> 缺少图像: {stem}"
                logger.error(error_msg)
                self.missing_image_files.append(xml_file)
                missing_count += 1
        
        # 如果有XML没有对应图片，报错并退出
        if self.missing_image_files:
            error_msg = f"发现 {len(self.missing_image_files)} 个XML文件没有对应的图像文件，请检查数据集完整性"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            for xml_file in self.missing_image_files[:5]:
                logger.error(f"  - {xml_file.absolute()}")
            if len(self.missing_image_files) > 5:
                logger.error(f"  ... 还有 {len(self.missing_image_files) - 5} 个文件")
            raise FileNotFoundError(error_msg)
        
        # 检查有图像但没有XML的文件（记录警告，不报错）
        print("⚠️  检查缺少XML标注的图像...")
        no_xml_count = 0
        for stem, image_file in tqdm(image_stems.items(), desc="检查图像->XML匹配", unit="文件"):
            if stem not in xml_stems:
                self.images_without_xml.append(image_file)
                no_xml_count += 1
                if no_xml_count <= 10:
                    logger.warning(f"图像文件缺少对应的XML标注: {image_file.name}")
        
        if no_xml_count > 10:
            logger.warning(f"还有 {no_xml_count - 10} 个图像文件缺少XML标注（已省略日志）")
        
        # 并行验证有效的文件对
        print("📝 使用线程池并行验证文件对...")
        valid_stems = set(image_stems.keys()) & set(xml_stems.keys())
        
        logger.info(f"🧵 启动线程池进行并行处理，工作线程数: {self.max_workers}")
        logger.info(f"📊 需要验证的文件对: {len(valid_stems)} 个")
        
        # 准备并行处理的任务
        tasks = []
        for stem in valid_stems:
            image_file = image_stems[stem]
            xml_file = xml_stems[stem]
            tasks.append((image_file, xml_file))
        
        # 使用线程池并行处理文件对验证
        valid_pairs = []
        processed_count = 0
        
        with tqdm(total=len(tasks), desc="🧵 并行验证文件对", unit="对") as pbar:
            # 提交所有任务到线程池
            futures = []
            for image_file, xml_file in tasks:
                future = self.thread_pool.submit(self._validate_file_pair_with_check, image_file, xml_file)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        # 线程安全地添加到结果列表
                        with self._lock:
                            valid_pairs.append(result)
                            processed_count += 1
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"🚫 验证文件对时出错: {e}")
                    pbar.update(1)
        
        self.valid_pairs = valid_pairs
        
        logger.info(f"🧵 线程池并行处理完成:")
        logger.info(f"   处理任务数: {processed_count}/{len(tasks)}")
        logger.info(f"   有效文件对: {len(valid_pairs)} 个")
        
        logger.info(f"文件匹配完成:")
        logger.info(f"  有效文件对: {len(self.valid_pairs)} 个")
        logger.info(f"  缺少XML的图像: {len(self.images_without_xml)} 个")
        logger.info(f"  缺少图像的XML: {len(self.missing_image_files)} 个")
        
        print(f"✅ 并行文件匹配完成: {len(self.valid_pairs)} 对有效文件")
        if self.images_without_xml:
            print(f"⚠️  发现 {len(self.images_without_xml)} 个图像缺少XML标注（已跳过）")
    
    def _validate_file_pair_with_check(self, image_file: Path, xml_file: Path) -> Tuple[Path, Path]:
        """使用线程池验证单个文件对的有效性，包括更详细的检查"""
        try:
            # 检查文件是否存在
            if not image_file.exists():
                logger.error(f"图像文件不存在: {image_file.absolute()}")
                return None
            
            if not xml_file.exists():
                logger.error(f"XML文件不存在: {xml_file.absolute()}")
                return None
            
            # 检查文件大小（避免空文件）
            if image_file.stat().st_size == 0:
                logger.warning(f"图像文件为空: {image_file.name}")
                return None
                
            if xml_file.stat().st_size == 0:
                logger.warning(f"XML文件为空: {xml_file.name}")
                return None
            
            # 简单验证XML文件格式
            try:
                import xml.etree.ElementTree as ET
                ET.parse(xml_file)
            except ET.ParseError as e:
                logger.error(f"XML文件格式错误: {xml_file.name} - {e}")
                return None
            
            return (image_file, xml_file)
            
        except Exception as e:
            logger.error(f"验证文件对失败: {image_file.name}, {xml_file.name} - {e}")
            return None
    
    def _validate_file_pair(self, image_file: Path, xml_file: Path) -> Tuple[Path, Path]:
        """验证单个文件对的有效性（保留原方法兼容性）"""
        return self._validate_file_pair_with_check(image_file, xml_file)
    
    def _remove_empty_annotations_and_clean(self):
        """删除空标注文件并清洗XML到输出目录"""
        logger.info("开始检查空标注并清洗XML文件到输出目录...")
        
        print("🧹 正在清洗XML文件...")
        
        # 清空输出目录
        if self.annotations_output_dir.exists():
            shutil.rmtree(self.annotations_output_dir)
        self.annotations_output_dir.mkdir(parents=True, exist_ok=True)
        
        valid_pairs_after_cleanup = []
        empty_annotations = []
        
        # 并行处理XML文件清洗
        with tqdm(total=len(self.valid_pairs), desc="清洗XML文件", unit="文件") as pbar:
            futures = []
            for image_file, xml_file in self.valid_pairs:
                future = self.thread_pool.submit(self._process_xml_file, image_file, xml_file)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        if result['is_valid']:
                            valid_pairs_after_cleanup.append((result['image_file'], result['output_xml_file']))
                        else:
                            empty_annotations.append((result['image_file'], result['xml_file']))
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"处理XML文件时出错: {e}")
                    pbar.update(1)
        
        # 更新有效文件对，现在指向清洗后的XML文件
        self.valid_pairs = valid_pairs_after_cleanup
        
        # 更新annotations_dir指向清洗后的目录
        self.annotations_dir = self.annotations_output_dir
        
        logger.info(f"XML清洗完成:")
        logger.info(f"  发现空标注: {len(empty_annotations)} 个")
        logger.info(f"  清洗后有效文件: {len(self.valid_pairs)} 个")
        logger.info(f"  清洗输出目录: {self.annotations_output_dir}")
        
        print(f"🧹 XML清洗完成:")
        print(f"   发现空标注: {len(empty_annotations)} 个")
        print(f"   清洗后有效文件: {len(self.valid_pairs)} 个")
        print(f"   输出目录: {self.annotations_output_dir}")
        
        if empty_annotations:
            logger.warning("以下文件为空标注（已跳过）:")
            for i, (img_file, xml_file) in enumerate(empty_annotations[:10]):
                logger.warning(f"  {i+1}. {xml_file.name}")
            if len(empty_annotations) > 10:
                logger.warning(f"  ... 还有 {len(empty_annotations) - 10} 个空标注文件")
    
    def _process_xml_file(self, image_file: Path, xml_file: Path) -> Dict:
        """处理单个XML文件，保持原有格式"""
        try:
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 检查是否有object标签
            objects = root.findall('object')
            
            if not objects:
                # 空标注文件，不复制到输出目录
                logger.warning(f"发现空标注文件: {xml_file.name}")
                return {
                    'is_valid': False,
                    'image_file': image_file,
                    'xml_file': xml_file,
                    'output_xml_file': None
                }
            else:
                # 有效标注文件，复制到输出目录并保持格式
                output_xml_file = Path(os.path.join(str(self.annotations_output_dir), xml_file.name))
                
                # 使用minidom保持原有格式
                self._copy_xml_with_format(xml_file, output_xml_file)
                
                logger.debug(f"清洗XML文件: {xml_file.name} -> {output_xml_file.name}")
                return {
                    'is_valid': True,
                    'image_file': image_file,
                    'xml_file': xml_file,
                    'output_xml_file': output_xml_file
                }
                
        except Exception as e:
            logger.error(f"处理XML文件失败: {xml_file.name} - {e}")
            return None
    
    def _copy_xml_with_format(self, source_xml: Path, target_xml: Path):
        """复制XML文件并保持原有格式（换行和缩进）"""
        try:
            # 直接复制文件以保持原有格式
            shutil.copy2(source_xml, target_xml)
        except Exception as e:
            logger.error(f"复制XML文件失败: {source_xml} -> {target_xml} - {e}")
            raise
    
    def _extract_classes(self):
        """提取所有类别"""
        logger.info("开始提取类别信息...")
        
        self.classes = set()
        class_count = {}
        
        for image_file, xml_file in tqdm(self.valid_pairs, desc="提取类别", unit="文件"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 查找所有object标签
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        class_name = name_elem.text.strip()
                        self.classes.add(class_name)
                        class_count[class_name] = class_count.get(class_name, 0) + 1
                        
            except Exception as e:
                logger.error(f"提取类别时解析XML失败: {xml_file.name} - {e}")
        
        logger.info(f"类别提取完成:")
        logger.info(f"  发现类别: {len(self.classes)} 个")
        logger.info(f"  类别列表: {sorted(self.classes)}")
        
        # 记录每个类别的数量统计
        logger.info("📊 类别数量统计:")
        for class_name in sorted(class_count.keys()):
            logger.info(f"   {class_name}: {class_count[class_name]} 个对象")
        
        print(f"🏷️  类别提取完成: 发现 {len(self.classes)} 个类别")
    
    def _write_labels_file(self):
        """写入标签文件"""
        logger.info("写入标签文件...")
        
        if not self.classes:
            logger.warning("没有类别信息，跳过标签文件写入")
            return
        
        labels_file = Path(os.path.join(str(self.imagesets_dir), LABELS_TXT))
        
        try:
            with open(labels_file, 'w', encoding=DEFAULT_ENCODING) as f:
                for class_name in sorted(self.classes):
                    f.write(f"{class_name}\n")
            
            logger.info(f"标签文件写入完成: {labels_file}")
            logger.info(f"写入类别: {len(self.classes)} 个")
            
        except Exception as e:
            logger.error(f"写入标签文件失败: {e}")
            raise
    
    def check_and_fix_image_dimensions_parallel(self, auto_fix: bool = False):
        """
        并行检查XML中记录的图像尺寸信息是否与实际图像匹配
        
        Args:
            auto_fix: 是否自动修正不匹配的信息
        
        Returns:
            dict: 检查结果统计信息
        """
        logger.info(f"开始并行检查图像尺寸信息 - 自动修正: {auto_fix}")
        
        stats = {
            'total_checked': 0,
            'dimension_mismatches': 0,
            'channel_mismatches': 0,
            'read_errors': 0,
            'fixed_xmls': 0,
            'converted_images': 0,
            'mismatch_details': []
        }
        
        # 清空之前的记录
        self.dimension_mismatches = []
        self.channel_mismatches = []
        
        # 并行处理图像尺寸检查
        print("📐 并行检查图像尺寸...")
        with tqdm(total=len(self.valid_pairs), desc="检查图像尺寸", unit="文件") as pbar:
            futures = []
            for image_file, xml_file in self.valid_pairs:
                future = self.thread_pool.submit(self._check_single_image_dimension, image_file, xml_file, auto_fix)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        # 线程安全地更新统计信息
                        with self._lock:
                            stats['total_checked'] += 1
                            if result.get('dimension_mismatch'):
                                stats['dimension_mismatches'] += 1
                                self.dimension_mismatches.append(result)
                            if result.get('channel_mismatch'):
                                stats['channel_mismatches'] += 1
                                self.channel_mismatches.append(result)
                            if result.get('read_error'):
                                stats['read_errors'] += 1
                            if result.get('fixed_xml'):
                                stats['fixed_xmls'] += 1
                            if result.get('converted_image'):
                                stats['converted_images'] += 1
                            if result.get('mismatch_details'):
                                stats['mismatch_details'].append(result['mismatch_details'])
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"检查图像尺寸时出错: {e}")
                    pbar.update(1)
        
        # 输出统计结果
        logger.info(f"📊 并行图像尺寸检查完成:")
        logger.info(f"  检查文件: {stats['total_checked']} 个")
        logger.info(f"  尺寸不匹配: {stats['dimension_mismatches']} 个")
        logger.info(f"  通道数不匹配: {stats['channel_mismatches']} 个")
        logger.info(f"  读取错误: {stats['read_errors']} 个")
        
        if auto_fix:
            logger.info(f"  修正XML: {stats['fixed_xmls']} 个")
            logger.info(f"  转换图像: {stats['converted_images']} 个")
        
        print(f"📐 并行图像尺寸检查完成:")
        print(f"   检查文件: {stats['total_checked']} 个")
        print(f"   尺寸不匹配: {stats['dimension_mismatches']} 个")
        print(f"   通道数不匹配: {stats['channel_mismatches']} 个")
        if auto_fix:
            print(f"   修正XML: {stats['fixed_xmls']} 个")
            print(f"   转换图像: {stats['converted_images']} 个")
        
        return stats
    
    def _check_single_image_dimension(self, image_file: Path, xml_file: Path, auto_fix: bool = False) -> Dict:
        """检查单个图像的尺寸信息"""
        try:
            # 读取XML中的尺寸信息
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size_elem = root.find('size')
            if size_elem is None:
                logger.warning(f"XML文件缺少size标签: {xml_file.name}")
                return None
            
            # 获取XML中记录的尺寸
            xml_width = size_elem.find('width')
            xml_height = size_elem.find('height')
            xml_depth = size_elem.find('depth')
            
            if xml_width is None or xml_height is None or xml_depth is None:
                logger.warning(f"XML文件size标签不完整: {xml_file.name}")
                return None
            
            xml_w = int(xml_width.text)
            xml_h = int(xml_height.text)
            xml_d = int(xml_depth.text)
            
            # 读取实际图像
            img = cv2.imread(str(image_file))
            if img is None:
                logger.error(f"无法读取图像文件: {image_file.name}")
                return {'read_error': True}
            
            # 获取实际图像尺寸
            actual_h, actual_w, actual_d = img.shape
            
            # 检查尺寸是否匹配
            dimension_mismatch = (xml_w != actual_w or xml_h != actual_h)
            channel_mismatch = (xml_d != actual_d or actual_d != 3)
            
            result = {
                'xml_file': str(xml_file.absolute()),
                'image_file': str(image_file.absolute()),
                'xml_size': (xml_w, xml_h, xml_d),
                'actual_size': (actual_w, actual_h, actual_d),
                'dimension_mismatch': dimension_mismatch,
                'channel_mismatch': channel_mismatch,
                'fixed_xml': False,
                'converted_image': False
            }
            
            if dimension_mismatch or channel_mismatch:
                if dimension_mismatch:
                    warning_msg = f"📐 尺寸不匹配 - {xml_file.name}: XML({xml_w}x{xml_h}) vs 实际({actual_w}x{actual_h})"
                    logger.warning(warning_msg)
                
                if channel_mismatch:
                    warning_msg = f"🎨 通道数不匹配 - {xml_file.name}: XML({xml_d}) vs 实际({actual_d})"
                    logger.warning(warning_msg)
                
                # 如果启用自动修正
                if auto_fix:
                    # 处理图像通道数
                    img_modified = False
                    if actual_d != 3:
                        if actual_d == 1:
                            # 灰度图转RGB
                            img_fixed = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            img_modified = True
                            logger.info(f"🔧 转换灰度图为RGB: {image_file.name}")
                        elif actual_d == 4:
                            # RGBA转RGB
                            img_fixed = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                            img_modified = True
                            logger.info(f"🔧 转换RGBA为RGB: {image_file.name}")
                        else:
                            logger.error(f"不支持的通道数: {actual_d} - {image_file.name}")
                            return result
                        
                        if img_modified:
                            # 覆盖原图像
                            cv2.imwrite(str(image_file), img_fixed)
                            logger.info(f"✅ 已转换图像为3通道并覆盖原图: {image_file.name}")
                            result['converted_image'] = True
                            
                            # 更新实际尺寸信息
                            actual_h, actual_w, actual_d = img_fixed.shape
                    
                    # 修正XML中的尺寸信息
                    xml_width.text = str(actual_w)
                    xml_height.text = str(actual_h)
                    xml_depth.text = "3"  # 强制设为3通道
                    
                    # 保存修正后的XML
                    tree.write(xml_file, encoding=DEFAULT_ENCODING, xml_declaration=True)
                    logger.info(f"✅ 已修正XML尺寸信息: {xml_file.name} -> ({actual_w}x{actual_h}x3)")
                    result['fixed_xml'] = True
                
                result['mismatch_details'] = {
                    'xml_file': str(xml_file.absolute()),
                    'image_file': str(image_file.absolute()),
                    'xml_size': (xml_w, xml_h, xml_d),
                    'actual_size': (actual_w, actual_h, actual_d),
                    'dimension_mismatch': dimension_mismatch,
                    'channel_mismatch': channel_mismatch
                }
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像文件失败: {image_file.name} - {e}")
            return {'read_error': True}
    
    def _split_dataset(self):
        """数据集划分功能"""
        logger.info("开始数据集划分...")
        print("📊 正在划分数据集...")
        
        if not self.valid_pairs:
            logger.warning("没有有效的文件对，跳过数据集划分")
            return {"success": False, "message": "没有有效的文件对"}
        
        # 创建ImageSets/Main目录
        self.imagesets_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建文件名到完整路径的映射
        print("🗂️  创建文件映射...")
        stem_to_pair = {pair[0].stem: pair for pair in self.valid_pairs}
        file_names = list(stem_to_pair.keys())
        
        # 设置随机种子确保可重复性
        random.seed(RANDOM_SEED)
        random.shuffle(file_names)
        
        total_count = len(file_names)
        train_count = int(total_count * self.train_ratio)
        val_count = int(total_count * self.val_ratio)
        
        print(f"📈 划分统计: 总计 {total_count} 个文件")
        print(f"   - 训练集: {train_count} 个 ({self.train_ratio*100:.1f}%)")
        print(f"   - 验证集: {val_count} 个 ({self.val_ratio*100:.1f}%)")
        print(f"   - 测试集: {total_count - train_count - val_count} 个 ({self.test_ratio*100:.1f}%)")
        
        # 划分数据集
        train_files = file_names[:train_count]
        val_files = file_names[train_count:train_count + val_count]
        test_files = file_names[train_count + val_count:]
        
        # 创建trainval集合（训练集+验证集）
        trainval_files = train_files + val_files
        
        # 写入文件
        print("💾 写入划分文件...")
        self._write_split_file_optimized(TRAIN_TXT, train_files, stem_to_pair)
        self._write_split_file_optimized(VAL_TXT, val_files, stem_to_pair)
        if test_files:
            self._write_split_file_optimized(TEST_TXT, test_files, stem_to_pair)
        self._write_split_file_optimized(TRAINVAL_TXT, trainval_files, stem_to_pair)
        
        logger.info(f"📊 数据集划分完成:")
        logger.info(f"  训练集: {len(train_files)} 个文件")
        logger.info(f"  验证集: {len(val_files)} 个文件")
        logger.info(f"  测试集: {len(test_files)} 个文件")
        logger.info(f"  训练验证集: {len(trainval_files)} 个文件")
        
        print("✅ 数据集划分完成!")
        
        return {
            "success": True,
            "message": "数据集划分完成",
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files)
        }
    
    def _write_split_file_optimized(self, filename: str, file_list: List[str], stem_to_pair: dict):
        """写入划分文件 - 优化版本"""
        file_path = Path(os.path.join(str(self.imagesets_dir), filename))
        
        try:
            with open(file_path, 'w', encoding=DEFAULT_ENCODING) as f:
                for file_name in tqdm(file_list, desc=f"写入{filename}", unit="文件"):
                    if file_name in stem_to_pair:
                        img_file, xml_file = stem_to_pair[file_name]
                        # 写入格式: 图像路径\t标注路径
                        image_path = os.path.join(JPEGS_DIR, img_file.name)
                        # 注意：现在XML文件在清洗后的目录中
                        annotation_path = os.path.join(ANNOTATIONS_OUTPUT_DIR, xml_file.name)
                        line_content = f"{image_path}\t{annotation_path}\n"
                        f.write(line_content)
            
            logger.debug(f"写入划分文件: {filename} ({len(file_list)} 个文件)")
            
        except Exception as e:
            logger.error(f"写入划分文件失败: {filename} - {e}")
    
    def _convert_to_coco(self):
        """转换为COCO格式"""
        logger.info("开始转换为COCO格式...")
        print("🔄 正在转换为COCO格式...")
        
        try:
            # 检查必需文件
            labels_file = Path(os.path.join(str(self.imagesets_dir), LABELS_TXT))
            
            if not labels_file.exists():
                error_msg = f"{LABELS_TXT}文件不存在: {labels_file}"
                logger.error(error_msg)
                return {"success": False, "message": error_msg}
            
            # 读取类别信息
            with open(labels_file, 'r', encoding=DEFAULT_ENCODING) as f:
                label_lines = f.readlines()
                categories = [{'id': i + 1, 'name': label.strip()} for i, label in enumerate(label_lines)]
            
            logger.info(f"📋 加载了 {len(categories)} 个类别")
            
            # 只转换train和val集
            result_files = {}
            for split in ['train', 'val']:
                list_file = Path(os.path.join(str(self.imagesets_dir), f"{split}.txt"))
                if not list_file.exists():
                    logger.warning(f"{split}.txt 不存在，跳过 {split} 集转换")
                    continue
                
                # 检查文件是否有内容
                with open(list_file, 'r', encoding=DEFAULT_ENCODING) as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                if not lines:
                    logger.warning(f"{split}.txt 文件为空，跳过 {split} 集转换")
                    continue
                
                output_json = Path(os.path.join(str(self.dataset_path), f"{split}_coco.json"))
                self._convert_split_to_coco_optimized(list_file, categories, output_json)
                result_files[split] = str(output_json)
                logger.info(f"✅ {split} 集转换完成: {output_json}")
            
            print("✅ COCO格式转换完成!")
            return {"success": True, "message": "COCO格式转换完成", "files": result_files}
            
        except Exception as e:
            error_msg = f"COCO转换失败: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def _convert_split_to_coco_optimized(self, list_file: Path, categories: List[Dict], output_json: Path):
        """转换单个数据集划分到COCO格式"""
        import json
        from xml.etree import ElementTree as ET
        
        images = []
        annotations = []
        annotation_id = 1
        
        # 创建类别名称到ID的映射
        category_name_to_id = {cat['name']: cat['id'] for cat in categories}
        
        # 读取图像列表
        with open(list_file, 'r', encoding=DEFAULT_ENCODING) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"📝 处理 {len(lines)} 个文件...")
        
        for i, line in enumerate(tqdm(lines, desc=f"转换{list_file.stem}", unit="文件")):
            # 解析图像路径和标注路径
            parts = line.split('\t')
            if len(parts) != 2:
                logger.warning(f"跳过格式错误的行: {line}")
                continue
                
            image_path, annotation_path = parts
            image_id = i + 1
            image_filename = Path(image_path).name
            
            # 从XML文件提取图像尺寸
            xml_path = Path(os.path.join(str(self.dataset_path), annotation_path))
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
                    
                    # 使用映射快速查找类别ID
                    category_id = category_name_to_id.get(label)
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
                        'iscrowd': 0,
                        'segmentation': []
                    })
                    annotation_id += 1
                    
            except ET.ParseError as e:
                logger.error(f"XML解析错误 {xml_path}: {e}")
                continue
            except Exception as e:
                logger.error(f"处理文件时出错 {xml_path}: {e}")
                continue
        
        # 构建COCO格式数据
        coco_data = {
            'info': {
                'description': f'{self.dataset_name} Dataset',
                'url': '',
                'version': '1.0',
                'year': 2024,
                'contributor': 'VOC to COCO Converter',
                'date_created': '2024-01-01'
            },
            'licenses': [],
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        
        # 写入JSON文件
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"COCO格式转换完成: {len(images)} 张图像, {len(annotations)} 个标注")

    def one_click_complete_conversion(self, skip_confirmation=False):
        """一键完成转换并修复所有问题"""
        try:
            logger.info("=" * 80)
            logger.info("🚀 开始一键完成转换处理（多线程版本）")
            logger.info("=" * 80)
            logger.info(f"数据集名称: {self.dataset_name}")
            logger.info(f"数据集路径: {self.dataset_path}")
            logger.info(f"线程池配置: {self.max_workers} 个工作线程")
            logger.info(f"处理时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if not skip_confirmation:
                print("\n⚠️  注意：请确保已备份原始数据集！")
                print("🚀 开始一键完成转换...")
                print("⚠️  注意：此操作将修改您的数据集文件！")
                
                user_input = input("\n请确认您已备份数据集 (输入 Y 继续，N 取消): ")
                if user_input.lower() != 'y':
                    logger.info("❌ 用户取消操作")
                    print("❌ 操作已取消")
                    return {"success": False, "message": "用户取消操作"}
                
                logger.info("✅ 用户确认继续处理")
            
            print("\n📋 步骤1: 数据验证和清洗...")
            logger.info("📋 步骤1: 开始数据验证和清洗")
            
            # 验证基本结构
            logger.info("🔍 验证数据集基本结构...")
            self._validate_basic_structure()
            logger.info("✅ 数据集结构验证完成")
            
            # 并行匹配文件
            logger.info("🔗 开始并行匹配图像和标注文件...")
            self._match_files_parallel()
            logger.info(f"✅ 并行文件匹配完成 - 有效文件对: {len(self.valid_pairs)} 个")
            
            # 删除空标注并清洗XML
            logger.info("🧹 开始清理空标注文件并清洗XML...")
            empty_count_before = len(self.valid_pairs)
            self._remove_empty_annotations_and_clean()
            empty_count_after = len(self.valid_pairs)
            removed_empty = empty_count_before - empty_count_after
            logger.info(f"✅ XML清洗完成 - 删除空标注: {removed_empty} 个")
            
            # 提取类别
            logger.info("🏷️  开始提取类别信息...")
            self._extract_classes()
            logger.info(f"✅ 类别提取完成 - 发现类别: {len(self.classes)} 个")
            logger.info(f"📝 类别列表: {sorted(list(self.classes))}")
            
            # 写入标签文件
            logger.info("💾 写入标签文件...")
            self._write_labels_file()
            logger.info("✅ 标签文件写入完成")
            
            print("\n🔧 步骤2: 并行图像尺寸检查和修正...")
            logger.info("📋 步骤2: 开始并行图像尺寸检查和修正")
            
            # 并行检查并修正图像尺寸
            logger.info("📐 开始并行检查图像尺寸一致性...")
            dimension_stats = self.check_and_fix_image_dimensions_parallel(auto_fix=True)
            logger.info("✅ 并行图像尺寸检查和修正完成")
            logger.info(f"📊 处理统计:")
            logger.info(f"   检查文件: {dimension_stats.get('total_checked', 0)} 个")
            logger.info(f"   尺寸不匹配: {dimension_stats.get('dimension_mismatches', 0)} 个")
            logger.info(f"   通道数修正: {dimension_stats.get('converted_images', 0)} 个")
            logger.info(f"   XML修正: {dimension_stats.get('fixed_xmls', 0)} 个")
            
            print("\n📊 步骤3: 数据集划分...")
            logger.info("📋 步骤3: 开始数据集划分")
            
            # 划分数据集
            split_result = self._split_dataset()
            logger.info("✅ 数据集划分完成")
            if split_result.get('success'):
                logger.info(f"📊 划分结果:")
                logger.info(f"   训练集: {split_result.get('train_count', 0)} 个")
                logger.info(f"   验证集: {split_result.get('val_count', 0)} 个")
                logger.info(f"   测试集: {split_result.get('test_count', 0)} 个")
            
            print("\n🔄 步骤4: COCO格式转换...")
            logger.info("📋 步骤4: 开始COCO格式转换")
            
            # 转换为COCO格式
            coco_result = self._convert_to_coco()
            logger.info("✅ COCO格式转换完成")
            
            # 最终统计
            final_stats = {
                "success": True,
                "message": "一键转换完成",
                "valid_pairs": len(self.valid_pairs) if hasattr(self, 'valid_pairs') else 0,
                "classes": len(self.classes) if hasattr(self, 'classes') else 0,
                "dimension_mismatches": len(self.dimension_mismatches) if hasattr(self, 'dimension_mismatches') else 0,
                "channel_mismatches": len(self.channel_mismatches) if hasattr(self, 'channel_mismatches') else 0,
                "missing_xml_files": len(self.missing_xml_files) if hasattr(self, 'missing_xml_files') else 0,
                "missing_image_files": len(self.missing_image_files) if hasattr(self, 'missing_image_files') else 0,
                "annotations_output_dir": str(self.annotations_output_dir)
            }
            
            logger.info("=" * 80)
            logger.info("🎉 一键转换处理完成！")
            logger.info("=" * 80)
            logger.info("📊 最终处理统计:")
            logger.info(f"   有效文件对: {final_stats['valid_pairs']} 个")
            logger.info(f"   类别数量: {final_stats['classes']} 个")
            logger.info(f"   尺寸不匹配: {final_stats['dimension_mismatches']} 个")
            logger.info(f"   通道不匹配: {final_stats['channel_mismatches']} 个")
            logger.info(f"   缺少XML: {final_stats['missing_xml_files']} 个")
            logger.info(f"   缺少图像: {final_stats['missing_image_files']} 个")
            logger.info(f"   清洗输出目录: {final_stats['annotations_output_dir']}")
            logger.info("=" * 80)
            
            print("✅ 一键转换完成！")
            print("📋 详细处理日志已保存到日志文件中")
            print(f"🗂️  清洗后的XML文件保存在: {self.annotations_output_dir}")
            
            return final_stats
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logger.error("=" * 80)
            logger.error("❌ 一键转换处理失败！")
            logger.error("=" * 80)
            logger.error(f"错误信息: {error_msg}")
            logger.error(f"错误类型: {type(e).__name__}")
            logger.error(f"错误位置: {__import__('traceback').format_exc()}")
            logger.error("=" * 80)
            
            print(f"❌ 一键转换失败！")
            print(f"❗ 错误信息: {error_msg}")
            print("📋 详细错误日志已保存到日志文件中")
            
            return {"success": False, "message": error_msg}
        
        finally:
            # 确保线程池正确关闭
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
                logger.info("🧵 线程池已关闭")

    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'dataset_name': self.dataset_name,
            'dataset_path': str(self.dataset_path),
            'annotations_dir': str(self.annotations_dir),
            'annotations_output_dir': str(self.annotations_output_dir),
            'valid_pairs': len(self.valid_pairs),
            'classes': sorted(list(self.classes)),
            'class_count': len(self.classes),
            'missing_xml_files': len(self.missing_xml_files),
            'missing_image_files': len(self.missing_image_files),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'max_workers': self.max_workers
        }


        logger.info("🚀 开始一键完成转换...")

        # 提示用户确认
        print("\n" + "="*80)
        print("⚠️  警告：即将开始处理数据集")
        print("="*80)
        print("📋 处理内容包括：")
        print("   1. 清洗和修正XML标注文件")
        print("   2. 修正图片和XML尺寸不匹配问题")
        print("   3. 删除空标注和无效文件")
        print("   4. 划分训练/验证/测试集")
        print("   5. 转换为COCO格式")
        print("\n🔥 重要提醒：请确保已备份原始数据集！")
        print("="*80)

        if not skip_confirmation:
            user_input = input("是否继续处理？输入 Y 继续，N 取消: ").strip()
            if user_input.upper() != 'Y':
                print("❌ 操作已取消")
                logger.info("用户取消了一键转换操作")
                return {"success": False, "message": "用户取消操作"}

            print("\n✅ 用户确认继续，开始处理...")
            logger.info("用户确认开始一键转换")

            try:
                # 创建清洗后的输出目录
                self.annotations_output_dir.mkdir(exist_ok=True)
                logger.info(f"创建清洗输出目录: {self.annotations_output_dir}")

                # 步骤1: 数据集基本验证
                print("\n📋 步骤1: 数据集基本验证...")
                logger.info("开始数据集基本验证")
                self.validate_dataset()

                # 步骤2: 检查和修正图像尺寸
                print("\n🔧 步骤2: 检查和修正图像尺寸...")
                logger.info("开始检查和修正图像尺寸")
                self.check_and_fix_image_dimensions()

                # 步骤3: 清洗XML文件
                print("\n🧹 步骤3: 清洗XML文件...")
                logger.info("开始清洗XML文件")
                self._clean_xml_files_parallel()

                # 步骤4: 删除空标注文件
                print("\n🗑️  步骤4: 删除空标注文件...")
                logger.info("开始删除空标注文件")
                self.remove_empty_annotations()

                # 步骤5: 提取类别信息
                print("\n🏷️  步骤5: 提取类别信息...")
                logger.info("开始提取类别信息")
                self.extract_classes()

                # 步骤6: 划分数据集
                print("\n📊 步骤6: 划分数据集...")
                logger.info("开始划分数据集")
                self.split_dataset()

                # 步骤7: 转换为COCO格式（只生成train和val）
                print("\n🔄 步骤7: 转换为COCO格式...")
                logger.info("开始转换为COCO格式")
                self._convert_to_coco_train_val_only()

                # 收集处理结果
                result = {
                    "success": True,
                    "message": "一键转换完成",
                    "dataset_name": self.dataset_name,
                    "valid_pairs": len(self.valid_pairs),
                    "classes": len(self.classes),
                    "dimension_mismatches": len(self.dimension_mismatches),
                    "channel_mismatches": len(self.channel_mismatches),
                    "missing_xml_files": len(self.missing_xml_files),
                    "missing_image_files": len(self.missing_image_files),
                    "annotations_output_dir": str(self.annotations_output_dir)
                }

                print("\n" + "="*80)
                print("🎉 一键转换完成！")
                print("="*80)
                print(f"📁 清洗后的XML文件保存在: {self.annotations_output_dir}")
                print(f"📊 有效文件对: {len(self.valid_pairs)} 个")
                print(f"🏷️  发现类别: {len(self.classes)} 个")
                print(f"📐 修正尺寸不匹配: {len(self.dimension_mismatches)} 个")
                print(f"🎨 修正通道不匹配: {len(self.channel_mismatches)} 个")
                print("="*80)

                logger.info("一键转换成功完成")
                return result

            except Exception as e:
                error_msg = f"一键转换过程中出错: {str(e)}"
                logger.error(error_msg)
                print(f"\n❌ {error_msg}")
                return {"success": False, "message": error_msg}

    def _clean_xml_files_parallel(self):
        """使用线程池并行清洗XML文件并保存到Annotations_clear目录"""
        xml_files = list(self.annotations_dir.glob("*.xml"))
        
        if not xml_files:
            logger.warning("未找到XML文件")
            return
        
        logger.info(f"开始并行清洗 {len(xml_files)} 个XML文件")
        
        # 使用线程池并行处理
        futures = []
        for xml_file in xml_files:
            future = self.thread_pool.submit(self._clean_single_xml_file, xml_file)
            futures.append(future)
        
        # 等待所有任务完成
        cleaned_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="清洗XML文件"):
            try:
                if future.result():
                    cleaned_count += 1
            except Exception as e:
                logger.error(f"清洗XML文件时出错: {e}")
        
        logger.info(f"XML文件清洗完成，成功清洗 {cleaned_count}/{len(xml_files)} 个文件")
    
    def _clean_single_xml_file(self, xml_file: Path) -> bool:
        """清洗单个XML文件并保存到输出目录
        
        Args:
            xml_file: XML文件路径
            
        Returns:
            bool: 是否成功清洗
        """
        try:
            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 检查是否需要清洗
            needs_cleaning = False
            
            # 移除无效的object标签
            objects_to_remove = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is None or not name_elem.text or name_elem.text.strip() == '':
                    objects_to_remove.append(obj)
                    needs_cleaning = True
                    continue
                
                # 检查边界框
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    objects_to_remove.append(obj)
                    needs_cleaning = True
                    continue
                
                # 检查边界框坐标
                try:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    if xmin >= xmax or ymin >= ymax:
                        objects_to_remove.append(obj)
                        needs_cleaning = True
                except (ValueError, AttributeError):
                    objects_to_remove.append(obj)
                    needs_cleaning = True
            
            # 移除无效对象
            for obj in objects_to_remove:
                root.remove(obj)
            
            # 如果需要清洗或者强制保存清洗版本，则保存到输出目录
            output_file = Path(os.path.join(str(self.annotations_output_dir), xml_file.name))
            
            # 使用minidom保持格式化
            rough_string = ET.tostring(root, 'unicode')
            reparsed = minidom.parseString(rough_string)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # 移除第一行的XML声明，然后添加自定义的
                lines = reparsed.toprettyxml(indent="  ").split('\n')[1:]
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('\n'.join(line for line in lines if line.strip()))
            
            if needs_cleaning:
                logger.info(f"清洗并保存XML文件: {xml_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"清洗XML文件 {xml_file.name} 时出错: {e}")
            return False
    
    def _convert_to_coco_train_val_only(self):
        """转换为COCO格式，只生成train_coco.json和val_coco.json"""
        try:
            # 读取train.txt和val.txt
            train_file = Path(os.path.join(str(self.imagesets_dir), "train.txt"))
            val_file = Path(os.path.join(str(self.imagesets_dir), "val.txt"))
            
            if train_file.exists():
                logger.info("转换训练集为COCO格式")
                self._convert_split_to_coco("train", train_file)
            
            if val_file.exists():
                logger.info("转换验证集为COCO格式")
                self._convert_split_to_coco("val", val_file)
            
            # 确保不生成test_coco.json
            test_coco_file = Path(os.path.join(str(self.dataset_path), "test_coco.json"))
            if test_coco_file.exists():
                test_coco_file.unlink()
                logger.info("删除了不需要的test_coco.json文件")
                
        except Exception as e:
            logger.error(f"转换COCO格式时出错: {e}")
            raise
    
    def _convert_split_to_coco(self, split_name: str, split_file: Path):
        """转换指定划分为COCO格式
        
        Args:
            split_name: 划分名称 (train/val)
            split_file: 划分文件路径
        """
        import json
        from datetime import datetime
        
        # 读取文件列表
        with open(split_file, 'r', encoding='utf-8') as f:
            file_names = [line.strip() for line in f.readlines()]
        
        # 初始化COCO格式数据
        coco_data = {
            "info": {
                "description": f"{self.dataset_name} {split_name} dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "VOCDataset Converter",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 添加类别信息
        class_list = sorted(list(self.classes))
        for idx, class_name in enumerate(class_list, 1):
            coco_data["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": "object"
            })
        
        # 创建类别名称到ID的映射
        class_to_id = {class_name: idx for idx, class_name in enumerate(class_list, 1)}
        
        image_id = 1
        annotation_id = 1
        
        # 处理每个文件
        for file_name in tqdm(file_names, desc=f"转换{split_name}集"):
            # 图像文件路径
            img_file = Path(os.path.join(str(self.images_dir), f"{file_name}.jpg"))
            xml_file = Path(os.path.join(str(self.annotations_output_dir), f"{file_name}.xml"))  # 使用清洗后的XML
            
            if not img_file.exists() or not xml_file.exists():
                continue
            
            # 读取图像信息
            try:
                import cv2
                img = cv2.imread(str(img_file))
                height, width = img.shape[:2]
            except:
                continue
            
            # 添加图像信息
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": f"{file_name}.jpg",
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })
            
            # 解析XML标注
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in class_to_id:
                        continue
                    
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    # 计算COCO格式的边界框 [x, y, width, height]
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    area = bbox_width * bbox_height
                    
                    # 添加标注信息
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_to_id[name],
                        "segmentation": [],
                        "area": area,
                        "bbox": [xmin, ymin, bbox_width, bbox_height],
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
                    
            except Exception as e:
                logger.warning(f"解析XML文件 {xml_file.name} 时出错: {e}")
                continue
            
            image_id += 1
        
        # 保存COCO格式文件
        output_file = Path(os.path.join(str(self.dataset_path), f"{split_name}_coco.json"))
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"COCO格式文件已保存: {output_file}")
        logger.info(f"{split_name}集包含 {len(coco_data['images'])} 张图像，{len(coco_data['annotations'])} 个标注")


if __name__ == "__main__":
    # 测试VOC数据集类
    dataset_path = os.path.join(".", "dataset", "Fruit")
    
    try:
        voc_dataset = VOCDataset(dataset_path)
        print("数据集信息:", voc_dataset.get_dataset_info())
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误: {e}")

    