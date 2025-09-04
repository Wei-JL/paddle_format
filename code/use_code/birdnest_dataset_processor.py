#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BirdNest数据集一键处理器

功能：
1. 一键清洗BirdNest数据集
2. 转换为VOC格式并划分数据集 (训练集:验证集:测试集=0.88:0.11:0.01)
3. 转换为COCO格式
4. 转换为YOLOv13格式 (使用线程池拷贝图像)
5. 输出到指定路径: D:\WJL\project\BirdNest\BirdNest_yolov13
"""

import os
import sys
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

# 添加项目路径到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from dataset_handler.voc_dataset import VOCDataset
from dataset_handler.yolo_series_dataset import YOLOSeriesDataset
from logger_code.logger_sys import get_logger
from global_var.global_cls import *

# 获取当前文件名作为日志标识
current_filename = Path(__file__).stem
logger = get_logger(current_filename)


class BirdNestDatasetProcessor:
    """
    BirdNest数据集一键处理器
    
    处理流程：
    1. VOC格式清洗和数据集划分
    2. COCO格式转换
    3. YOLOv13格式转换（多线程拷贝图像）
    """
    
    def __init__(self, dataset_path: str = r"D:\WJL\project\BirdNest"):
        """
        初始化BirdNest数据集处理器
        
        Args:
            dataset_path: BirdNest数据集路径
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.dataset_name = os.path.basename(os.path.normpath(dataset_path))
        
        # 输出路径
        self.yolo_output_path = os.path.join(dataset_path, f"{self.dataset_name}_yolov13")
        
        # 数据集划分比例 (训练集:验证集:测试集=0.88:0.11:0.01)
        self.train_ratio = 0.88
        self.val_ratio = 0.11
        self.test_ratio = 0.01
        
        # 线程池 - 用于多线程拷贝图像
        self.max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"初始化BirdNest数据集处理器")
        logger.info(f"数据集路径: {self.dataset_path}")
        logger.info(f"YOLOv13输出路径: {self.yolo_output_path}")
        logger.info(f"数据集划分比例 - 训练集:{self.train_ratio}, 验证集:{self.val_ratio}, 测试集:{self.test_ratio}")
        logger.info(f"线程池大小: {self.max_workers}")
    
    def __del__(self):
        """析构函数，关闭线程池"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
    
    def _validate_dataset_path(self) -> bool:
        """
        验证数据集路径是否有效
        
        Returns:
            验证是否通过
        """
        if not os.path.exists(self.dataset_path):
            logger.error(f"数据集路径不存在: {self.dataset_path}")
            return False
        
        # 检查必要的文件夹
        required_dirs = [ANNOTATIONS_DIR, JPEGS_DIR]
        for dir_name in required_dirs:
            dir_path = os.path.join(self.dataset_path, dir_name)
            if not os.path.exists(dir_path):
                logger.error(f"必要文件夹不存在: {dir_path}")
                return False
        
        logger.info("数据集路径验证通过")
        return True
    
    def process_voc_dataset(self) -> bool:
        """
        处理VOC格式数据集（清洗和划分）
        
        Returns:
            处理是否成功
        """
        try:
            logger.info("开始VOC格式数据集处理...")
            
            # 创建VOC数据集处理器
            voc_dataset = VOCDataset(
                dataset_path=self.dataset_path,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio
            )
            
            # 一键完成转换（包括清洗、划分、COCO转换）
            success = voc_dataset.one_click_complete_conversion()
            
            if success:
                logger.info("VOC格式数据集处理完成")
                return True
            else:
                logger.error("VOC格式数据集处理失败")
                return False
                
        except Exception as e:
            logger.error(f"VOC格式数据集处理异常: {str(e)}")
            return False
    
    def process_yolo_dataset(self) -> bool:
        """
        处理YOLOv13格式数据集转换
        
        Returns:
            处理是否成功
        """
        try:
            logger.info("开始YOLOv13格式数据集转换...")
            
            # 创建自定义YOLO转换器（支持多线程拷贝和自定义输出路径）
            yolo_converter = CustomYOLOConverter(
                dataset_path=self.dataset_path,
                output_path=self.yolo_output_path,
                thread_pool=self.thread_pool
            )
            
            # 执行转换
            success = yolo_converter.convert_to_yolo()
            
            if success:
                logger.info("YOLOv13格式数据集转换完成")
                return True
            else:
                logger.error("YOLOv13格式数据集转换失败")
                return False
                
        except Exception as e:
            logger.error(f"YOLOv13格式数据集转换异常: {str(e)}")
            return False
    
    def process_complete_pipeline(self) -> bool:
        """
        执行完整的数据集处理流程
        
        Returns:
            处理是否成功
        """
        try:
            logger.info("=" * 60)
            logger.info("开始BirdNest数据集完整处理流程")
            logger.info("=" * 60)
            
            # 1. 验证数据集路径
            if not self._validate_dataset_path():
                return False
            
            # 2. VOC格式处理（清洗、划分、COCO转换）
            logger.info("步骤1: VOC格式数据集处理")
            if not self.process_voc_dataset():
                logger.error("VOC格式处理失败，终止流程")
                return False
            
            # 3. YOLOv13格式转换
            logger.info("步骤2: YOLOv13格式数据集转换")
            if not self.process_yolo_dataset():
                logger.error("YOLOv13格式转换失败，终止流程")
                return False
            
            logger.info("=" * 60)
            logger.info("BirdNest数据集完整处理流程成功完成！")
            logger.info("=" * 60)
            
            # 打印处理结果摘要
            self._print_processing_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"完整处理流程异常: {str(e)}")
            return False
        finally:
            # 确保线程池关闭
            self.thread_pool.shutdown(wait=True)
    
    def _print_processing_summary(self):
        """打印处理结果摘要"""
        try:
            logger.info("处理结果摘要:")
            logger.info(f"  数据集名称: {self.dataset_name}")
            logger.info(f"  原始数据路径: {self.dataset_path}")
            logger.info(f"  YOLOv13输出路径: {self.yolo_output_path}")
            logger.info(f"  数据集划分比例: 训练集{self.train_ratio}, 验证集{self.val_ratio}, 测试集{self.test_ratio}")
            
            # 统计处理后的文件数量
            if os.path.exists(self.yolo_output_path):
                for split in ['train', 'val', 'test']:
                    images_dir = os.path.join(self.yolo_output_path, 'images', split)
                    labels_dir = os.path.join(self.yolo_output_path, 'labels', split)
                    
                    if os.path.exists(images_dir):
                        image_count = len([f for f in os.listdir(images_dir) 
                                         if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)])
                        label_count = len([f for f in os.listdir(labels_dir) 
                                         if f.lower().endswith('.txt')]) if os.path.exists(labels_dir) else 0
                        
                        logger.info(f"  {split}集: {image_count} 张图片, {label_count} 个标签文件")
            
        except Exception as e:
            logger.error(f"打印摘要失败: {str(e)}")


class CustomYOLOConverter(YOLOSeriesDataset):
    """
    自定义YOLO转换器
    
    扩展YOLOSeriesDataset以支持：
    1. 自定义输出路径
    2. 多线程拷贝图像
    3. 相对路径和COCO格式yml文件
    """
    
    def __init__(self, dataset_path: str, output_path: str, thread_pool: ThreadPoolExecutor, 
                 annotations_folder_name: str = ANNOTATIONS_OUTPUT_DIR):
        """
        初始化自定义YOLO转换器
        
        Args:
            dataset_path: 数据集路径
            output_path: 自定义输出路径
            thread_pool: 线程池
            annotations_folder_name: 标签文件夹名称
        """
        # 调用父类初始化
        super().__init__(dataset_path, annotations_folder_name)
        
        # 覆盖输出路径
        self.output_dir = os.path.abspath(output_path)
        self.output_images_dir = os.path.join(self.output_dir, "images")
        self.output_labels_dir = os.path.join(self.output_dir, "labels")
        
        # 线程池
        self.thread_pool = thread_pool
        
        logger.info(f"自定义YOLO转换器初始化完成")
        logger.info(f"输出路径: {self.output_dir}")
    
    def _copy_image_file(self, source_path: str, target_path: str) -> bool:
        """
        拷贝单个图像文件（线程安全）
        
        Args:
            source_path: 源文件路径
            target_path: 目标文件路径
            
        Returns:
            拷贝是否成功
        """
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 拷贝文件
            shutil.copy2(source_path, target_path)
            return True
            
        except Exception as e:
            logger.error(f"拷贝图像文件失败: {source_path} -> {target_path}, 错误: {str(e)}")
            return False
    
    def _process_split(self, split_name: str):
        """
        处理单个数据集划分（重写以支持多线程拷贝）
        
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
        
        # 准备拷贝任务列表
        copy_tasks = []
        label_tasks = []
        
        for file_name in file_names:
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
            
            # 准备拷贝任务
            image_ext = os.path.splitext(source_image_path)[1]
            target_image_path = os.path.join(self.output_images_dir, split_name, f"{file_name}{image_ext}")
            copy_tasks.append((source_image_path, target_image_path))
            
            # 准备标签任务
            target_label_path = os.path.join(self.output_labels_dir, split_name, f"{file_name}.txt")
            label_tasks.append((target_label_path, yolo_lines))
        
        # 使用线程池并发拷贝图像文件
        logger.info(f"使用线程池拷贝 {len(copy_tasks)} 个图像文件...")
        
        from tqdm import tqdm
        
        # 提交拷贝任务到线程池
        copy_futures = []
        for source_path, target_path in copy_tasks:
            future = self.thread_pool.submit(self._copy_image_file, source_path, target_path)
            copy_futures.append(future)
        
        # 等待所有拷贝任务完成并显示进度
        success_count = 0
        for future in tqdm(copy_futures, desc=f"拷贝{split_name}图像"):
            if future.result():
                success_count += 1
        
        # 保存标签文件（单线程，因为文件较小）
        logger.info(f"保存 {len(label_tasks)} 个标签文件...")
        for target_label_path, yolo_lines in tqdm(label_tasks, desc=f"保存{split_name}标签"):
            try:
                # 确保目标目录存在
                os.makedirs(os.path.dirname(target_label_path), exist_ok=True)
                
                with open(target_label_path, 'w', encoding=DEFAULT_ENCODING) as f:
                    f.write(NEWLINE.join(yolo_lines))
            except Exception as e:
                logger.error(f"保存标签文件失败: {target_label_path}, 错误: {str(e)}")
        
        logger.info(f"{split_name} 数据集处理完成: {success_count}/{len(copy_tasks)} 个图像文件成功拷贝")


def main():
    """主函数"""
    try:
        # 创建BirdNest数据集处理器
        processor = BirdNestDatasetProcessor()
        
        # 执行完整处理流程
        success = processor.process_complete_pipeline()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 BirdNest数据集处理完成！")
            print("=" * 60)
            print(f"📁 YOLOv13格式数据集已保存到: {processor.yolo_output_path}")
            print("📊 数据集划分比例: 训练集88%, 验证集11%, 测试集1%")
            print("✅ 包含VOC格式清洗、COCO格式转换、YOLOv13格式转换")
            print("🚀 可直接用于YOLOv13模型训练！")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ BirdNest数据集处理失败！")
            print("=" * 60)
            print("请检查日志文件获取详细错误信息")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n用户中断处理流程")
    except Exception as e:
        print(f"\n处理流程异常: {str(e)}")
        logger.error(f"主函数异常: {str(e)}")


if __name__ == "__main__":
    main()