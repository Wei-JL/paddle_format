"""
VOC数据集处理器 - 优化版本
支持自定义标注文件夹名称和标签筛选功能
"""

import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import random

# 导入全局变量和日志
from code.global_var.global_cls import *
from code.logger_code.my_logger import logger


class VOCDataset:
    """VOC数据集处理器 - 优化版本"""
    
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
        
        # 数据集统计信息
        self.classes = set()
        self.class_counts = {}
        self.total_images = 0
        self.total_annotations = 0
        
        # 日志记录
        logger.info(f"初始化VOC数据集: {self.dataset_name}")
        logger.info(f"数据集路径: {self.dataset_path}")
        logger.info(f"标注文件夹: {self.annotations_folder_name}")
        logger.info(f"划分比例 - 训练集: {self.train_ratio} 验证集: {self.val_ratio} 测试集: {self.test_ratio}")
        logger.info(f"线程池配置 - 最大工作线程: {self.max_workers}")

    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)

    def filter_specified_labels(self, label_list: List[str], keep_mode: bool = True) -> Dict:
        """
        筛选指定标签的标注框
        
        Args:
            label_list: 标签列表
            keep_mode: True表示保留label_list中的标签，False表示移除label_list中的标签
        
        Returns:
            dict: 处理结果统计
        """
        mode_desc = "保留" if keep_mode else "移除"
        logger.info(f"开始{mode_desc}指定标签: {label_list}")
        
        # 使用指定的标注文件夹
        annotations_dir = self.dataset_path / self.annotations_folder_name
        if not annotations_dir.exists():
            logger.error(f"标注文件夹不存在: {annotations_dir}")
            return {"success": False, "error": "标注文件夹不存在"}
        
        removed_count = 0
        kept_count = 0
        processed_count = 0
        empty_files = []
        
        # 创建输出目录
        output_dir = self.dataset_path / f"{self.annotations_folder_name}_filtered"
        output_dir.mkdir(exist_ok=True)
        logger.info(f"筛选结果将保存到: {output_dir}")
        
        xml_files = list(annotations_dir.glob("*.xml"))
        logger.info(f"发现 {len(xml_files)} 个XML文件需要处理")
        
        for xml_file in tqdm(xml_files, desc=f"{mode_desc}标签"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 找到需要处理的object元素
                objects_to_process = []
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None:
                        label_name = name_elem.text
                        
                        if keep_mode:
                            # 保留模式：移除不在列表中的标签
                            if label_name not in label_list:
                                objects_to_process.append(obj)
                        else:
                            # 移除模式：移除在列表中的标签
                            if label_name in label_list:
                                objects_to_process.append(obj)
                
                # 处理找到的object元素
                for obj in objects_to_process:
                    root.remove(obj)
                    removed_count += 1
                
                # 统计保留的标注框
                remaining_objects = len(root.findall('object'))
                kept_count += remaining_objects
                
                # 保存到输出目录
                output_file = output_dir / xml_file.name
                
                if remaining_objects == 0:
                    # 如果没有剩余标注框，记录但不保存文件
                    empty_files.append(xml_file.name)
                    logger.debug(f"文件 {xml_file.name} 筛选后无标注框，跳过保存")
                else:
                    # 保持原有XML格式
                    tree.write(output_file, encoding='utf-8', xml_declaration=True)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"处理文件 {xml_file} 时出错: {e}")
        
        result = {
            "success": True,
            "mode": mode_desc,
            "processed_files": processed_count,
            "removed_annotations": removed_count,
            "kept_annotations": kept_count,
            "empty_files": len(empty_files),
            "output_dir": str(output_dir)
        }
        
        logger.info(f"标签筛选完成:")
        logger.info(f"  模式: {mode_desc}标签 {label_list}")
        logger.info(f"  处理文件: {processed_count} 个")
        logger.info(f"  移除标注框: {removed_count} 个")
        logger.info(f"  保留标注框: {kept_count} 个")
        logger.info(f"  空文件数: {len(empty_files)} 个")
        logger.info(f"  输出目录: {output_dir}")
        
        return result

    def one_click_complete_conversion(self, skip_confirmation=False):
        """一键完成转换并修复所有问题"""
        logger.info("="*80)
        logger.info("🚀 开始一键完成转换处理（多线程版本）")
        logger.info("="*80)
        logger.info(f"数据集名称: {self.dataset_name}")
        logger.info(f"数据集路径: {self.dataset_path}")
        logger.info(f"标注文件夹: {self.annotations_folder_name}")
        logger.info(f"线程池配置: {self.max_workers} 个工作线程")
        logger.info(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
                return {"success": False, "message": "用户取消操作"}

        try:
            # 创建清洗后的输出目录
            self.annotations_output_dir.mkdir(exist_ok=True)
            logger.info(f"创建清洗输出目录: {self.annotations_output_dir}")

            # 步骤1: 数据集基本验证
            print("\n📋 步骤1: 数据集基本验证...")
            logger.info("📋 步骤1: 开始数据验证和清洗")
            
            logger.info("🔍 验证数据集基本结构...")
            self._validate_basic_structure()
            logger.info("✅ 数据集结构验证完成")
            
            logger.info("🔗 开始并行匹配图像和标注文件...")
            self._match_files_parallel()
            logger.info("✅ 并行文件匹配完成 - 有效文件对: {len(self.valid_files)} 个")
            
            logger.info("🧹 开始清理空标注文件并清洗XML...")
            empty_count = self._remove_empty_annotations_and_clean()
            logger.info(f"✅ XML清洗完成 - 删除空标注: {empty_count} 个")
            
            # 步骤2: 提取类别信息
            logger.info("🏷️  开始提取类别信息...")
            self._extract_classes()
            logger.info("✅ 类别信息提取完成")
            
            # 步骤3: 检查和修正图像尺寸
            print("\n📋 步骤2: 检查和修正图像尺寸...")
            logger.info("📋 步骤2: 开始检查和修正图像尺寸")
            self.check_and_fix_image_dimensions()
            logger.info("✅ 图像尺寸检查和修正完成")
            
            # 步骤4: 划分数据集
            print("\n📋 步骤3: 划分数据集...")
            logger.info("📋 步骤3: 开始划分数据集")
            self.split_dataset_optimized()
            logger.info("✅ 数据集划分完成")
            
            # 步骤5: 转换为COCO格式
            print("\n📋 步骤4: 转换为COCO格式...")
            logger.info("📋 步骤4: 开始转换为COCO格式")
            self._convert_to_coco_train_val_only()
            logger.info("✅ COCO格式转换完成")
            
            # 完成处理
            print("\n" + "="*80)
            print("🎉 数据集处理完成！")
            print("="*80)
            
            result = {
                "success": True,
                "dataset_name": self.dataset_name,
                "dataset_path": str(self.dataset_path),
                "annotations_folder": self.annotations_folder_name,
                "total_classes": len(self.classes),
                "classes": list(self.classes),
                "total_images": self.total_images,
                "total_annotations": self.total_annotations,
                "clean_output_dir": str(self.annotations_output_dir)
            }
            
            logger.info("🎉 一键转换处理完成！")
            logger.info(f"处理结果: {result}")
            
            return result
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {e}"
            logger.error(error_msg)
            print(f"\n❌ {error_msg}")
            return {"success": False, "error": str(e)}

    def get_dataset_info(self):
        """获取数据集基本信息"""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path),
            "annotations_folder": self.annotations_folder_name,
            "annotations_dir": str(self.annotations_dir),
            "images_dir": str(self.images_dir),
            "imagesets_dir": str(self.imagesets_dir),
            "clean_output_dir": str(self.annotations_output_dir),
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "max_workers": self.max_workers
        }

    # 这里需要包含原有的所有其他方法...
    # 为了简化，我只展示了关键的新增和修改的方法
    # 实际使用时需要从原文件复制所有其他方法


if __name__ == "__main__":
    # 测试优化版本
    dataset_path = r"D:\WJL\project\BirdNest"
    
    try:
        # 测试自定义标注文件夹
        voc_dataset = VOCDataset(
            dataset_path=dataset_path,
            annotations_folder_name="Annotations"  # 可以自定义
        )
        
        print("数据集信息:", voc_dataset.get_dataset_info())
        
        # 测试标签筛选功能
        keep_labels = [100, 101, 102, 113, 140, 150, 153, 154, 155, 200, 201, 202, 203, 204, 205, 220, 221, 240]
        keep_labels_str = [str(label) for label in keep_labels]  # 转换为字符串
        
        print(f"\n开始筛选标签，保留: {keep_labels}")
        result = voc_dataset.filter_specified_labels(keep_labels_str, keep_mode=True)
        print(f"筛选结果: {result}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")