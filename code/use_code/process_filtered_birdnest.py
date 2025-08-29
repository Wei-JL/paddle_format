#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理已过滤的BirdNest数据集
使用Annotations_filtered_birdnest文件夹进行数据清洗、划分和COCO转换
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from code.dataset_handler.voc_dataset import VOCDataset

def main():
    """
    处理已过滤的BirdNest数据集
    """
    
    print("BirdNest数据集处理 - 使用已过滤的标注文件")
    print("=" * 80)
    
    # 数据集配置
    dataset_path = "D:/WJL/project/BirdNest"
    annotations_folder = "Annotations_filtered_birdnest"  # 使用已过滤的标注文件夹
    
    print(f"📁 数据集路径: {dataset_path}")
    print(f"📂 标注文件夹: {annotations_folder}")
    print("=" * 80)
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        return False
    
    # 检查标注文件夹是否存在
    annotations_path = os.path.join(dataset_path, annotations_folder)
    if not os.path.exists(annotations_path):
        print(f"❌ 错误: 标注文件夹不存在: {annotations_path}")
        print("请确认标签过滤步骤已完成")
        return False
    
    try:
        # 初始化VOC数据集，使用已过滤的标注文件夹
        dataset = VOCDataset(
            dataset_path=dataset_path,
            annotations_folder_name=annotations_folder,  # 指定使用过滤后的标注文件夹
            max_workers=4,  # 使用4个线程并发处理
            train_ratio=0.8,  # 训练集80%
            val_ratio=0.2,   # 验证集20%
            test_ratio=0.0,  # 测试集0%
            output_annotations_name="Annotations_clean"  # 指定输出目录名称
        )
        
        print("✅ VOCDataset对象创建成功")
        print()
        print(f"📊 数据集信息:")
        print(f"   数据集名称: {dataset.dataset_name}")
        print(f"   数据集路径: {dataset_path}")
        print(f"   标注文件夹: {annotations_folder}")
        print(f"   线程数: 4")
        print(f"   划分比例: 训练集80% | 验证集20% | 测试集0%")
        print()
        
        # 执行一键转换处理
        print("🚀 开始执行数据清洗、划分和COCO转换...")
        result = dataset.one_click_complete_conversion()
        
        if result:
            print()
            print("🎉 BirdNest数据集处理完成!")
            print("=" * 80)
            print(f"📊 处理结果:")
            print(f"   有效文件对: {result.get('valid_pairs', 'N/A')} 个")
            print(f"   发现类别: {result.get('total_classes', 'N/A')} 个") 
            print(f"   尺寸不匹配修正: {result.get('dimension_mismatches', 0)} 个")
            print(f"   通道不匹配修正: {result.get('channel_mismatches', 0)} 个")
            print("=" * 80)
            
            # 显示输出文件
            print()
            print("📁 生成的文件:")
            
            # VOC格式文件
            imagesets_dir = os.path.join(dataset_path, "ImageSets", "Main")
            voc_files = ["train.txt", "val.txt", "trainval.txt", "labels.txt"]
            for file_name in voc_files:
                file_path = os.path.join(imagesets_dir, file_name)
                if os.path.exists(file_path):
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path} (未生成)")
            
            # COCO格式文件
            coco_files = ["train_coco.json", "val_coco.json"]
            for file_name in coco_files:
                file_path = os.path.join(dataset_path, file_name)
                if os.path.exists(file_path):
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path} (未生成)")
            
            # 显示类别信息
            labels_file = os.path.join(imagesets_dir, "labels.txt")
            if os.path.exists(labels_file):
                print()
                print("🏷️  数据集类别:")
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = f.read().strip().split('\n')
                    for i, label in enumerate(labels, 1):
                        print(f"   {i}. {label}")
                
                print()
                print(f"✅ 总共保留了 {len(labels)} 个类别")
            
            # 显示数据集划分信息
            train_file = os.path.join(imagesets_dir, "train.txt")
            val_file = os.path.join(imagesets_dir, "val.txt")
            
            if os.path.exists(train_file) and os.path.exists(val_file):
                with open(train_file, 'r') as f:
                    train_count = len(f.readlines())
                with open(val_file, 'r') as f:
                    val_count = len(f.readlines())
                
                print()
                print("📊 数据集划分:")
                print(f"   训练集: {train_count} 个文件")
                print(f"   验证集: {val_count} 个文件")
                print(f"   总计: {train_count + val_count} 个文件")
            
            return True
        else:
            print("❌ 处理失败，请检查日志获取详细信息")
            return False
            
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始处理BirdNest数据集（使用已过滤的标注文件）...")
    print()
    
    success = main()
    
    print()
    if success:
        print("🎉 BirdNest数据集处理完成!")
        print("📁 VOC格式文件已生成到 ImageSets/Main/ 目录")
        print("📁 COCO格式文件已生成到数据集根目录")
    else:
        print("❌ BirdNest数据集处理失败!")
    
    print("=" * 80)