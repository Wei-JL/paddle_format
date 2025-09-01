#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO系列数据格式转换示例

本示例展示如何使用YOLOSeriesDataset类将已处理的VOC格式数据集转换为YOLO格式。

前提条件:
1. 数据集已经通过VOCDataset类进行过一键处理
2. 存在Annotations_clear文件夹（清洗后的标注文件）
3. 存在ImageSets/Main/文件夹（数据集划分文件）

使用方法:
1. 确保数据集已经过VOCDataset处理
2. 运行此脚本
3. 查看output/数据集名称_yolo文件夹

输出格式:
- images/train/, images/val/, images/test/ : 图片文件
- labels/train/, labels/val/, labels/test/ : YOLO格式标签文件
- dataset_name.yaml : YOLO数据集配置文件
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from code.dataset_handler.yolo_series_dataset import YOLOSeriesDataset


def main():
    """主函数"""
    print("=" * 60)
    print("YOLO系列数据格式转换工具")
    print("支持: YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv13")
    print("=" * 60)
    
    # 已处理的数据集路径
    processed_dataset_path = os.path.join(project_root, "dataset", "Fruit")
    
    if not os.path.exists(processed_dataset_path):
        print(f"错误: 数据集路径不存在: {processed_dataset_path}")
        return
    
    # 检查是否已经过VOCDataset处理
    annotations_clear_dir = os.path.join(processed_dataset_path, "Annotations_clear")
    imagesets_dir = os.path.join(processed_dataset_path, "ImageSets", "Main")
    
    if not os.path.exists(annotations_clear_dir):
        print(f"错误: 未找到清洗后的标注文件夹: {annotations_clear_dir}")
        print("请先使用VOCDataset类进行数据集处理")
        return
    
    if not os.path.exists(imagesets_dir):
        print(f"错误: 未找到数据集划分文件夹: {imagesets_dir}")
        print("请先使用VOCDataset类进行数据集处理")
        return
    
    print(f"已处理数据集路径: {processed_dataset_path}")
    
    try:
        # 创建YOLO转换器
        yolo_converter = YOLOSeriesDataset(
            processed_dataset_path=processed_dataset_path,
            annotations_folder_name="Annotations_clear"
        )
        
        print("\n开始转换为YOLO格式...")
        
        # 执行转换
        success = yolo_converter.convert_to_yolo()
        
        if success:
            print("\n" + "=" * 60)
            print("YOLO格式转换完成!")
            print("=" * 60)
            
            # 输出结果路径
            output_dir = os.path.join("output", f"{os.path.basename(processed_dataset_path)}_yolo")
            print(f"输出目录: {output_dir}")
            
            # 验证输出结果
            verify_conversion_result(output_dir)
            
        else:
            print("\nYOLO格式转换失败，请检查日志信息")
    
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


def verify_conversion_result(output_dir):
    """验证转换结果"""
    try:
        print("\n验证转换结果:")
        
        if not os.path.exists(output_dir):
            print("  ❌ 输出目录不存在")
            return
        
        # 检查目录结构
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        
        if not os.path.exists(images_dir):
            print("  ❌ images目录不存在")
            return
        
        if not os.path.exists(labels_dir):
            print("  ❌ labels目录不存在")
            return
        
        print("  ✅ 目录结构正确")
        
        # 统计各个划分的文件数量
        splits = ['train', 'val', 'test']
        total_images = 0
        total_labels = 0
        
        for split in splits:
            split_images_dir = os.path.join(images_dir, split)
            split_labels_dir = os.path.join(labels_dir, split)
            
            if os.path.exists(split_images_dir):
                image_files = [f for f in os.listdir(split_images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                image_count = len(image_files)
                total_images += image_count
                
                label_files = [f for f in os.listdir(split_labels_dir) 
                             if f.lower().endswith('.txt')] if os.path.exists(split_labels_dir) else []
                label_count = len(label_files)
                total_labels += label_count
                
                print(f"  {split}集: {image_count} 张图片, {label_count} 个标签文件")
        
        print(f"  总计: {total_images} 张图片, {total_labels} 个标签文件")
        
        # 检查配置文件
        yaml_files = [f for f in os.listdir(output_dir) if f.endswith('.yaml')]
        if yaml_files:
            print(f"  ✅ 配置文件: {yaml_files[0]}")
        else:
            print("  ❌ 未找到YAML配置文件")
        
        # 检查标签文件格式
        sample_label_file = None
        for split in splits:
            split_labels_dir = os.path.join(labels_dir, split)
            if os.path.exists(split_labels_dir):
                label_files = [f for f in os.listdir(split_labels_dir) if f.endswith('.txt')]
                if label_files:
                    sample_label_file = os.path.join(split_labels_dir, label_files[0])
                    break
        
        if sample_label_file and os.path.exists(sample_label_file):
            with open(sample_label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    sample_line = lines[0].strip()
                    parts = sample_line.split()
                    if len(parts) == 5:
                        print(f"  ✅ 标签格式正确: {sample_line}")
                    else:
                        print(f"  ❌ 标签格式错误: {sample_line}")
        
        print("\n使用说明:")
        print("1. 将输出文件夹复制到YOLO训练环境")
        print("2. 使用.yaml配置文件进行模型训练")
        print("3. 标签格式: class_id center_x center_y width height (归一化坐标)")
        
    except Exception as e:
        print(f"验证转换结果时出错: {str(e)}")


if __name__ == "__main__":
    main()