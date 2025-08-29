#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BirdNest数据集标签过滤处理脚本
只保留指定的类别ID，其他类别将被移除
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
    BirdNest数据集处理主函数
    只保留指定的类别ID: [100,101,102,113,140,150,153,154,155,200,201,202,203,204,205,220,221,240]
    """
    
    print("BirdNest数据集标签过滤处理")
    print("=" * 80)
    
    # 数据集配置
    dataset_path = "D:/WJL/project/BirdNest"
    
    # 需要保留的类别ID列表
    keep_class_ids = [100, 101, 102, 113, 140, 150, 153, 154, 155, 
                      200, 201, 202, 203, 204, 205, 220, 221, 240]
    
    # 转换为字符串列表（XML中的类别名称通常是字符串）
    keep_labels = [str(class_id) for class_id in keep_class_ids]
    
    print(f"📁 数据集路径: {dataset_path}")
    print(f"✅ 保留类别ID: {keep_class_ids}")
    print(f"📂 输出目录: Annotations_filtered_birdnest")
    print("=" * 80)
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        print("请确认路径是否正确")
        return False
    
    try:
        # 初始化VOC数据集，使用include_labels只保留指定类别
        dataset = VOCDataset(
            dataset_path=dataset_path,
            include_labels=keep_labels,  # 只保留这些类别
            max_workers=4,  # 使用4个线程并发处理
            output_annotations_name="Annotations_filtered_birdnest"  # 输出文件夹名称
        )
        
        print("✅ VOCDataset对象创建成功")
        print()
        print(f"📊 数据集信息:")
        print(f"   数据集名称: {dataset.dataset_name}")
        print(f"   数据集路径: {dataset_path}")
        print(f"   输出目录: {dataset_path}/Annotations_filtered_birdnest")
        print(f"   线程数: 4")
        print()
        
        # 执行一键转换处理
        print("🚀 开始执行一键转换（只保留指定类别ID）...")
        result = dataset.one_click_complete_conversion()
        
        if result:
            print()
            print("🎉 BirdNest数据集标签过滤完成!")
            print("=" * 80)
            print(f"📊 处理结果:")
            print(f"   有效文件对: {result.get('valid_pairs', 'N/A')} 个")
            print(f"   发现类别: {result.get('total_classes', 'N/A')} 个") 
            print(f"   尺寸不匹配修正: {result.get('dimension_mismatches', 0)} 个")
            print(f"   通道不匹配修正: {result.get('channel_mismatches', 0)} 个")
            print(f"   输出目录: {dataset_path}/Annotations_filtered_birdnest")
            print("=" * 80)
            
            # 显示最终保留的类别
            labels_file = os.path.join(dataset_path, "ImageSets", "Main", "labels.txt")
            if os.path.exists(labels_file):
                print()
                print("🏷️  最终保留的类别ID:")
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = f.read().strip().split('\n')
                    for i, label in enumerate(labels, 1):
                        print(f"   {i}. {label}")
                
                print()
                print(f"✅ 成功保留类别ID: {keep_class_ids}")
                print(f"📁 过滤后的XML文件保存在: {dataset_path}/Annotations_filtered_birdnest")
                
                # 验证保留的类别是否正确
                retained_ids = [int(label) for label in labels if label.isdigit()]
                expected_ids = set(keep_class_ids)
                actual_ids = set(retained_ids)
                
                if actual_ids.issubset(expected_ids):
                    print()
                    print("✅ 验证通过: 保留的类别ID符合预期")
                else:
                    unexpected = actual_ids - expected_ids
                    missing = expected_ids - actual_ids
                    if unexpected:
                        print(f"⚠️  发现意外的类别ID: {list(unexpected)}")
                    if missing:
                        print(f"⚠️  缺少预期的类别ID: {list(missing)}")
            
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
    print("开始处理BirdNest数据集...")
    print()
    
    success = main()
    
    print()
    if success:
        print("🎉 BirdNest数据集处理完成!")
    else:
        print("❌ BirdNest数据集处理失败!")
    
    print("=" * 80)