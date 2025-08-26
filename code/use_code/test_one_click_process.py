#!/usr/bin/env python3
"""
测试一键数据集处理功能

演示如何使用VOCDataset类的一键处理功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from dataset_handler.voc_dataset import VOCDataset
from dataset_handler.one_click_functions import one_click_process_dataset
from logger_code.logger_sys import get_logger

# 获取日志记录器
logger = get_logger("test_one_click_process")

def test_one_click_process():
    """测试一键数据集处理功能"""
    
    # 数据集路径
    dataset_path = "../../dataset/Fruit"
    
    try:
        logger.info("开始测试一键数据集处理功能")
        
        # 初始化VOC数据集
        print("正在初始化VOC数据集...")
        voc_dataset = VOCDataset(dataset_path)
        
        # 显示当前数据集信息
        print("\n当前数据集信息:")
        voc_dataset.print_summary()
        
        # 执行一键处理
        # 这里可以指定要排除的类别，例如：exclude_classes=['unwanted_class']
        result = one_click_process_dataset(
            voc_dataset,
            exclude_classes=None,  # 不过滤任何类别
            auto_fix_images=True,  # 自动修正图像
            new_annotations_suffix="processed"
        )
        
        # 显示处理结果
        if result['status'] == 'success':
            print("\n✅ 一键处理成功完成！")
        elif result['status'] == 'cancelled':
            print("\n⏹️ 处理已取消")
        else:
            print(f"\n❌ 处理失败: {result.get('errors', [])}")
        
        return result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return None

def test_with_class_filtering():
    """测试带类别过滤的一键处理"""
    
    dataset_path = "../../dataset/Fruit"
    
    try:
        logger.info("开始测试带类别过滤的一键处理")
        
        # 初始化VOC数据集
        voc_dataset = VOCDataset(dataset_path)
        
        # 显示当前类别
        current_classes = voc_dataset.get_classes()
        print(f"当前类别: {sorted(current_classes)}")
        
        # 假设要排除某个类别（这里以示例为准，实际使用时根据需要修改）
        exclude_classes = []  # 可以添加要排除的类别，例如：['unwanted_fruit']
        
        if exclude_classes:
            print(f"将排除类别: {exclude_classes}")
            
            # 执行带类别过滤的一键处理
            result = one_click_process_dataset(
                voc_dataset,
                exclude_classes=exclude_classes,
                auto_fix_images=True,
                new_annotations_suffix="filtered"
            )
        else:
            print("未指定要排除的类别，执行常规处理")
            result = one_click_process_dataset(
                voc_dataset,
                exclude_classes=None,
                auto_fix_images=True
            )
        
        return result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 测试失败: {e}")
        return None

if __name__ == "__main__":
    print("VOC数据集一键处理功能测试")
    print("="*50)
    
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 基础一键处理（不过滤类别）")
    print("2. 带类别过滤的一键处理")
    
    while True:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "1":
            test_one_click_process()
            break
        elif choice == "2":
            test_with_class_filtering()
            break
        else:
            print("请输入 1 或 2")