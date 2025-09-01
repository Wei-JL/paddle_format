#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别筛选使用示例
演示如何筛选掉不想要的类，或者指定想要的类
使用dataset/Fruit数据集进行测试
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from code.dataset_handler.voc_dataset import VOCDataset


def example_exclude_labels():
    """
    示例1: 排除不想要的类别
    """
    print("🚀 示例1: 排除不想要的类别")
    print("=" * 50)
    
    dataset_path = os.path.join(project_root, "dataset", "Fruit")
    
    try:
        # 初始化数据集
        dataset = VOCDataset(
            dataset_path=dataset_path,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
            max_workers=4
        )
        
        # 排除不想要的类别（例如：排除 'banana' 和 'dragon fruit'）
        exclude_labels = ['banana', 'dragon fruit']
        print(f"🚫 排除的类别: {exclude_labels}")
        
        # 执行带类别过滤的处理
        dataset.one_click_complete_conversion(exclude_labels=exclude_labels, skip_confirmation=True)
        
        print("✅ 排除类别处理完成！")
        print(f"📁 输出目录: {os.path.join(dataset_path, 'Annotations_clear')}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def example_include_labels():
    """
    示例2: 只保留指定的类别
    """
    print("\n🚀 示例2: 只保留指定的类别")
    print("=" * 50)
    
    dataset_path = os.path.join(project_root, "dataset", "Fruit")
    
    try:
        # 初始化数据集
        dataset = VOCDataset(
            dataset_path=dataset_path,
            train_ratio=0.7,
            val_ratio=0.3,
            test_ratio=0.0,
            max_workers=4
        )
        
        # 只保留指定的类别（例如：只要 'pineapple' 和 'snake fruit'）
        include_labels = ['pineapple', 'snake fruit']
        print(f"✅ 保留的类别: {include_labels}")
        
        # 执行带类别过滤的处理
        dataset.one_click_complete_conversion(include_labels=include_labels, skip_confirmation=True)
        
        print("✅ 指定类别处理完成！")
        print(f"📁 输出目录: {os.path.join(dataset_path, 'Annotations_clear')}")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def example_advanced_filtering():
    """
    示例3: 高级筛选示例
    """
    print("\n🚀 示例3: 高级筛选示例")
    print("=" * 50)
    
    dataset_path = os.path.join(project_root, "dataset", "Fruit")
    
    try:
        # 初始化数据集
        dataset = VOCDataset(
            dataset_path=dataset_path,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            max_workers=6
        )
        
        print("📊 原始数据集信息:")
        print(f"   数据集名称: {dataset.dataset_name}")
        print(f"   数据集路径: {dataset.dataset_path}")
        
        # 先查看所有可用的类别
        print("\n🔍 正在扫描数据集中的所有类别...")
        
        # 只保留一个类别进行快速测试
        include_labels = ['banana']
        print(f"🎯 本次只处理类别: {include_labels}")
        
        # 执行处理
        dataset.one_click_complete_conversion(include_labels=include_labels, skip_confirmation=True)
        
        print("✅ 高级筛选处理完成！")
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")


def main():
    """
    主函数 - 运行所有示例
    """
    print("🎯 类别筛选功能演示")
    print("使用 dataset/Fruit 数据集")
    print("=" * 60)
    
    # 提示用户选择示例
    print("\n请选择要运行的示例:")
    print("1. 排除不想要的类别")
    print("2. 只保留指定的类别") 
    print("3. 高级筛选示例")
    print("4. 运行所有示例")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            example_exclude_labels()
        elif choice == "2":
            example_include_labels()
        elif choice == "3":
            example_advanced_filtering()
        elif choice == "4":
            example_exclude_labels()
            example_include_labels()
            example_advanced_filtering()
        else:
            print("❌ 无效选择，运行默认示例...")
            example_advanced_filtering()
            
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 运行出错: {str(e)}")


if __name__ == "__main__":
    main()