#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签过滤功能演示
演示如何使用include_labels和exclude_labels参数进行类别筛选
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from code.dataset_handler.voc_dataset import VOCDataset

def main():
    """标签过滤功能演示"""
    
    # 数据集路径
    dataset_path = os.path.join(project_root, "dataset", "Fruit")
    
    print("🏷️  标签过滤功能演示")
    print("=" * 60)
    print(f"📁 数据集路径: {dataset_path}")
    print()
    
    # 让用户选择筛选方式
    print("请选择筛选方式:")
    print("1. 只保留指定类别 (include_labels)")
    print("2. 排除指定类别 (exclude_labels)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    # 指定要处理的类别
    target_labels = ['pineapple', 'snake fruit']
    
    if choice == '1':
        print(f"\n📋 选择方式1: 只保留 {target_labels} 类别")
        print("-" * 40)
        
        try:
            # 初始化数据集处理器 - 只保留指定类别
            dataset = VOCDataset(
                dataset_path=dataset_path,
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
                max_workers=4,
                include_labels=target_labels  # 只保留指定类别
            )
            
            print(f"✅ 数据集初始化完成 (只保留{target_labels})")
            print(f"📊 筛选条件: include_labels={target_labels}")
            print()
            
            # 执行一键转换
            print("🚀 开始处理...")
            result = dataset.one_click_complete_conversion()
            
            if result.get("success", False):
                print("✅ 处理完成!")
                print(f"📊 处理结果: 只保留了 {target_labels} 类别的数据")
                
                # 验证输出的标注文件夹
                annotations_clear_dir = os.path.join(dataset_path, "Annotations_clear")
                if os.path.exists(annotations_clear_dir):
                    xml_files = [f for f in os.listdir(annotations_clear_dir) if f.endswith('.xml')]
                    print(f"📁 清洗后的XML文件数量: {len(xml_files)} 个")
                    print(f"📂 清洗输出目录: {annotations_clear_dir}")
                
            else:
                print(f"❌ 处理失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 执行出错: {str(e)}")
    
    elif choice == '2':
        print(f"\n📋 选择方式2: 排除 {target_labels} 类别")
        print("-" * 40)
        
        try:
            # 初始化数据集处理器 - 排除指定类别
            dataset = VOCDataset(
                dataset_path=dataset_path,
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.0,
                max_workers=4,
                exclude_labels=target_labels  # 排除指定类别
            )
            
            print(f"✅ 数据集初始化完成 (排除{target_labels})")
            print(f"📊 筛选条件: exclude_labels={target_labels}")
            print()
            
            # 执行一键转换
            print("🚀 开始处理...")
            result = dataset.one_click_complete_conversion()
            
            if result.get("success", False):
                print("✅ 处理完成!")
                print(f"📊 处理结果: 保留了除 {target_labels} 外的所有类别")
                
                # 验证输出的标注文件夹
                annotations_clear_dir = os.path.join(dataset_path, "Annotations_clear")
                if os.path.exists(annotations_clear_dir):
                    xml_files = [f for f in os.listdir(annotations_clear_dir) if f.endswith('.xml')]
                    print(f"📁 清洗后的XML文件数量: {len(xml_files)} 个")
                    print(f"📂 清洗输出目录: {annotations_clear_dir}")
                
            else:
                print(f"❌ 处理失败: {result.get('message', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 执行出错: {str(e)}")
    
    else:
        print("❌ 无效选择，请输入 1 或 2")
        return
    
    print("\n" + "🎉 标签过滤功能演示完成!")
    print("=" * 60)
    print("💡 提示:")
    print("   - include_labels: 只保留指定的类别")
    print("   - exclude_labels: 排除指定的类别，保留其他所有类别")
    print("   - 两个参数不能同时使用")
    print("   - 清洗后的XML文件保存在 Annotations_clear 目录中")
    print("   - 详细日志请查看日志文件")

if __name__ == "__main__":
    main()