#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的一键清洗+转换使用示例
使用dataset/Fruit数据集演示完整的数据处理流程
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from code.dataset_handler.voc_dataset import VOCDataset


def main():
    """
    简单的一键处理示例
    """
    print("🚀 开始简单的一键清洗+转换处理...")
    print("=" * 60)
    
    # 1. 设置数据集路径
    dataset_path = os.path.join(project_root, "dataset", "Fruit")
    print(f"📁 数据集路径: {dataset_path}")
    
    # 2. 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return
    
    try:
        # 3. 初始化VOC数据集处理器
        print("\n🔧 初始化数据集处理器...")
        dataset = VOCDataset(
            dataset_path=dataset_path,
            train_ratio=0.8,      # 训练集比例
            val_ratio=0.2,        # 验证集比例
            test_ratio=0.0,       # 测试集比例
            max_workers=4         # 线程池大小
        )
        
        print(f"✅ 数据集初始化完成: {dataset.dataset_name}")
        print(f"📊 划分比例 - 训练集: {dataset.train_ratio}, 验证集: {dataset.val_ratio}, 测试集: {dataset.test_ratio}")
        
        # 4. 执行一键完整转换
        print("\n🚀 开始一键完整转换...")
        dataset.one_click_complete_conversion(skip_confirmation=True)
        
        print("\n🎉 处理完成！")
        print("=" * 60)
        print("📋 处理结果:")
        print(f"   - 清洗后的XML文件: {os.path.join(dataset_path, 'Annotations_clear')}")
        print(f"   - 数据集划分文件: {os.path.join(dataset_path, 'ImageSets', 'Main')}")
        print(f"   - COCO格式文件: {dataset_path}")
        print("     * train_coco.json")
        print("     * val_coco.json")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()