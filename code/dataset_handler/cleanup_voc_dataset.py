#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理VOCDataset类中的重复功能

移除已经迁移到VOCDatasetProcessor的功能，保持代码整洁
"""

def cleanup_voc_dataset():
    """清理VOCDataset类，移除重复功能"""
    
    print("=== VOCDataset类清理建议 ===")
    print("\n需要从VOCDataset类中移除的功能：")
    print("1. one_click_process() - 已迁移到VOCDatasetProcessor")
    print("2. count_and_sort_classes() - 已迁移到VOCDatasetProcessor") 
    print("3. filter_classes_and_regenerate() - 与remove_classes_only()功能重复")
    
    print("\n保留在VOCDataset类中的核心功能：")
    print("1. 数据集初始化和验证")
    print("2. 文件匹配和基础清洗")
    print("3. 数据集划分（train/val/test）")
    print("4. 类别提取和管理")
    print("5. convert_to_coco_format() - 作为基础转换功能保留")
    print("6. 图像尺寸检查和修复")
    print("7. 基础统计信息")
    
    print("\n建议的使用模式：")
    print("```python")
    print("# 基础数据集管理和清洗")
    print("voc_dataset = VOCDataset(dataset_path)")
    print("voc_dataset.check_and_fix_image_dimensions()")
    print("")
    print("# 高级处理操作")
    print("processor = VOCDatasetProcessor(dataset_path)")
    print("processor.one_click_process()")
    print("processor.remove_classes_only(['unwanted_class'])")
    print("processor.count_and_sort_classes()")
    print("```")
    
    print("\n这样的拆分实现了：")
    print("✅ 单一职责原则")
    print("✅ 降低代码耦合性") 
    print("✅ 提高可维护性")
    print("✅ 更好的可测试性")

if __name__ == "__main__":
    cleanup_voc_dataset()