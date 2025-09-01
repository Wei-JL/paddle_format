"""
YOLO格式转换示例
演示如何将VOC格式数据集转换为YOLO系列训练格式

支持YOLOv6-YOLOv13通用格式
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from code.dataset_handler.yolo_series_dataset import YOLOSeriesDataset


def main():
    """主函数 - YOLO格式转换示例"""
    
    print("=" * 60)
    print("YOLO格式转换示例")
    print("支持YOLOv6-YOLOv13通用格式")
    print("=" * 60)
    
    # 数据集路径
    dataset_path = "dataset/Fruit"
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        print("请确保数据集路径正确")
        return
    
    print(f"📁 数据集路径: {dataset_path}")
    
    # 用户选择转换方式
    print("\n请选择转换方式:")
    print("1. 转换所有类别")
    print("2. 只保留指定类别 (pineapple 和 snake fruit)")
    print("3. 排除指定类别 (banana)")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    try:
        # 初始化YOLO数据集处理器
        yolo_dataset = YOLOSeriesDataset(
            dataset_path=dataset_path,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0
        )
        
        if choice == '1':
            print("\n🔄 开始转换所有类别...")
            success = yolo_dataset.one_click_complete_conversion()
            
        elif choice == '2':
            print("\n🔄 开始转换，只保留 pineapple 和 snake fruit...")
            target_labels = ['pineapple', 'snake fruit']
            success = yolo_dataset.one_click_complete_conversion(
                include_labels=target_labels
            )
            
        elif choice == '3':
            print("\n🔄 开始转换，排除 banana...")
            exclude_labels = ['banana']
            success = yolo_dataset.one_click_complete_conversion(
                exclude_labels=exclude_labels
            )
            
        else:
            print("❌ 无效选择")
            return
        
        if success:
            print("\n✅ YOLO格式转换完成！")
            print(f"📂 输出目录: {os.path.join(dataset_path, 'yolo_format')}")
            print(f"📄 配置文件: {os.path.join(dataset_path, 'yolo_format', 'Fruit.yaml')}")
            
            # 显示使用说明
            print("\n" + "=" * 60)
            print("🚀 使用说明:")
            print("=" * 60)
            print("1. 训练图片位于: yolo_format/images/train/")
            print("2. 验证图片位于: yolo_format/images/val/")
            print("3. 训练标签位于: yolo_format/labels/train/")
            print("4. 验证标签位于: yolo_format/labels/val/")
            print("5. 配置文件: yolo_format/Fruit.yaml")
            print("\n📖 YOLOv13训练命令示例:")
            print(f"python train.py --data {os.path.join(dataset_path, 'yolo_format', 'Fruit.yaml')} --epochs 100")
            
        else:
            print("❌ 转换失败，请检查日志")
            
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()