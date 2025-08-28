"""
测试VOC数据集类的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from code.dataset_handler.voc_dataset import VOCDataset

def test_voc_dataset():
    """测试VOC数据集类的所有功能"""
    
    try:
        # 初始化数据集
        dataset_path = "dataset/Fruit"
        dataset = VOCDataset(dataset_path)
        
        print(f"=== VOC数据集类测试 ===")
        print(f"数据集路径: {dataset_path}")
        
        # 测试数据集验证
        print(f"\n=== 测试数据集验证 ===")
        validation_result = dataset.validate_dataset()
        
        if validation_result["success"]:
            print("✅ 数据集验证成功!")
            print(f"   验证结果: {validation_result['message']}")
        else:
            print(f"❌ 数据集验证失败: {validation_result['message']}")
        
        # 测试数据集清洗
        print(f"\n=== 测试数据集清洗 ===")
        clean_result = dataset.clean_dataset()
        
        if clean_result["success"]:
            print("✅ 数据集清洗成功!")
            print(f"   清洗结果: {clean_result['message']}")
        else:
            print(f"❌ 数据集清洗失败: {clean_result['message']}")
        
        # 测试数据集分割
        print(f"\n=== 测试数据集分割 ===")
        split_result = dataset.split_dataset()
        
        if split_result["success"]:
            print("✅ 数据集分割成功!")
            print(f"   分割结果: {split_result['message']}")
        else:
            print(f"❌ 数据集分割失败: {split_result['message']}")
        
        # 测试COCO格式转换
        print(f"\n=== 测试COCO格式转换 ===")
        coco_result = dataset.convert_to_coco_format()
        
        if coco_result["success"]:
            print("✅ COCO格式转换成功!")
            print(f"   转换结果: {coco_result['message']}")
        else:
            print(f"❌ COCO格式转换失败: {coco_result['message']}")
        
        # 测试类别统计
        print(f"\n=== 测试类别统计 ===")
        count_result = dataset.count_and_sort_classes()
        
        if count_result["success"]:
            print("✅ 类别统计成功!")
            print(f"   类别总数: {count_result['total_classes']}")
            print(f"   排序后类别: {count_result['sorted_classes']}")
        else:
            print(f"❌ 类别统计失败: {count_result['message']}")
        
        # 测试类别删除功能
        print(f"\n=== 测试类别删除功能 ===")
        remove_result = dataset.remove_classes_only(
            exclude_classes=["dragon fruit"],
            new_annotations_suffix="test_removal"
        )
        
        if remove_result["success"]:
            print("✅ 类别删除成功!")
            print(f"   剩余类别: {remove_result['remaining_classes']}")
            print(f"   新目录: {remove_result['new_annotations_dir']}")
        else:
            print(f"❌ 类别删除失败: {remove_result['message']}")
        
        # 测试类别检查功能
        print(f"\n=== 测试类别检查功能 ===")
        check_result = dataset.check_classes_in_annotations(["dragon fruit"])
        
        if check_result["success"]:
            print("✅ 类别检查成功!")
            print(f"   检查结果: {check_result['message']}")
            for class_name, count in check_result['class_counts'].items():
                print(f"   {class_name}: {count} 个文件")
        else:
            print(f"❌ 类别检查失败: {check_result['message']}")
        
        print(f"\n=== 测试完成 ===")
        print("VOC数据集类功能测试完毕!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def test_one_click_complete_conversion():
    """测试一键完成转换功能（需要用户交互）"""
    print(f"\n=== 测试一键完成转换功能 ===")
    print("注意：此测试需要用户交互确认")
    
    try:
        dataset_path = "dataset/Fruit"
        dataset = VOCDataset(dataset_path)
        
        # dataset.one_click_complete_conversion()  # 取消注释以测试一键转换
        print("一键完成转换功能已准备就绪（已注释掉实际调用）")
        
    except Exception as e:
        print(f"❌ 一键转换测试失败: {str(e)}")

if __name__ == "__main__":
    test_voc_dataset()
    test_one_click_complete_conversion()