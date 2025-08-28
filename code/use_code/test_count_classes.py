"""
测试类别统计功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from code.dataset_handler.voc_dataset import VOCDataset

def test_count_classes():
    """测试类别统计功能"""
    
    try:
        # 初始化数据集
        dataset_path = "dataset/Fruit"
        dataset = VOCDataset(dataset_path)
        
        print(f"=== 类别统计功能测试 ===")
        print(f"数据集路径: {dataset_path}")
        
        # 测试类别统计
        print(f"\n=== 统计所有类别 ===")
        count_result = dataset.count_and_sort_classes()
        
        if count_result["success"]:
            print("✅ 类别统计成功!")
            print(f"   类别总数: {count_result['total_classes']}")
            print(f"   排序后类别: {count_result['sorted_classes']}")
            
            # 显示详细统计信息
            if 'class_counts' in count_result:
                print("\n详细类别统计:")
                for class_name, count in count_result['class_counts'].items():
                    print(f"   {class_name}: {count} 个标注")
        else:
            print(f"❌ 类别统计失败: {count_result['message']}")
        
        # 测试类别检查功能
        print(f"\n=== 检查特定类别 ===")
        classes_to_check = ["apple", "banana", "dragon fruit"]
        check_result = dataset.check_classes_in_annotations(classes_to_check)
        
        if check_result["success"]:
            print("✅ 类别检查成功!")
            print(f"   检查结果: {check_result['message']}")
            for class_name, count in check_result['class_counts'].items():
                print(f"   {class_name}: {count} 个文件包含此类别")
        else:
            print(f"❌ 类别检查失败: {check_result['message']}")
        
        # 测试检查过滤后的文件夹
        print(f"\n=== 检查过滤后的文件夹 ===")
        filtered_path = os.path.join(dataset_path, "Annotations_test_filtered")
        if os.path.exists(filtered_path):
            check_filtered = dataset.check_classes_in_annotations(
                classes_to_check, 
                annotations_path=filtered_path
            )
            
            if check_filtered["success"]:
                print("✅ 过滤文件夹检查成功!")
                print(f"   检查结果: {check_filtered['message']}")
                for class_name, count in check_filtered['class_counts'].items():
                    print(f"   {class_name}: {count} 个文件包含此类别")
            else:
                print(f"❌ 过滤文件夹检查失败: {check_filtered['message']}")
        else:
            print("过滤文件夹不存在，跳过检查")
        
        print(f"\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_count_classes()