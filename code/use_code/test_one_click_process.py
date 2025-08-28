"""
测试一键处理功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from code.dataset_handler.voc_dataset import VOCDataset

def test_one_click_process():
    """测试一键处理功能"""
    
    try:
        # 初始化数据集
        dataset_path = "dataset/Fruit"
        dataset = VOCDataset(dataset_path)
        
        print(f"=== 一键处理功能测试 ===")
        print(f"数据集路径: {dataset_path}")
        
        # 测试一键处理功能
        print(f"\n=== 执行一键处理 ===")
        print("注意：此功能会执行完整的数据集处理流程")
        
        result = dataset.one_click_process()
        
        if result["success"]:
            print("✅ 一键处理成功!")
            print(f"   处理结果: {result['message']}")
        else:
            print(f"❌ 一键处理失败: {result['message']}")
        
        print(f"\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def test_one_click_complete_conversion():
    """测试一键完成转换功能（带用户确认）"""
    
    try:
        # 初始化数据集
        dataset_path = "dataset/Fruit"
        dataset = VOCDataset(dataset_path)
        
        print(f"\n=== 一键完成转换功能测试 ===")
        print(f"数据集路径: {dataset_path}")
        
        # 测试一键完成转换功能
        print(f"\n=== 执行一键完成转换 ===")
        print("注意：此功能需要用户确认并会执行完整的转换流程")
        
        result = dataset.one_click_complete_conversion()
        
        if result["success"]:
            print("✅ 一键完成转换成功!")
            print(f"   转换结果: {result['message']}")
        else:
            print(f"❌ 一键完成转换失败: {result['message']}")
        
        print(f"\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("选择测试功能:")
    print("1. 一键处理功能 (one_click_process)")
    print("2. 一键完成转换功能 (one_click_complete_conversion)")
    print("3. 两个功能都测试")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_one_click_process()
    elif choice == "2":
        test_one_click_complete_conversion()
    elif choice == "3":
        test_one_click_process()
        test_one_click_complete_conversion()
    else:
        print("无效选择，退出测试")