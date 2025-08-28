"""
测试VOC数据集处理器的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from code.dataset_handler.voc_processor import VOCDatasetProcessor

def test_voc_processor():
    """测试VOC数据集处理器的所有功能"""
    
    try:
        # 初始化处理器
        dataset_path = "dataset/Fruit"
        processor = VOCDatasetProcessor(dataset_path)
        
        print(f"=== VOC数据集处理器测试 ===")
        print(f"数据集路径: {dataset_path}")
        
        # 获取数据集信息
        print(f"\n=== 获取数据集信息 ===")
        info = processor.get_dataset_info()
        print(f"✅ 数据集信息获取成功!")
        print(f"   XML文件数: {info['total_xml_files']}")
        print(f"   图像文件数: {info['total_image_files']}")
        print(f"   类别数量: {info['total_classes']}")
        print(f"   类别列表: {info['classes']}")
        
        # 打印数据集摘要
        print(f"\n=== 打印数据集摘要 ===")
        processor.print_summary()
        print("✅ 数据集摘要打印完成!")
        
        # 测试COCO格式转换
        print(f"\n=== 测试COCO格式转换 ===")
        coco_result = processor.convert_to_coco_format()
        
        if coco_result["success"]:
            print("✅ COCO格式转换成功!")
            print(f"   转换结果: {coco_result['message']}")
        else:
            print(f"❌ COCO格式转换失败: {coco_result['message']}")
        
        print(f"\n=== 测试完成 ===")
        print("VOC数据集处理器功能测试完毕!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_voc_processor()