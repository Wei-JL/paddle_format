"""
测试标签筛选功能
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from code.dataset_handler.voc_dataset_optimized import VOCDataset
from code.logger_code.my_logger import logger


def test_label_filtering():
    """测试标签筛选功能"""
    
    print("🏷️  标签筛选功能测试")
    print("="*80)
    
    # 数据集路径
    dataset_path = r"D:\WJL\project\BirdNest"
    
    # 要保留的标签类别
    keep_labels = [100, 101, 102, 113, 140, 150, 153, 154, 155, 200, 201, 202, 203, 204, 205, 220, 221, 240]
    keep_labels_str = [str(label) for label in keep_labels]
    
    print(f"📁 数据集路径: {dataset_path}")
    print(f"🏷️  要保留的标签: {keep_labels}")
    print(f"📊 标签数量: {len(keep_labels)} 个")
    
    try:
        # 初始化VOC数据集处理器
        print("\n📋 步骤1: 初始化VOC数据集处理器...")
        voc_dataset = VOCDataset(
            dataset_path=dataset_path,
            annotations_folder_name="Annotations"  # 使用原始标注文件夹
        )
        
        print("✅ 初始化完成")
        print(f"📊 数据集名称: {voc_dataset.dataset_name}")
        print(f"📂 标注文件夹: {voc_dataset.annotations_folder_name}")
        print(f"🧵 线程池配置: {voc_dataset.max_workers} 个工作线程")
        
        # 获取数据集基本信息
        print("\n📋 步骤2: 获取数据集基本信息...")
        dataset_info = voc_dataset.get_dataset_info()
        print(f"📁 数据集路径: {dataset_info['dataset_path']}")
        print(f"📂 原始标注目录: {dataset_info['annotations_dir']}")
        print(f"🗂️  清洗输出目录: {dataset_info['clean_output_dir']}")
        
        # 执行标签筛选
        print("\n📋 步骤3: 执行标签筛选...")
        print("⚠️  注意：这将筛选标签并保存到新的输出目录")
        print(f"🏷️  保留标签: {keep_labels}")
        
        # 确认是否继续
        user_input = input("\n是否继续执行标签筛选？输入 Y 继续，N 取消: ").strip()
        if user_input.upper() != 'Y':
            print("❌ 操作已取消")
            return
        
        # 执行筛选（保留模式）
        logger.info("开始执行标签筛选")
        result = voc_dataset.filter_specified_labels(keep_labels_str, keep_mode=True)
        
        if result["success"]:
            print("\n🎉 标签筛选完成！")
            print("="*80)
            print("📊 处理结果统计:")
            print(f"   📁 处理文件数: {result['processed_files']} 个")
            print(f"   ❌ 移除标注框: {result['removed_annotations']} 个")
            print(f"   ✅ 保留标注框: {result['kept_annotations']} 个")
            print(f"   📄 空文件数: {result['empty_files']} 个")
            print(f"   🗂️  输出目录: {result['output_dir']}")
            print("="*80)
            
            # 提示后续操作
            print("\n💡 后续操作建议:")
            print("1. 检查输出目录中的筛选结果")
            print("2. 使用筛选后的标注文件进行数据集划分")
            print("3. 转换为COCO格式进行训练")
            
        else:
            print(f"\n❌ 标签筛选失败: {result.get('error', '未知错误')}")
            
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def test_remove_mode():
    """测试移除模式"""
    
    print("\n🗑️  测试移除模式")
    print("="*50)
    
    dataset_path = r"D:\WJL\project\BirdNest"
    
    # 要移除的标签（示例）
    remove_labels = ["999", "invalid", "test"]  # 假设的无效标签
    
    try:
        voc_dataset = VOCDataset(dataset_path=dataset_path)
        
        print(f"🗑️  要移除的标签: {remove_labels}")
        
        # 执行筛选（移除模式）
        result = voc_dataset.filter_specified_labels(remove_labels, keep_mode=False)
        
        if result["success"]:
            print("✅ 移除模式测试完成")
            print(f"📊 移除标注框: {result['removed_annotations']} 个")
        else:
            print(f"❌ 移除模式测试失败: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ 移除模式测试错误: {e}")


if __name__ == "__main__":
    print("🧪 VOCDataset 标签筛选功能测试")
    print("="*80)
    
    # 测试保留模式
    test_label_filtering()
    
    # 测试移除模式（可选）
    # test_remove_mode()
    
    print("\n✅ 测试完成！")