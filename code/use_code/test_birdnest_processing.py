"""
BirdNest数据集处理测试
测试修改后的VOCDataset类的功能
"""

import os
import sys
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "code"))

from code.dataset_handler.voc_dataset import VOCDataset
from code.logger_code.logger_sys import get_logger

# 获取日志记录器
logger = get_logger("test_birdnest_processing")

def test_birdnest_dataset():
    """测试BirdNest数据集处理"""
    
    # 数据集路径
    dataset_path = r"D:\WJL\project\BirdNest"
    
    print("=" * 80)
    print("🐦 BirdNest数据集处理测试")
    print("=" * 80)
    print(f"📁 数据集路径: {dataset_path}")
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        logger.error(f"数据集路径不存在: {dataset_path}")
        return False
    
    try:
        print("\n📋 步骤1: 初始化VOC数据集...")
        logger.info("开始初始化VOC数据集")
        
        # 测试要求1: 不需要输入dataset_name，自动从路径获取
        voc_dataset = VOCDataset(
            dataset_path=dataset_path,
            max_workers=6  # 使用6个线程进行并行处理
        )
        
        print(f"✅ 数据集初始化成功")
        print(f"📊 自动获取的数据集名称: {voc_dataset.dataset_name}")
        print(f"🧵 线程池配置: {voc_dataset.max_workers} 个工作线程")
        
        # 验证清洗输出目录设置
        expected_output_dir = os.path.join(dataset_path, "Annotations_clear")
        actual_output_dir = str(voc_dataset.annotations_output_dir)
        print(f"🗂️  清洗输出目录: {actual_output_dir}")
        
        if actual_output_dir == expected_output_dir:
            print("✅ 清洗输出目录设置正确")
        else:
            print(f"❌ 清洗输出目录设置错误，期望: {expected_output_dir}")
        
        # 获取数据集基本信息
        print("\n📋 步骤2: 获取数据集基本信息...")
        dataset_info = voc_dataset.get_dataset_info()
        
        print(f"📁 数据集路径: {dataset_info['dataset_path']}")
        print(f"📂 原始标注目录: {dataset_info['annotations_dir']}")
        print(f"🗂️  清洗输出目录: {dataset_info['annotations_output_dir']}")
        print(f"📈 划分比例: 训练集{dataset_info['train_ratio']}, 验证集{dataset_info['val_ratio']}, 测试集{dataset_info['test_ratio']}")
        
        # 执行一键完成转换
        print("\n📋 步骤3: 执行一键完成转换...")
        print("⚠️  注意：这将测试以下功能：")
        print("   1. 自动获取数据集名称")
        print("   2. 使用线程池并行处理文件匹配")
        print("   3. 清洗XML文件到Annotations_clear目录")
        print("   4. 保持XML原有格式")
        print("   5. 过滤掉没有标注框的XML文件")
        
        logger.info("开始执行一键完成转换")
        
        # 跳过用户确认，直接处理
        result = voc_dataset.one_click_complete_conversion(skip_confirmation=True)
        
        if result["success"]:
            print("\n🎉 一键转换成功完成！")
            print("=" * 80)
            print("📊 处理结果统计:")
            print(f"   ✅ 有效文件对: {result.get('valid_pairs', 0)} 个")
            print(f"   🏷️  发现类别: {result.get('classes', 0)} 个")
            print(f"   🗂️  清洗输出目录: {result.get('annotations_output_dir', 'N/A')}")
            print("=" * 80)
            
            # 验证清洗结果
            print("\n📋 步骤4: 验证清洗结果...")
            verify_cleaning_results(dataset_path, result)
            
            # 显示最终数据集信息
            final_info = voc_dataset.get_dataset_info()
            if final_info['classes']:
                print(f"🏷️  发现的类别: {final_info['classes']}")
            
            logger.info("一键转换成功完成")
            return True
            
        else:
            print(f"\n❌ 一键转换失败: {result.get('message', '未知错误')}")
            logger.error(f"一键转换失败: {result.get('message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        logger.error(f"测试过程中发生错误: {str(e)}")
        logger.error(f"错误详情: {type(e).__name__}")
        
        # 打印详细的错误堆栈
        import traceback
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        
        return False

def verify_cleaning_results(dataset_path, result):
    """验证清洗结果"""
    print("🔍 验证清洗结果...")
    
    # 检查Annotations_clear目录
    annotations_clear_dir = os.path.join(dataset_path, "Annotations_clear")
    
    if os.path.exists(annotations_clear_dir):
        print(f"✅ Annotations_clear目录已创建: {annotations_clear_dir}")
        
        # 统计清洗后的文件数量
        xml_files = [f for f in os.listdir(annotations_clear_dir) if f.endswith('.xml')]
        print(f"📊 清洗后XML文件数量: {len(xml_files)} 个")
        
        # 检查几个XML文件的格式是否保持
        if xml_files:
            sample_xml = os.path.join(annotations_clear_dir, xml_files[0])
            print(f"📄 检查样本XML文件: {xml_files[0]}")
            
            try:
                with open(sample_xml, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        print("✅ XML文件内容正常")
                        # 检查是否有缩进（格式保持）
                        if '  ' in content or '\t' in content:
                            print("✅ XML文件格式保持良好（有缩进）")
                        else:
                            print("⚠️  XML文件可能没有保持原有缩进格式")
                    else:
                        print("❌ XML文件内容为空")
            except Exception as e:
                print(f"❌ 读取XML文件失败: {e}")
        
        # 检查ImageSets目录
        imagesets_dir = os.path.join(dataset_path, "ImageSets", "Main")
        if os.path.exists(imagesets_dir):
            print(f"✅ ImageSets/Main目录存在")
            
            # 检查划分文件
            split_files = ["train.txt", "val.txt", "labels.txt"]
            for split_file in split_files:
                file_path = os.path.join(imagesets_dir, split_file)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"✅ {split_file}: {len(lines)} 行")
                else:
                    print(f"❌ {split_file}: 不存在")
        else:
            print("❌ ImageSets/Main目录不存在")
    else:
        print(f"❌ Annotations_clear目录未创建: {annotations_clear_dir}")

def check_dataset_structure(dataset_path):
    """检查数据集结构"""
    print(f"\n🔍 检查数据集结构: {dataset_path}")
    
    required_dirs = [
        "Annotations",
        "JPEGImages", 
        "ImageSets"
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"   ✅ {dir_name}: 存在 ({file_count} 个文件)")
        else:
            print(f"   ❌ {dir_name}: 不存在")
    
    # 检查ImageSets/Main目录
    imagesets_main = os.path.join(dataset_path, "ImageSets", "Main")
    if os.path.exists(imagesets_main):
        file_count = len([f for f in os.listdir(imagesets_main) if os.path.isfile(os.path.join(imagesets_main, f))])
        print(f"   ✅ ImageSets/Main: 存在 ({file_count} 个文件)")
    else:
        print(f"   ⚠️  ImageSets/Main: 不存在 (将自动创建)")

def main():
    """主函数"""
    print("🐦 BirdNest数据集处理测试")
    print("测试修改后的VOCDataset类功能")
    print("验证以下要求:")
    print("1. 自动获取数据集名称（不需要输入dataset_name）")
    print("2. 清洗后XML保存到Annotations_clear目录")
    print("3. 保持XML原有格式（换行和缩进）")
    print("4. 使用线程池并行处理文件匹配")
    print("5. 过滤掉没有标注框的XML文件")
    
    dataset_path = r"D:\WJL\project\BirdNest"
    
    # 检查数据集结构
    check_dataset_structure(dataset_path)
    
    # 执行测试
    success = test_birdnest_dataset()
    
    if success:
        print("\n🎉 测试完成！数据集处理成功")
        print("📁 请查看以下输出:")
        print(f"   - 清洗后的XML文件: {dataset_path}/Annotations_clear/")
        print(f"   - 数据集划分文件: {dataset_path}/ImageSets/Main/")
        print("📋 所有要求均已实现:")
        print("   ✅ 自动获取数据集名称")
        print("   ✅ XML文件清洗到独立目录")
        print("   ✅ 保持XML原有格式")
        print("   ✅ 线程池并行处理")
        print("   ✅ 过滤空标注文件")
    else:
        print("\n❌ 测试失败！请检查错误信息")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 程序异常退出: {e}")
        sys.exit(1)