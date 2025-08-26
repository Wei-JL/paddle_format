"""
VOC数据集一键处理功能扩展

为VOCDataset类添加一键完成转换并修复所有问题的功能
"""

from typing import List, Dict
from pathlib import Path
import sys

# 导入日志系统
sys.path.append(str(Path(__file__).parent.parent))
from logger_code.logger_sys import get_logger

# 获取日志记录器
logger = get_logger("one_click_functions")


def one_click_process_dataset(voc_dataset, exclude_classes: List[str] = None, 
                             auto_fix_images: bool = True,
                             new_annotations_suffix: str = "filtered") -> Dict:
    """
    一键完成数据集转换并修复所有问题
    
    包含以下步骤：
    1. 提醒用户备份数据集
    2. 过滤指定类别（如果提供）
    3. 修正图像和XML文件
    4. 重新划分VOC数据集
    5. 转换为COCO格式
    
    Args:
        voc_dataset: VOCDataset实例
        exclude_classes: 要剔除的类别列表，None表示不过滤
        auto_fix_images: 是否自动修正图像和XML文件
        new_annotations_suffix: 过滤后新标注目录的后缀名
        
    Returns:
        Dict: 处理结果统计信息
    """
    logger.info("=== 开始一键数据集处理 ===")
    
    # 步骤1：提醒用户备份数据集
    print("\n" + "="*60)
    print("⚠️  重要提醒：数据集处理警告")
    print("="*60)
    print("此操作将对您的数据集进行以下修改：")
    print("1. 删除空标注文件")
    if exclude_classes:
        print(f"2. 过滤指定类别: {exclude_classes}")
    if auto_fix_images:
        print("3. 修正图像尺寸和通道数")
        print("4. 更新XML文件中的尺寸信息")
    print("5. 重新划分数据集")
    print("6. 生成COCO格式文件")
    print("\n⚠️  强烈建议在处理前备份您的原始数据集！")
    print("="*60)
    
    # 获取用户确认
    while True:
        user_input = input("请确认您已备份数据集，输入 'Y' 继续处理，'N' 取消处理: ").strip().upper()
        if user_input == 'Y':
            logger.info("用户确认继续处理")
            break
        elif user_input == 'N':
            logger.info("用户取消处理")
            print("处理已取消")
            return {'status': 'cancelled', 'message': '用户取消处理'}
        else:
            print("请输入 'Y' 或 'N'")
    
    # 初始化结果统计
    result = {
        'status': 'success',
        'steps_completed': [],
        'statistics': {},
        'generated_files': {},
        'errors': []
    }
    
    try:
        # 步骤2：类别过滤（如果需要）
        if exclude_classes:
            logger.info(f"步骤2: 开始类别过滤 - 排除类别: {exclude_classes}")
            filter_stats = voc_dataset.filter_classes_and_regenerate(exclude_classes, new_annotations_suffix)
            result['steps_completed'].append('class_filtering')
            result['statistics']['class_filtering'] = filter_stats
            logger.info("类别过滤完成")
        else:
            logger.info("步骤2: 跳过类别过滤")
        
        # 步骤3：图像和XML修正
        if auto_fix_images:
            logger.info("步骤3: 开始图像和XML文件修正")
            fix_stats = voc_dataset.check_and_fix_image_dimensions(auto_fix=True)
            result['steps_completed'].append('image_fixing')
            result['statistics']['image_fixing'] = fix_stats
            logger.info("图像和XML文件修正完成")
        else:
            logger.info("步骤3: 跳过图像修正")
        
        # 步骤4：重新提取类别和划分数据集（已在前面步骤中完成）
        logger.info("步骤4: 数据集划分已完成")
        result['steps_completed'].append('dataset_splitting')
        
        # 步骤5：转换为COCO格式
        logger.info("步骤5: 开始转换为COCO格式")
        coco_files = voc_dataset.convert_to_coco_format()
        result['steps_completed'].append('coco_conversion')
        result['generated_files']['coco'] = coco_files
        logger.info("COCO格式转换完成")
        
        # 生成最终统计信息
        final_info = voc_dataset.get_dataset_info()
        result['statistics']['final_dataset'] = final_info
        
        logger.info("=== 一键数据集处理完成 ===")
        
        # 打印处理结果摘要
        _print_process_summary(result)
        
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        logger.error(error_msg)
        result['status'] = 'error'
        result['errors'].append(error_msg)
        print(f"\n❌ {error_msg}")
    
    return result


def _print_process_summary(result: Dict):
    """打印处理结果摘要"""
    print("\n" + "="*60)
    print("📊 数据集处理完成摘要")
    print("="*60)
    
    # 显示完成的步骤
    print("✅ 已完成的步骤:")
    step_names = {
        'class_filtering': '类别过滤',
        'image_fixing': '图像和XML修正',
        'dataset_splitting': '数据集划分',
        'coco_conversion': 'COCO格式转换'
    }
    for step in result['steps_completed']:
        print(f"   • {step_names.get(step, step)}")
    
    # 显示统计信息
    if 'final_dataset' in result['statistics']:
        final_stats = result['statistics']['final_dataset']
        print(f"\n📈 最终数据集统计:")
        print(f"   • 有效文件对: {final_stats['total_valid_pairs']} 个")
        print(f"   • 类别数量: {final_stats['total_classes']} 个")
        print(f"   • 类别列表: {', '.join(final_stats['classes'])}")
    
    # 显示生成的文件
    if 'coco' in result['generated_files']:
        print(f"\n📁 生成的COCO文件:")
        for split, file_path in result['generated_files']['coco'].items():
            print(f"   • {split}: {file_path}")
    
    # 显示错误信息
    if result['errors']:
        print(f"\n❌ 错误信息:")
        for error in result['errors']:
            print(f"   • {error}")
    
    print("="*60)