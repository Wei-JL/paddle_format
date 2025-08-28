"""
VOC与COCO格式数据集一致性检查类

用于验证VOC格式数据集与转换后的COCO格式数据集是否完全一致，
包括训练集、验证集和测试集的文件分配对应关系。
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

# 导入日志系统和全局变量
sys.path.append(str(Path(__file__).parent.parent))
from logger_code.logger_sys import get_logger
from global_var.global_cls import *

# 获取当前文件名作为日志标识
current_filename = Path(__file__).stem
logger = get_logger(current_filename)


class VOCCOCOComparison:
    """VOC与COCO格式数据集一致性检查类"""
    
    def __init__(self, dataset_path: str):
        """
        初始化比较器
        
        Args:
            dataset_path: 已处理好的数据集根目录路径（包含VOC和COCO格式文件）
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        
        # VOC格式相关路径
        self.annotations_dir = self.dataset_path / ANNOTATIONS_DIR
        self.images_dir = self.dataset_path / JPEGS_DIR
        self.imagesets_dir = self.dataset_path / IMAGESETS_DIR / MAIN_DIR
        
        # 数据集划分
        self.splits = ['train', 'val', 'test']
        
        # 存储比较结果
        self.comparison_results = {}
        self.inconsistencies = []
        
        logger.info(f"初始化VOC-COCO比较器: {self.dataset_name}")
        logger.info(f"数据集路径: {self.dataset_path.absolute()}")
        
        # 验证基本结构
        self._validate_structure()
    
    def _validate_structure(self):
        """验证数据集基本结构"""
        logger.info("验证数据集基本结构...")
        
        # 检查VOC格式必要目录
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"VOC标注目录不存在: {self.annotations_dir}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"VOC图像目录不存在: {self.images_dir}")
        
        if not self.imagesets_dir.exists():
            raise FileNotFoundError(f"VOC划分目录不存在: {self.imagesets_dir}")
        
        # 检查COCO格式文件
        coco_files = list(self.dataset_path.glob("*_coco.json"))
        if not coco_files:
            raise FileNotFoundError(f"未找到COCO格式文件: {self.dataset_path}")
        
        logger.info(f"结构验证通过 - 找到 {len(coco_files)} 个COCO文件")
    
    def _load_voc_split(self, split: str) -> Set[str]:
        """
        加载VOC格式的数据集划分
        
        Args:
            split: 划分名称 ('train', 'val', 'test')
            
        Returns:
            文件名集合（不含扩展名）
        """
        split_file = self.imagesets_dir / f"{split}.txt"
        
        if not split_file.exists():
            logger.warning(f"VOC划分文件不存在: {split_file}")
            return set()
        
        file_names = set()
        try:
            with open(split_file, 'r', encoding=DEFAULT_ENCODING) as f:
                for line in f:
                    filename = line.strip()
                    if filename:
                        # 处理可能包含路径信息的格式
                        # 例如: "JPEGImages/fruit126.png Annotations/fruit126.xml" 或 "fruit126"
                        if ' ' in filename:
                            # 如果包含空格，取第一部分（图像路径）
                            image_path = filename.split()[0]
                            # 提取文件名（不含路径和扩展名）
                            file_stem = Path(image_path).stem
                        else:
                            # 直接是文件名，移除扩展名
                            file_stem = Path(filename).stem
                        
                        if file_stem:
                            file_names.add(file_stem)
        except Exception as e:
            logger.error(f"读取VOC划分文件失败: {split_file} - {e}")
            return set()
        
        logger.debug(f"VOC {split} 划分: {len(file_names)} 个文件")
        return file_names
    
    def _load_coco_split(self, split: str) -> Set[str]:
        """
        加载COCO格式的数据集划分
        
        Args:
            split: 划分名称 ('train', 'val', 'test')
            
        Returns:
            文件名集合（含扩展名）
        """
        coco_file = self.dataset_path / f"{split}_coco.json"
        
        if not coco_file.exists():
            logger.warning(f"COCO文件不存在: {coco_file}")
            return set()
        
        try:
            with open(coco_file, 'r', encoding=DEFAULT_ENCODING) as f:
                coco_data = json.load(f)
            
            file_names = set()
            for image_info in coco_data.get('images', []):
                filename = image_info.get('file_name', '')
                if filename:
                    # 移除扩展名以便与VOC格式比较
                    file_stem = Path(filename).stem
                    file_names.add(file_stem)
            
            logger.debug(f"COCO {split} 划分: {len(file_names)} 个文件")
            return file_names
            
        except Exception as e:
            logger.error(f"读取COCO文件失败: {coco_file} - {e}")
            return set()
    
    def _compare_split(self, split: str) -> Dict:
        """
        比较单个划分的一致性
        
        Args:
            split: 划分名称
            
        Returns:
            比较结果字典
        """
        logger.info(f"比较 {split} 划分...")
        
        voc_files = self._load_voc_split(split)
        coco_files = self._load_coco_split(split)
        
        # 如果两个都为空，跳过
        if not voc_files and not coco_files:
            logger.info(f"{split} 划分为空，跳过比较")
            return {
                'split': split,
                'status': 'empty',
                'voc_count': 0,
                'coco_count': 0,
                'consistent': True,
                'only_in_voc': set(),
                'only_in_coco': set(),
                'common_files': set()
            }
        
        # 计算差异
        only_in_voc = voc_files - coco_files
        only_in_coco = coco_files - voc_files
        common_files = voc_files & coco_files
        
        is_consistent = len(only_in_voc) == 0 and len(only_in_coco) == 0
        
        result = {
            'split': split,
            'status': 'consistent' if is_consistent else 'inconsistent',
            'voc_count': len(voc_files),
            'coco_count': len(coco_files),
            'consistent': is_consistent,
            'only_in_voc': only_in_voc,
            'only_in_coco': only_in_coco,
            'common_files': common_files
        }
        
        # 记录不一致的情况
        if not is_consistent:
            if only_in_voc:
                logger.warning(f"{split} - 只在VOC中存在的文件: {len(only_in_voc)} 个")
                for filename in sorted(only_in_voc):
                    logger.warning(f"  VOC独有: {filename}")
            
            if only_in_coco:
                logger.warning(f"{split} - 只在COCO中存在的文件: {len(only_in_coco)} 个")
                for filename in sorted(only_in_coco):
                    logger.warning(f"  COCO独有: {filename}")
        else:
            logger.info(f"{split} 划分一致性检查通过: {len(common_files)} 个文件")
        
        return result
    
    def _verify_file_existence(self, split: str, file_names: Set[str]) -> Dict:
        """
        验证文件是否真实存在
        
        Args:
            split: 划分名称
            file_names: 文件名集合
            
        Returns:
            验证结果
        """
        missing_images = []
        missing_xmls = []
        
        for filename in file_names:
            # 检查图像文件
            image_found = False
            for ext in IMAGE_EXTENSIONS:
                image_file = self.images_dir / f"{filename}{ext}"
                if image_file.exists():
                    image_found = True
                    break
            
            if not image_found:
                missing_images.append(filename)
            
            # 检查XML文件
            xml_file = self.annotations_dir / f"{filename}{XML_EXTENSION}"
            if not xml_file.exists():
                missing_xmls.append(filename)
        
        return {
            'split': split,
            'total_files': len(file_names),
            'missing_images': missing_images,
            'missing_xmls': missing_xmls,
            'missing_image_count': len(missing_images),
            'missing_xml_count': len(missing_xmls)
        }
    
    def compare_all_splits(self) -> Dict:
        """
        比较所有数据集划分的一致性
        
        Returns:
            完整的比较结果
        """
        logger.info("开始比较所有数据集划分...")
        
        results = {}
        overall_consistent = True
        
        for split in self.splits:
            split_result = self._compare_split(split)
            results[split] = split_result
            
            if not split_result['consistent'] and split_result['status'] != 'empty':
                overall_consistent = False
                self.inconsistencies.append(split_result)
        
        # 验证文件存在性
        logger.info("验证文件存在性...")
        file_verification = {}
        
        for split in self.splits:
            if results[split]['status'] != 'empty':
                common_files = results[split]['common_files']
                if common_files:
                    verification = self._verify_file_existence(split, common_files)
                    file_verification[split] = verification
                    
                    if verification['missing_image_count'] > 0 or verification['missing_xml_count'] > 0:
                        overall_consistent = False
                        logger.error(f"{split} - 缺失文件: 图像 {verification['missing_image_count']} 个, XML {verification['missing_xml_count']} 个")
        
        self.comparison_results = {
            'dataset_name': self.dataset_name,
            'overall_consistent': overall_consistent,
            'split_results': results,
            'file_verification': file_verification,
            'summary': self._generate_summary(results, file_verification)
        }
        
        logger.info(f"比较完成 - 整体一致性: {'通过' if overall_consistent else '失败'}")
        return self.comparison_results
    
    def _generate_summary(self, split_results: Dict, file_verification: Dict) -> Dict:
        """生成比较摘要"""
        total_voc_files = sum(r['voc_count'] for r in split_results.values())
        total_coco_files = sum(r['coco_count'] for r in split_results.values())
        consistent_splits = sum(1 for r in split_results.values() if r['consistent'])
        
        total_missing_images = sum(v['missing_image_count'] for v in file_verification.values())
        total_missing_xmls = sum(v['missing_xml_count'] for v in file_verification.values())
        
        return {
            'total_splits_checked': len([r for r in split_results.values() if r['status'] != 'empty']),
            'consistent_splits': consistent_splits,
            'total_voc_files': total_voc_files,
            'total_coco_files': total_coco_files,
            'total_missing_images': total_missing_images,
            'total_missing_xmls': total_missing_xmls,
            'splits_status': {split: result['status'] for split, result in split_results.items()}
        }
    
    def print_comparison_report(self):
        """打印详细的比较报告"""
        if not self.comparison_results:
            logger.warning("尚未执行比较，请先调用 compare_all_splits()")
            return
        
        results = self.comparison_results
        
        print("\n" + "="*60)
        print("VOC-COCO数据集一致性检查报告")
        print("="*60)
        
        print(f"数据集名称: {results['dataset_name']}")
        print(f"整体一致性: {'✅ 通过' if results['overall_consistent'] else '❌ 失败'}")
        
        # 摘要信息
        summary = results['summary']
        print(f"\n📊 摘要统计:")
        print(f"  检查的划分数: {summary['total_splits_checked']}")
        print(f"  一致的划分数: {summary['consistent_splits']}")
        print(f"  VOC总文件数: {summary['total_voc_files']}")
        print(f"  COCO总文件数: {summary['total_coco_files']}")
        
        if summary['total_missing_images'] > 0 or summary['total_missing_xmls'] > 0:
            print(f"  缺失图像文件: {summary['total_missing_images']} 个")
            print(f"  缺失XML文件: {summary['total_missing_xmls']} 个")
        
        # 各划分详情
        print(f"\n📋 各划分详情:")
        for split, result in results['split_results'].items():
            status_icon = "✅" if result['consistent'] else "❌"
            if result['status'] == 'empty':
                status_icon = "⏭️"
            
            print(f"  {split.upper()}: {status_icon} {result['status']}")
            
            if result['status'] != 'empty':
                print(f"    VOC文件数: {result['voc_count']}")
                print(f"    COCO文件数: {result['coco_count']}")
                print(f"    共同文件数: {len(result['common_files'])}")
                
                if result['only_in_voc']:
                    print(f"    仅VOC存在: {len(result['only_in_voc'])} 个")
                
                if result['only_in_coco']:
                    print(f"    仅COCO存在: {len(result['only_in_coco'])} 个")
        
        # 文件存在性验证
        if results['file_verification']:
            print(f"\n🔍 文件存在性验证:")
            for split, verification in results['file_verification'].items():
                print(f"  {split.upper()}:")
                print(f"    总文件数: {verification['total_files']}")
                
                if verification['missing_image_count'] > 0:
                    print(f"    缺失图像: {verification['missing_image_count']} 个")
                    for img in verification['missing_images'][:5]:  # 只显示前5个
                        print(f"      - {img}")
                    if len(verification['missing_images']) > 5:
                        print(f"      ... 还有 {len(verification['missing_images']) - 5} 个")
                
                if verification['missing_xml_count'] > 0:
                    print(f"    缺失XML: {verification['missing_xml_count']} 个")
                    for xml in verification['missing_xmls'][:5]:  # 只显示前5个
                        print(f"      - {xml}")
                    if len(verification['missing_xmls']) > 5:
                        print(f"      ... 还有 {len(verification['missing_xmls']) - 5} 个")
        
        print("="*60)
    
    def get_inconsistencies(self) -> List[Dict]:
        """
        获取所有不一致的详情
        
        Returns:
            不一致项列表
        """
        return self.inconsistencies
    
    def is_consistent(self) -> bool:
        """
        检查整体是否一致
        
        Returns:
            True if consistent, False otherwise
        """
        if not self.comparison_results:
            return False
        return self.comparison_results['overall_consistent']


def main():
    """测试函数"""
    # 测试数据集路径
    dataset_path = "../../dataset/Fruit"
    
    try:
        # 创建比较器
        comparator = VOCCOCOComparison(dataset_path)
        
        # 执行比较
        results = comparator.compare_all_splits()
        
        # 打印报告
        comparator.print_comparison_report()
        
        # 检查是否一致
        if comparator.is_consistent():
            print("\n✅ VOC和COCO格式数据集完全一致！")
        else:
            print("\n❌ 发现不一致，请检查上述报告")
            inconsistencies = comparator.get_inconsistencies()
            print(f"不一致的划分数: {len(inconsistencies)}")
        
    except Exception as e:
        logger.error(f"比较过程中出错: {e}")
        print(f"❌ 比较失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()