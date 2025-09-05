"""
vocab_mgr模块主程序

负责协调专业术语的抽取、处理和保存
"""

import os
import sys
import json
import logging
import argparse
import config
from datetime import datetime
from typing import Dict, Any, List
from dataupload.data_vector_v3 import SemanticDocumentChunker
  
# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vocab_mgr.vocab_extractor import VocabExtractor
from vocab_mgr.vocab_mgr import VocabMgr
from vocab_mgr.vocab_updater import VocabUp
from vocab_mgr.vocab_loader import VocabLoad

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VocabMgr - 专业术语词典管理器')
    parser.add_argument('doc_path', help='文档文件路径')
    parser.add_argument('--record-change', action='store_true', help='记录本次抽取结果到默认路径')
    parser.add_argument('--update-dict', action='store_true', help='更新词典文件')
    parser.add_argument('--overwrite', '--ow', action='store_true', help='强制覆盖现有词典记录')
    parser.add_argument('--export-syn', action='store_true', help='仅导出同义词文件（不导出术语词典）')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()


def load_document(doc_path: str) -> str:
    """加载文档内容"""
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"文档文件不存在: {doc_path}")
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"成功加载文档: {doc_path} ({len(text)} 字符)")
        return text
    except Exception as e:
        raise Exception(f"加载文档失败: {e}")


def chunk_document(text: str) -> List[Dict[str, Any]]:
    """使用SemanticDocumentChunker进行文档切片"""
    try:
      
        chunker = SemanticDocumentChunker()
        logger.info("开始文档切片...")
        
        chunks = chunker.chunk_text(text)
        logger.info(f"文档切片完成，共生成 {len(chunks)} 个切片")
        
        return chunks
        
    except Exception as e:
        logger.error(f"文档切片失败: {e}")
        raise


def extract_terms_from_chunks(chunks: List[Dict[str, Any]], vocab_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """从chunks中抽取专业术语"""
    try:
        extractor = VocabExtractor(vocab_path=vocab_path)
        logger.info("开始抽取专业术语...")
        
        extracted_terms = extractor.extract_from_chunks(chunks)
        
        # 统计结果
        total_terms = sum(len(terms) for terms in extracted_terms.values())
        logger.info(f"术语抽取完成，共抽取 {total_terms} 个术语")
        
        for category, terms in extracted_terms.items():
            if terms:
                logger.info(f"  {category}: {len(terms)} 个")
        
        return extracted_terms
        
    except Exception as e:
        logger.error(f"术语抽取失败: {e}")
        raise


def save_terms_to_vocabulary(extracted_terms: Dict[str, List[Dict[str, Any]]], vocab_path: str, overwrite: bool = False) -> bool:
    """将抽取的术语保存到词典"""
    try:
        vocab_mgr = VocabMgr(vocab_path=vocab_path)
        
        # 将抽取结果转换为词典格式
        vocabulary = {}
        for category, terms in extracted_terms.items():
            if not terms:
                continue
            
            category_vocab = vocab_mgr.get_vocab().get(category, {})
            
            # 添加新术语
            added_count = 0
            updated_count = 0
            for term_info in terms:
                term = term_info.get("term", "")
                original_text = term_info.get("original", term)
                synonyms = term_info.get("synonyms", [])
                unit_category = term_info.get("category", None)  # 获取单位类别信息
                
                if term and term not in category_vocab:
                    # 创建术语条目
                    term_entry = create_term_entry(term, category, original_text, synonyms, unit_category)
                    category_vocab[term] = term_entry
                    added_count += 1
                elif term and term in category_vocab:
                    # 无论是否覆盖，都要增加频率计数
                    if "frequency" in category_vocab[term]:
                        category_vocab[term]["frequency"] += 1
                    
                    if overwrite:
                        # 覆盖模式：强制更新现有记录
                        term_entry = create_term_entry(term, category, original_text, synonyms, unit_category)
                        category_vocab[term] = term_entry
                        updated_count += 1
            
            if added_count > 0 or updated_count > 0:
                vocabulary[category] = category_vocab
                if added_count > 0:
                    logger.info(f"  {category}: 新增 {added_count} 个术语")
                if updated_count > 0:
                    logger.info(f"  {category}: 覆盖 {updated_count} 个术语")
        
        # 保存词典
        if vocabulary:
            success = vocab_mgr.save_vocabulary(vocabulary)
            if success:
                logger.info("成功保存术语到词典")
                return True
            else:
                logger.error("保存词典失败")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"保存术语到词典失败: {e}")
        return False


def create_term_entry(term: str, category: str, original_text: str, synonyms: List[str], unit_category: str = None) -> Dict[str, Any]:
    """创建术语条目"""
    entry = {
        "type": category.upper(),
        "frequency": 1
    }
    
    # 添加同义词
    if synonyms:
        entry["synonyms"] = synonyms
    
    # 根据分类添加特定字段
    if category == "mechanics":
        entry["category"] = "游戏机制"
    elif category == "units":
        # 如果有单位类别信息，使用它；否则使用默认值
        if unit_category:
            entry["category"] = unit_category
        else:
            entry["category"] = "单位"
    elif category == "detachments":
        entry["category"] = "分队"
    elif category == "factions":
        entry["category"] = "派系"
    
    return entry


def display_results(results: Dict[str, List[Dict[str, Any]]], stats: Dict[str, Any]):
    """显示抽取结果"""
    print("\n" + "="*50)
    print("专业术语抽取结果")
    print("="*50)
    
    total_terms = sum(len(terms) for terms in results.values())
    print(f"总计抽取术语: {total_terms} 个")
    print(f"处理时间: {stats.get('processing_time', 'N/A')}")
    print(f"文档大小: {stats.get('doc_size', 'N/A')} 字符")
    print(f"切片数量: {stats.get('chunk_count', 'N/A')}")
    
    print("\n分类统计:")
    for category, terms in results.items():
        if terms:
            print(f"  {category}: {len(terms)} 个")
            # 显示前3个术语
            for i, term_info in enumerate(terms[:3]):
                term = term_info.get("term", "")
                synonyms = term_info.get("synonyms", [])
                if synonyms:
                    print(f"    - {term} (同义词: {', '.join(synonyms)})")
                else:
                    print(f"    - {term}")
            if len(terms) > 3:
                print(f"    ... 还有 {len(terms) - 3} 个")


def save_results_to_file(results: Dict[str, List[Dict[str, Any]]], output_path: str):
    """保存结果到文件"""
    try:
        # 转换为可序列化的格式
        serializable_results = {}
        for category, terms in results.items():
            serializable_results[category] = []
            for term_info in terms:
                serializable_results[category].append({
                    "term": term_info.get("term", ""),
                    "original": term_info.get("original", ""),
                    "synonyms": term_info.get("synonyms", [])
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")
        
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置详细输出
        # 移除 verbose 参数，保持默认日志级别
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 1. 加载文档
        logger.info("开始处理文档...")
        text = load_document(args.doc_path)
        
        # 2. 文档切片
        chunks = chunk_document(text)
        
        # 3. 抽取术语
        # 词典路径使用配置常量
 
        extracted_terms = extract_terms_from_chunks(chunks, config.VOCAB_EXPORT_DIR)
        
        # 4. 保存到词典（如果指定）
        if args.update_dict:
            success = save_terms_to_vocabulary(extracted_terms, config.VOCAB_EXPORT_DIR, args.overwrite)
            if not success:
                logger.warning("词典更新失败，但继续执行")
        
        # 5. 显示结果
        stats = {
            "processing_time": str(datetime.now() - start_time),
            "doc_size": len(text),
            "chunk_count": len(chunks)
        }
        display_results(extracted_terms, stats)
        
        # 6. 记录抽取结果（如果指定）
        if getattr(args, 'record_change', False):
            # 默认保存到 dict/record/YYMMDDHHMMSS_vocab_extraction_results.json
            ts = datetime.now().strftime('%y%m%d%H%M%S')
            output_dir = os.path.join('dict', 'record')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{ts}_vocab_extraction_results.json")
            save_results_to_file(extracted_terms, output_path)
        
        # 7. 仅导出同义词（如指定）
        if getattr(args, 'export_syn', False):
            vocab_loader = VocabLoad(vocab_path=config.VOCAB_EXPORT_DIR)
            syn_result = vocab_loader.export_synonyms_only(None)
            if syn_result["success"]:
                logger.info("同义词导出成功")
                logger.info(f"同义词文件: {syn_result['synonyms_file']}")
                logger.info(f"导出统计: {syn_result['stats']}")
            else:
                logger.error(f"同义词导出失败: {syn_result['message']}")
        
        logger.info("处理完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 