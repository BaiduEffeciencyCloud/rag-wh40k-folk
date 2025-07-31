# -*- coding: utf-8 -*-
"""
快速CoNLL生成器
基于jieba分词和现有意图分类器，快速生成CoNLL文件
"""

import logging
import os
import argparse
from typing import List, Dict, Tuple
from .tokenizer.jieba_tokenizer import JiebaTokenizer
from .classifier.intent_classifier_wrapper import IntentClassifierWrapper
from .tokenizer.conll_converter import ConllConverter

logger = logging.getLogger(__name__)


class QuickConllGenerator:
    """快速CoNLL生成器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化快速CoNLL生成器
        
        Args:
            model_path: 意图分类模型路径，如果为None则使用规则分类
        """
        self.tokenizer = JiebaTokenizer()
        self.classifier = IntentClassifierWrapper(model_path) if model_path else None
        self.converter = ConllConverter()
        
    def process_annotation_file(self, input_path: str, output_dir: str):
        """
        处理标注文件，生成CoNLL文件
        
        Args:
            input_path: 输入文件路径
            output_dir: 输出目录路径
        """
        # 读取文件
        texts = self._read_file(input_path)
        if not texts:
            logger.warning("输入文件为空")
            return
            
        # 批量预测意图
        intents = self._predict_intents(texts)
        
        # 按意图分类
        classified_texts = self._classify_by_intent(texts, intents)
        
        # 生成CoNLL文件
        self._generate_conll_files(classified_texts, output_dir)
        
        logger.info(f"处理完成，共处理{len(texts)}个查询")
        
    def _read_file(self, file_path: str) -> List[str]:
        """读取文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return []
            
    def _predict_intents(self, texts: List[str]) -> List[str]:
        """预测意图"""
        if self.classifier:
            return self.classifier.predict_batch(texts)
        else:
            # 使用规则分类
            return [self._rule_based_classify(text) for text in texts]
            
    def _rule_based_classify(self, text: str) -> str:
        """基于规则的简单分类"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["哪些", "有多少", "有哪些", "多少种", "多少个"]):
            return "list"
        elif any(word in text_lower for word in ["哪个", "哪个更高", "哪个更远", "相比", "哪个更高", "哪个更强"]):
            return "compare"
        elif any(word in text_lower for word in ["规则", "条件", "何时", "如何", "什么情况下", "什么时候"]):
            return "rule"
        else:
            return "query"
            
    def _classify_by_intent(self, texts: List[str], intents: List[str]) -> Dict[str, List[str]]:
        """按意图分类文本"""
        classified = {
            "query": [],
            "rule": [],
            "list": [],
            "compare": []
        }
        
        for text, intent in zip(texts, intents):
            if intent in classified:
                classified[intent].append(text)
            else:
                classified["query"].append(text)
                
        return classified
        
    def _generate_conll_files(self, classified_texts: Dict[str, List[str]], output_dir: str):
        """生成CoNLL文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for intent, texts in classified_texts.items():
            if not texts:
                continue
                
            # 收集所有query的(text, tokens)数据
            queries_data = []
            for text in texts:
                # 分词
                tokens = self.tokenizer.tokenize(text)
                if not tokens:
                    continue
                queries_data.append((text, tokens))
                    
            if queries_data:
                # 使用新的简单格式，不带O标签，query之间用换行符分隔
                conll_content = self.converter.convert_multiple_queries_to_simple_conll(queries_data)
                output_path = os.path.join(output_dir, f"{intent}_queries.conll")
                
                # 直接写入文件
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(conll_content)
                    logger.info(f"生成{intent}意图的CoNLL文件: {output_path} ({len(queries_data)}个样本)")
                except Exception as e:
                    logger.error(f"保存CoNLL文件失败: {e}")
                    raise
                
    def get_statistics(self, classified_texts: Dict[str, List[str]]) -> Dict[str, int]:
        """获取统计信息"""
        stats = {}
        for intent, texts in classified_texts.items():
            stats[intent] = len(texts)
        return stats


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="快速生成CoNLL文件")
    parser.add_argument("--input", "-i", required=True, 
                       help="输入文件路径 (annotation.txt)")
    parser.add_argument("--output", "-o", 
                       help="输出目录路径 (可选，默认根据mode自动生成)")
    parser.add_argument("--mode", "-m", choices=["basic", "optimize"], 
                       default="basic", help="处理模式: basic(基础jieba) 或 optimize(优化算法)")
    parser.add_argument("--model", 
                       help="意图分类模型路径 (可选，默认使用规则分类)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return 1
    
    # 根据mode自动生成输出目录
    if not args.output:
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        if args.mode == "basic":
            args.output = f"intent_training/data/auto_conll/basic/{timestamp}"
        else:  # optimize
            args.output = f"intent_training/data/auto_conll/optimize/{timestamp}"
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 创建生成器
    if args.mode == "basic":
        generator = QuickConllGenerator(args.model)
        logger.info("使用基础jieba分词模式")
    else:  # optimize
        # TODO: 这里可以添加优化算法的生成器
        generator = QuickConllGenerator(args.model)
        logger.info("使用优化算法模式 (当前使用基础模式)")
    
    try:
        # 处理文件
        generator.process_annotation_file(args.input, args.output)
        
        # 显示统计信息
        texts = generator._read_file(args.input)
        intents = generator._predict_intents(texts)
        classified_texts = generator._classify_by_intent(texts, intents)
        stats = generator.get_statistics(classified_texts)
        
        print("\n=== 处理统计 ===")
        print(f"处理模式: {args.mode}")
        for intent, count in stats.items():
            print(f"{intent}: {count}个查询")
        print(f"总计: {len(texts)}个查询")
        print(f"输出目录: {args.output}")
        
        # 显示生成的文件
        print("\n=== 生成的文件 ===")
        for file_name in os.listdir(args.output):
            file_path = os.path.join(args.output, file_name)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  {file_name} ({file_size} bytes)")
        
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 