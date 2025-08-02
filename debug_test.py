#!/usr/bin/env python3
"""
调试_find_best_keyword_match方法
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qProcessor.adaptive import AdaptiveProcessor

def debug_keyword_match(query):
    """调试关键词匹配"""
    print("开始调试关键词匹配...")
    
    # 初始化处理器
    processor = AdaptiveProcessor()
    
    # 使用传入的查询
    print(f"测试查询: {query}")
    
    # 测试_find_best_keyword_match
    result = processor._find_best_keyword_match(query)
    print(f"_find_best_keyword_match结果: {result}")
    
    # 获取ML模型预测
    features = processor._extract_features(query)
    intent_prediction = processor.intent_classifier.predict(features)[0]
    intent_probabilities = processor.intent_classifier.predict_proba(features)[0]
    max_confidence = intent_probabilities.max()
    
    print(f"\n=== ML模型预测 ===")
    print(f"原始预测意图(predict): {intent_prediction}")
    print(f"原始预测意图(predict_proba): {intent_probabilities}")
    print(f"转换为小写: {intent_prediction.lower()}")
    print(f"最大置信度: {max_confidence:.4f}")
    
    # 显示概率分布
    print(f"\n=== 概率分布 ===")
    # 使用正确的模型类别顺序
    intent_names = ['Compare', 'List', 'Query', 'Rule']
    for i, intent in enumerate(intent_names):
        prob = intent_probabilities[i] if i < len(intent_probabilities) else 0
        print(f"{intent.lower()}: {prob:.4f}")
    
    # 找到最大概率的索引
    max_prob_idx = intent_probabilities.argmax()
    max_prob_intent = intent_names[max_prob_idx].lower() if max_prob_idx < len(intent_names) else 'unknown'
    print(f"最大概率意图: {max_prob_intent} ({intent_probabilities[max_prob_idx]:.4f})")
    
    # 检查模型类别标签
    try:
        if hasattr(processor.intent_classifier, 'classes_'):
            print(f"\n=== 模型类别标签 ===")
            for i, class_name in enumerate(processor.intent_classifier.classes_):
                print(f"索引 {i}: {class_name}")
    except:
        print("无法获取模型类别标签")
    
    # 测试关键词匹配检查
    print(f"\n=== 关键词匹配检查 ===")
    for intent_type in ['compare', 'list', 'query', 'rule']:
        keyword_match = processor._check_keyword_match(query, intent_type)
        print(f"{intent_type} 关键词匹配: {keyword_match}")
    
    # 测试_classify_intent_with_gradient
    intent_result = processor._classify_intent_with_gradient(query)
    print(f"\n_classify_intent_with_gradient结果: {intent_result.intent_type}")
    print(f"置信度: {intent_result.confidence}")
    
    # 检查WH40K词汇匹配
    contains_wh40k = processor._contains_wh40k_vocabulary(query)
    print(f"包含WH40K词汇: {contains_wh40k}")
    
    # 手动验证概率最大值逻辑
    features = processor._extract_features(query)
    intent_probabilities = processor.intent_classifier.predict_proba(features)[0]
    max_prob_idx = intent_probabilities.argmax()
    intent_prediction = processor.intent_classifier.classes_[max_prob_idx]
    print(f"\n=== 手动验证概率最大值逻辑 ===")
    print(f"max_prob_idx: {max_prob_idx}")
    print(f"intent_prediction: {intent_prediction}")
    print(f"intent_prediction.lower(): {intent_prediction.lower()}")
    print(f"classes_: {processor.intent_classifier.classes_}")
    
    # 显示所有意图的概率分布
    print("\n=== 所有意图的概率分布 ===")
    # 使用正确的模型类别顺序
    intent_names = ['Compare', 'List', 'Query', 'Rule']
    for i, intent in enumerate(intent_names):
        prob = intent_result.probabilities[i] if i < len(intent_result.probabilities) else 0
        print(f"{intent.lower()}: {prob:.4f}")
    
    # 显示最大概率的意图
    max_prob_idx = intent_result.probabilities.argmax()
    max_prob_intent = intent_names[max_prob_idx].lower() if max_prob_idx < len(intent_names) else 'unknown'
    print(f"\n最大概率意图: {max_prob_intent} ({intent_result.probabilities[max_prob_idx]:.4f})")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='调试查询意图分类')
    parser.add_argument('--query', type=str, required=True, help='要测试的查询')
    
    args = parser.parse_args()
    
    debug_keyword_match(args.query)

if __name__ == "__main__":
    main() 