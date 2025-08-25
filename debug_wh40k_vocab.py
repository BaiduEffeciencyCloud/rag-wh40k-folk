#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：分析wh40k_vocabulary加载过程
找出为什么"死神军毒灾飞艇"没有被正确加载
"""

import json
import os
from pathlib import Path
from typing import Set, Dict, List


def debug_wh40k_vocabulary_loading(vocab_dir: str):
    """调试wh40k_vocabulary加载过程"""
    vocab_dir_path = Path(vocab_dir)
    if not vocab_dir_path.exists():
        print(f"❌ 目录不存在: {vocab_dir}")
        return
    
    print("🔍 开始调试wh40k_vocabulary加载过程...")
    print("="*60)
    
    json_files = list(vocab_dir_path.glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件:")
    for f in json_files:
        print(f"  - {f.name}")
    
    print("\n" + "="*60)
    
    all_words = set()
    all_main_words = set()
    all_synonyms = set()
    
    for json_file in json_files:
        print(f"\n📁 处理文件: {json_file.name}")
        print("-" * 40)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_main_words = set()
            file_synonyms = set()
            
            print(f"文件包含 {len(data)} 个主词汇")
            
            # 详细分析每个词汇
            for main_word, word_info in data.items():
                file_main_words.add(main_word)
                all_main_words.add(main_word)
                
                # 检查词汇信息结构
                if isinstance(word_info, dict):
                    print(f"  ✓ {main_word}: {type(word_info)}")
                    
                    # 检查同义词
                    if "synonyms" in word_info:
                        synonyms = word_info["synonyms"]
                        if isinstance(synonyms, list):
                            print(f"    - 同义词: {synonyms}")
                            file_synonyms.update(synonyms)
                            all_synonyms.update(synonyms)
                        else:
                            print(f"    ⚠️  同义词格式错误: {type(synonyms)}")
                    else:
                        print(f"    - 无同义词")
                else:
                    print(f"  ⚠️  {main_word}: 格式错误 {type(word_info)}")
                
                # 检查是否包含目标词汇
                if "死神军毒灾飞艇" in main_word:
                    print(f"    🎯 找到目标词汇: {main_word}")
                if "毒灾" in main_word:
                    print(f"    🎯 找到包含'毒灾'的词汇: {main_word}")
                if "飞艇" in main_word:
                    print(f"    🎯 找到包含'飞艇'的词汇: {main_word}")
            
            # 检查同义词中是否包含目标词汇
            for synonym in file_synonyms:
                if "死神军毒灾飞艇" in synonym:
                    print(f"    🎯 在同义词中找到目标词汇: {synonym}")
                if "毒灾" in synonym:
                    print(f"    🎯 在同义词中找到包含'毒灾'的词汇: {synonym}")
                if "飞艇" in synonym:
                    print(f"    🎯 在同义词中找到包含'飞艇'的词汇: {synonym}")
            
            print(f"  📊 本文件统计:")
            print(f"    - 主词汇: {len(file_main_words)}")
            print(f"    - 同义词: {len(file_synonyms)}")
            
        except Exception as e:
            print(f"❌ 加载 {json_file.name} 失败: {e}")
    
    print("\n" + "="*60)
    print("📊 总体统计:")
    print(f"主词汇总数: {len(all_main_words)}")
    print(f"同义词总数: {len(all_synonyms)}")
    print(f"总词汇数: {len(all_main_words) + len(all_synonyms)}")
    
    # 检查目标词汇
    print("\n" + "="*60)
    print("🎯 目标词汇检查:")
    
    target_word = "死神军毒灾飞艇"
    found_in_main = target_word in all_main_words
    found_in_synonyms = target_word in all_synonyms
    
    print(f"目标词汇: '{target_word}'")
    print(f"在主词汇中: {'✅' if found_in_main else '❌'}")
    print(f"在同义词中: {'✅' if found_in_synonyms else '❌'}")
    
    if not found_in_main and not found_in_synonyms:
        print("\n🔍 搜索相似词汇:")
        similar_words = []
        for word in all_main_words:
            if any(part in word for part in ["死神军", "毒灾", "飞艇"]):
                similar_words.append(word)
        
        for word in all_synonyms:
            if any(part in word for part in ["死神军", "毒灾", "飞艇"]):
                similar_words.append(word)
        
        if similar_words:
            print("找到相似词汇:")
            for word in sorted(set(similar_words)):
                print(f"  - {word}")
        else:
            print("未找到任何相似词汇")
    
    # 检查wh40k_vocabulary中实际包含的词汇
    print("\n" + "="*60)
    print("📋 wh40k_vocabulary实际包含的词汇样本:")
    
    # 显示一些主词汇
    print("\n主词汇样本 (前20个):")
    for word in sorted(list(all_main_words))[:20]:
        print(f"  - {word}")
    
    # 显示一些同义词
    print("\n同义词样本 (前20个):")
    for word in sorted(list(all_synonyms))[:20]:
        print(f"  - {word}")


def main():
    """主函数"""
    wh40k_vocab_dir = "dict/wh40k_vocabulary"
    
    if not os.path.exists(wh40k_vocab_dir):
        print(f"❌ wh40k_vocabulary 目录不存在: {wh40k_vocab_dir}")
        return
    
    debug_wh40k_vocabulary_loading(wh40k_vocab_dir)


if __name__ == "__main__":
    main()
