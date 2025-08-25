#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证脚本：检查warhammer_dict.txt的完整性
1. 检查是否与all_docs_dict_2508242302.txt.meta.json中的词汇一一对应
2. 检查是否包含了wh40k_vocabulary下4个json文件里的所有专业词汇
"""

import json
import os
from pathlib import Path
from typing import Set, Dict, List, Tuple


def load_warhammer_dict(file_path: str) -> Set[str]:
    """加载warhammer_dict.txt中的词汇"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 每行一个词汇，去除空白字符
            words = {line.strip() for line in f if line.strip()}
        print(f"✅ 成功加载warhammer_dict.txt: {len(words)}个词汇")
        return words
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return set()
    except Exception as e:
        print(f"❌ 加载warhammer_dict.txt失败: {e}")
        return set()


def load_all_docs_dict_meta(file_path: str) -> Set[str]:
    """加载all_docs_dict_2508242302.txt.meta.json中的词汇"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 提取所有词汇（JSON的键）
        words = set(data.keys())
        print(f"✅ 成功加载all_docs_dict_2508242302.txt.meta.json: {len(words)}个词汇")
        return words
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
        return set()
    except Exception as e:
        print(f"❌ 加载all_docs_dict_2508242302.txt.meta.json失败: {e}")
        return set()


def load_wh40k_vocabulary(vocab_dir: str) -> Set[str]:
    """加载wh40k_vocabulary下所有JSON文件中的专业词汇"""
    vocab_dir_path = Path(vocab_dir)
    if not vocab_dir_path.exists():
        print(f"❌ 目录不存在: {vocab_dir}")
        return set()
    
    all_words = set()
    json_files = list(vocab_dir_path.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取主词汇和同义词
            for main_word, word_info in data.items():
                all_words.add(main_word)
                # 添加同义词
                if isinstance(word_info, dict) and "synonyms" in word_info:
                    all_words.update(word_info["synonyms"])
            
            print(f"✅ 成功加载 {json_file.name}: {len(data)}个主词汇")
            
        except Exception as e:
            print(f"❌ 加载 {json_file.name} 失败: {e}")
    
    print(f"✅ 总计加载wh40k_vocabulary: {len(all_words)}个专业词汇")
    return all_words


def compare_vocabularies(warhammer_words: Set[str], all_docs_meta_words: Set[str], wh40k_words: Set[str]) -> Tuple[Dict, Dict]:
    """比较三个词汇集合"""
    results = {
        "warhammer_vs_all_docs_meta": {},
        "warhammer_vs_wh40k": {}
    }
    
    # 1. 比较warhammer_dict vs all_docs_dict_meta
    print("\n" + "="*60)
    print("1. 检查warhammer_dict.txt vs all_docs_dict_2508242302.txt.meta.json")
    print("="*60)
    
    # 检查warhammer_dict中是否包含所有all_docs_dict_meta词汇
    missing_in_warhammer = all_docs_meta_words - warhammer_words
    extra_in_warhammer = warhammer_words - all_docs_meta_words
    
    results["warhammer_vs_all_docs_meta"] = {
        "warhammer_total": len(warhammer_words),
        "all_docs_meta_total": len(all_docs_meta_words),
        "missing_in_warhammer": len(missing_in_warhammer),
        "extra_in_warhammer": len(extra_in_warhammer),
        "common": len(warhammer_words & all_docs_meta_words),
        "missing_list": sorted(list(missing_in_warhammer)),
        "extra_list": sorted(list(extra_in_warhammer))
    }
    
    print(f"warhammer_dict.txt 总词汇数: {len(warhammer_words)}")
    print(f"all_docs_dict_meta.json 总词汇数: {len(all_docs_meta_words)}")
    print(f"共同词汇数: {len(warhammer_words & all_docs_meta_words)}")
    print(f"warhammer_dict.txt 缺失的词汇数: {len(missing_in_warhammer)}")
    print(f"warhammer_dict.txt 多余的词汇数: {len(extra_in_warhammer)}")
    
    if missing_in_warhammer:
        print(f"\n❌ warhammer_dict.txt 缺失的词汇 (前20个):")
        for word in sorted(list(missing_in_warhammer))[:20]:
            print(f"  - {word}")
        if len(missing_in_warhammer) > 20:
            print(f"  ... 还有 {len(missing_in_warhammer) - 20} 个")
    
    if extra_in_warhammer:
        print(f"\n⚠️  warhammer_dict.txt 多余的词汇 (前20个):")
        for word in sorted(list(extra_in_warhammer))[:20]:
            print(f"  - {word}")
        if len(extra_in_warhammer) > 20:
            print(f"  ... 还有 {len(extra_in_warhammer) - 20} 个")
    
    # 2. 比较warhammer_dict vs wh40k_vocabulary
    print("\n" + "="*60)
    print("2. 检查warhammer_dict.txt vs wh40k_vocabulary")
    print("="*60)
    
    # 检查warhammer_dict中是否包含所有wh40k专业词汇
    missing_wh40k_in_warhammer = wh40k_words - warhammer_words
    
    results["warhammer_vs_wh40k"] = {
        "warhammer_total": len(warhammer_words),
        "wh40k_total": len(wh40k_words),
        "missing_wh40k_in_warhammer": len(missing_wh40k_in_warhammer),
        "common": len(warhammer_words & wh40k_words),
        "missing_list": sorted(list(missing_wh40k_in_warhammer))
    }
    
    print(f"warhammer_dict.txt 总词汇数: {len(warhammer_words)}")
    print(f"wh40k_vocabulary 总专业词汇数: {len(wh40k_words)}")
    print(f"共同词汇数: {len(warhammer_words & wh40k_words)}")
    print(f"warhammer_dict.txt 缺失的wh40k专业词汇数: {len(missing_wh40k_in_warhammer)}")
    
    if missing_wh40k_in_warhammer:
        print(f"\n❌ warhammer_dict.txt 缺失的wh40k专业词汇 (前30个):")
        for word in sorted(list(missing_wh40k_in_warhammer))[:30]:
            print(f"  - {word}")
        if len(missing_wh40k_in_warhammer) > 30:
            print(f"  ... 还有 {len(missing_wh40k_in_warhammer) - 30} 个")
    
    return results


def generate_summary_report(results: Dict) -> str:
    """生成总结报告"""
    report = []
    report.append("="*80)
    report.append("📊 WARHAMMER_DICT.TXT 验证总结报告")
    report.append("="*80)
    
    # 1. warhammer vs all_docs_dict_meta 总结
    all_docs_result = results["warhammer_vs_all_docs_meta"]
    report.append(f"\n1️⃣ 与 all_docs_dict_2508242302.txt.meta.json 对比:")
    report.append(f"   ✅ 共同词汇: {all_docs_result['common']}")
    report.append(f"   ❌ 缺失词汇: {all_docs_result['missing_in_warhammer']}")
    report.append(f"   ⚠️  多余词汇: {all_docs_result['extra_in_warhammer']}")
    
    if all_docs_result['missing_in_warhammer'] == 0:
        report.append("   🎉 完全匹配！warhammer_dict.txt 包含所有 all_docs_dict_meta 词汇")
    else:
        report.append("   ⚠️  存在缺失，需要检查")
    
    # 2. warhammer vs wh40k 总结
    wh40k_result = results["warhammer_vs_wh40k"]
    report.append(f"\n2️⃣ 与 wh40k_vocabulary 对比:")
    report.append(f"   ✅ 共同专业词汇: {wh40k_result['common']}")
    report.append(f"   ❌ 缺失专业词汇: {wh40k_result['missing_wh40k_in_warhammer']}")
    
    if wh40k_result['missing_wh40k_in_warhammer'] == 0:
        report.append("   🎉 完全覆盖！warhammer_dict.txt 包含所有 wh40k 专业词汇")
    else:
        report.append("   ⚠️  存在缺失，需要检查")
    
    # 3. 总体评估
    report.append(f"\n3️⃣ 总体评估:")
    total_issues = all_docs_result['missing_in_warhammer'] + wh40k_result['missing_wh40k_in_warhammer']
    if total_issues == 0:
        report.append("   🎉 完美！warhammer_dict.txt 完全满足要求")
    elif total_issues <= 10:
        report.append("   ✅ 良好！存在少量问题，基本满足要求")
    elif total_issues <= 50:
        report.append("   ⚠️  一般！存在一些问题，需要关注")
    else:
        report.append("   ❌ 较差！存在较多问题，需要修复")
    
    report.append("="*80)
    return "\n".join(report)


def main():
    """主函数"""
    print("🔍 开始验证 warhammer_dict.txt 的完整性...")
    
    # 文件路径
    warhammer_dict_path = "dict/warhammer_dict_2508242302.txt"
    all_docs_dict_meta_path = "dict/all_docs_dict_2508242302.txt.meta.json"
    wh40k_vocab_dir = "dict/wh40k_vocabulary"
    
    # 检查文件是否存在
    if not os.path.exists(warhammer_dict_path):
        print(f"❌ warhammer_dict.txt 不存在: {warhammer_dict_path}")
        return
    
    if not os.path.exists(all_docs_dict_meta_path):
        print(f"❌ all_docs_dict_2508242302.txt.meta.json 不存在: {all_docs_dict_meta_path}")
        return
    
    if not os.path.exists(wh40k_vocab_dir):
        print(f"❌ wh40k_vocabulary 目录不存在: {wh40k_vocab_dir}")
        return
    
    # 加载词汇
    warhammer_words = load_warhammer_dict(warhammer_dict_path)
    all_docs_meta_words = load_all_docs_dict_meta(all_docs_dict_meta_path)
    wh40k_words = load_wh40k_vocabulary(wh40k_vocab_dir)
    
    if not warhammer_words or not all_docs_meta_words or not wh40k_words:
        print("❌ 词汇加载失败，无法继续验证")
        return
    
    # 比较词汇
    results = compare_vocabularies(warhammer_words, all_docs_meta_words, wh40k_words)
    
    # 生成总结报告
    summary = generate_summary_report(results)
    print(summary)
    
    # 保存详细结果到JSON文件
    output_file = "warhammer_dict_verification_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 详细验证结果已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存验证结果失败: {e}")


if __name__ == "__main__":
    main()
