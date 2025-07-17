import os
import sys
import re
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataupload.bm25_manager import BM25Manager

# 停用词、标点、主语等过滤同前
punctuations = set('，。！？：；、""''（）—…《》·,.!?;:"\'()[]{}<>|/~`+=-_')
pronouns = set(['你', '我', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '本人', '自己', '咱们'])
particles = set(['的', '了', '着', '过', '和', '与', '及', '或', '也', '都', '就', '还', '被', '把', '在', '是', '为', '对', '以', '于', '而', '并', '从', '到', '由', '向', '给', '被', '让', '被', '把', '将', '呢', '吗', '吧', '啊', '呀', '嘛', '哦', '呗', '啦', '哇', '么', '嘛', '喽', '罢', '咧', '哟', '咦', '呦', '哎', '欸', '呃', '呜', '呗', '呸', '哼', '唉', '嘻', '嘶', '嘭', '噢', '噗', '噼', '哗', '咚', '咔', '咯', '咳', '咩', '咪', '咻', '咿', '咽', '咱', '咬', '咳', '咦', '咧', '咱', '咩', '咻', '咿', '咽', '咬', '咳', '咦', '咧'])

# 过滤函数
def filter_word(word):
    if word in punctuations or word in pronouns or word in particles:
        return False
    if len(word) == 1 and word not in set('战术规则军队单位模型分队技能'):
        return False
    if re.match(r'^#+$', word):
        return False
    if re.match(r'^\s+$', word):
        return False
    if re.match(r'^<.*>$', word):
        return False
    return True

def vocab_coverage(vocab, queries, doc_tokens):
    # 查询OOV率
    oov_count = 0
    for q in queries:
        if q not in vocab:
            oov_count += 1
    oov_rate = oov_count / len(queries) if queries else 0
    # 高频词覆盖率
    covered = [w for w in doc_tokens if w in vocab]
    coverage = len(set(covered)) / len(set(doc_tokens)) if doc_tokens else 0
    return oov_rate, coverage

def idf_distribution(bm25_manager):
    if not bm25_manager.is_fitted:
        return []
    idf_vals = [bm25_manager.bm25_model.idf.get(w, 0) for w in bm25_manager.vocabulary]
    return idf_vals

def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def run_vocab_eval(vocab_path, out_dir):
    # 只加载json，做统计、可视化、报告输出
    if not os.path.exists(vocab_path):
        print(f"[ERROR] 词典文件不存在: {vocab_path}")
        return
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    tokens = list(vocab_dict.keys())
    if not tokens:
        print("[WARN] 词典为空，无可评估内容。")
    # 统计token长度分布
    token_lens = [len(t) for t in tokens]
    # 统计token类型分布
    single = sum(1 for t in tokens if len(t) == 1)
    multi = sum(1 for t in tokens if len(t) > 1)
    # 生成时间戳
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    # 产物输出
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 画token长度分布图
    if token_lens:
        plt.figure(figsize=(6,4))
        plt.hist(token_lens, bins=range(1, max(token_lens)+2), color='skyblue', edgecolor='black', align='left')
        plt.xlabel('Token长度')
        plt.ylabel('数量')
        plt.title('Token长度分布')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'token_len_hist_{timestamp}.png'))
        plt.close()
    else:
        print("[WARN] 无有效token，未生成长度分布图。")
    # 生成.md报告
    report_lines = []
    report_lines.append(f'# 词典评估报告 {timestamp}\n')
    report_lines.append('## 1. Token长度分布')
    if not token_lens:
        report_lines.append('- 无有效token')
    else:
        for l in sorted(set(token_lens)):
            report_lines.append(f'- 长度={l}: {token_lens.count(l)}')
    report_lines.append('')
    report_lines.append('## 2. Token类型分布（单字/多字/短语）')
    if not tokens:
        report_lines.append('- 无有效token')
    else:
        report_lines.append(f'- 单字token数: {single}')
        report_lines.append(f'- 多字/短语token数: {multi}')
        report_lines.append(f'- 总token数: {len(tokens)}')
    report_path = os.path.join(out_dir, f'vocab_eval_report_{timestamp}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'[INFO] 评估报告已保存: {report_path}')
    print(f'[INFO] 评估产物输出目录: {out_dir}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BM25词典评估工具：支持自动生成+分析或只分析已有词典")
    parser.add_argument('--input-folder', type=str, help='如指定则自动生成词典')
    parser.add_argument('--output-dir', type=str, default='dict', help='词典产出目录')
    parser.add_argument('--filter', action='store_true', default=True, help='是否执行token过滤，默认True')
    parser.add_argument('--no-filter', dest='filter', action='store_false', help='不执行token过滤')
    parser.add_argument('--dict-dir', type=str, help='已有词典目录，优先级高于output-dir')
    args = parser.parse_args()

    # 1. 如有input-folder，先生成词典
    if args.input_folder:
        gen_cmd = [
            'python3', 'DATAUPLOD/gen_userdict.py',
            '--input-folder', args.input_folder,
            '--output-dir', args.output_dir
        ]
        if args.filter:
            gen_cmd.append('--filter')
        else:
            gen_cmd.append('--no-filter')
        print(f"[INFO] 正在生成词典: {' '.join(gen_cmd)}")
        ret = subprocess.run(gen_cmd)
        if ret.returncode != 0:
            print("[ERROR] 词典生成失败，终止评估。")
            return
    # 2. 查找词典目录
    dict_dir = args.dict_dir if args.dict_dir else args.output_dir
    vocab_path = find_latest_file(os.path.join(dict_dir, 'bm25_vocab_*.json'))
    if not vocab_path:
        print(f"[ERROR] 未找到bm25_vocab_*.json，请检查目录: {dict_dir}")
        return
    # 3. 评估产物输出目录
    day_folder = datetime.now().strftime('%m%d')
    eval_out_dir = os.path.join(os.path.dirname(__file__), day_folder)
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)
    # 4. 执行评估
    run_vocab_eval(vocab_path, eval_out_dir)

if __name__ == "__main__":
    main() 