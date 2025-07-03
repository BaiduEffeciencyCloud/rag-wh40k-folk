import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List

# 动态加载cot_search.py
import importlib.util
cot_search_path = Path(__file__).parent.parent / 'cot_search.py'
spec = importlib.util.spec_from_file_location('cot_search', str(cot_search_path))
cot_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cot_search)

# 动态加载config.py获取embedding模型
config_path = Path(__file__).parent.parent / 'config.py'
spec2 = importlib.util.spec_from_file_location('config', str(config_path))
config = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(config)

from sklearn.metrics.pairwise import cosine_similarity

def compute_embedding_similarity(model, a: str, b: str) -> float:
    """用embedding余弦相似度衡量两个文本的语义相似度"""
    try:
        vecs = model.embed_documents([a, b])
        sim = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
        return float(sim)
    except Exception as e:
        print(f"[ERROR] 计算embedding相似度失败: {e}")
        return 0.0

def run_answer_eval_v2(csv_path, top_k=5, report_prefix=None):
    print(f"[INFO] 读取评测csv文件: {csv_path}")
    df = pd.read_csv(csv_path)
    queries = df.iloc[:, 0].astype(str).tolist()
    std_answers = df.iloc[:, 1].astype(str).tolist()
    print(f"[INFO] 共{len(queries)}条query，准备评测生成答案效果")

    print("[INFO] 初始化cot_search主流程...")
    cot = cot_search.CoTSearch()
    print("[INFO] 初始化embedding模型...")
    embedding_model = config.get_embedding_model()

    results = []
    for i, (query, std_answer) in enumerate(zip(queries, std_answers)):
        print(f"[INFO] [{i+1}/{len(queries)}] 正在评测: {query}")
        try:
            sys_result = cot.search(query, top_k=top_k)
            gen_answer = sys_result.get('final_answer', '').strip()
        except Exception as e:
            print(f"[ERROR] cot_search失败: {e}")
            gen_answer = ''
        # 完全匹配
        em = int(gen_answer == std_answer)
        # embedding相似度
        try:
            sim = compute_embedding_similarity(embedding_model, gen_answer, std_answer)
        except Exception as e:
            print(f"[ERROR] 计算相似度失败: {e}")
            sim = 0.0
        results.append({
            'query': query,
            'gen_answer': gen_answer,
            'std_answer': std_answer,
            'exact_match': em,
            'similarity': sim
        })
        print(f"[INFO] 完全匹配: {'是' if em else '否'}，相似度: {sim:.2f}")

    print("[INFO] 统计整体评测指标...")
    n = len(results)
    em_rate = sum(r['exact_match'] for r in results) / n if n else 0
    avg_sim = np.mean([r['similarity'] for r in results]) if n else 0

    print("[INFO] 筛选典型错误case...")
    error_cases = [r for r in results if r['similarity'] < 0.7 or r['exact_match'] == 0][:5]

    print("[INFO] 生成详细对比表...")
    detail_md = '| query | 生成答案 | 标准答案 | 完全匹配 | 相似度 |\n|-------|----------|----------|----------|--------|\n'
    for r in results:
        detail_md += f"| {r['query'][:20]} | {r['gen_answer'][:20]} | {r['std_answer'][:20]} | {'是' if r['exact_match'] else '否'} | {r['similarity']:.2f} |\n"

    print("[INFO] 生成典型错误case明细...")
    error_md = ''
    for r in error_cases:
        error_md += f"- query: {r['query']}\n  生成答案: {r['gen_answer']}\n  标准答案: {r['std_answer']}\n  相似度: {r['similarity']:.2f}\n\n"

    print("[INFO] 生成业务解读建议...")
    advice = []
    if em_rate < 0.7:
        advice.append('完全匹配率较低，建议优化答案生成prompt或丰富知识库内容。')
    if avg_sim < 0.8:
        advice.append('平均相似度偏低，建议检查embedding模型与业务场景的适配性。')
    if not advice:
        advice.append('系统整体表现良好，可继续扩展评测样本。')
    advice_md = '\n'.join(f'- {a}' for a in advice)

    print("[INFO] 写入markdown评测报告...")
    today = datetime.now().strftime('%y%m%d')
    prefix = report_prefix or today
    md_path = Path(__file__).parent.parent / f'document/{prefix}-答案生成效果评测报告.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"""# {prefix}-答案生成效果评测报告\n\n""")
        f.write(f"## 一、整体测评指标\n- 样本数：{n}\n- 完全匹配率：{em_rate:.2f}\n- 平均相似度：{avg_sim:.2f}\n\n")
        f.write(f"## 二、详细case对比\n\n{detail_md}\n")
        f.write(f"## 三、典型错误case明细\n\n{error_md}\n")
        f.write(f"## 四、业务解读建议\n\n{advice_md}\n")
    print(f"[INFO] 评测报告已生成：{md_path}") 