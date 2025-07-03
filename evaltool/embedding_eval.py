import os
import sys
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
# import 可视化工具
from visualization import plot_embedding_2d, plot_similarity_hist
from sklearn.manifold import TSNE

# 动态加载config.py，获取get_embedding_model
config_path = Path(__file__).parent.parent / 'config.py'
spec = importlib.util.spec_from_file_location('config', str(config_path))
config = importlib.util.module_from_spec(spec)
sys.modules['config'] = config
spec.loader.exec_module(config)

def run_embedding_eval(csv_path, model_name=None, top_k=5, config_file=None):
    print(f"[INFO] 加载csv: {csv_path}")
    df = pd.read_csv(csv_path)
    queries = df.iloc[:, 0].astype(str).tolist()
    print(f"[INFO] 共{len(queries)}条query")

    print(f"[INFO] 加载embedding模型（项目config）...")
    embedding_model = config.get_embedding_model()

    print("[INFO] 生成query embedding...")
    query_embeddings = embedding_model.embed_documents(queries)
    query_embeddings = np.array(query_embeddings)

    print("[INFO] 可视化embedding二维分布...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(query_embeddings)
    plot_embedding_2d(query_embeddings, labels=queries, title='Query Embedding 2D Visualization', reduced=reduced)

    # 保存降维坐标到csv
    reduced_df = pd.DataFrame(reduced, columns=['x', 'y'])
    reduced_df['query'] = queries
    reduced_df.to_csv('evaltool/query_2d_coords.csv', index=False)
    print('[INFO] 降维坐标已保存到 evaltool/query_2d_coords.csv')

    # 四象限划分
    q1, q2, q3, q4 = [], [], [], []
    for i, (x, y) in enumerate(reduced):
        if x >= 0 and y >= 0:
            q1.append(queries[i])
        elif x < 0 and y >= 0:
            q2.append(queries[i])
        elif x < 0 and y < 0:
            q3.append(queries[i])
        else:
            q4.append(queries[i])
    print('\n【第一象限】')
    for q in q1:
        print(q)
    print('\n【第二象限】')
    for q in q2:
        print(q)
    print('\n【第三象限】')
    for q in q3:
        print(q)
    print('\n【第四象限】')
    for q in q4:
        print(q)

    print("[INFO] 计算query间相似度分布...")
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(query_embeddings)
    plot_similarity_hist(sim_matrix, title='Query Similarity Distribution')

    print("[INFO] 评估完成，结果已输出")

    print("[INFO] 输出典型case分析表...")
    # TODO: 输出分析表
    print("[TODO] 详细实现待补充") 