import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
# import umap, tsne等

def plot_embedding_2d(embeddings, labels=None, title='Embedding 2D Visualization', save_path='embedding_2d.png', reduced=None):
    if reduced is None:
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.6)
    if labels is not None:
        for i, txt in enumerate(labels):
            plt.annotate(txt[:10], (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] embedding二维分布图已保存: {save_path}")
    plt.close()

def plot_similarity_hist(similarity_matrix, title='Similarity Distribution', save_path='similarity_hist.png'):
    # 只统计上三角（去除对角线和重复）
    sim_vals = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    plt.figure(figsize=(8, 6))
    plt.hist(sim_vals, bins=30, color='green', alpha=0.7)
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] 相似度分布直方图已保存: {save_path}")
    plt.close() 