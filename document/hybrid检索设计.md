# 250703-BM25_hybrid检索设计文档（补丁式修订版）

## 1. 项目目标与范围

（同前，略）

---

## 2. 系统架构与数据流



## 3. BM25Manager设计与实现

### 3.1 类结构
```python
class BM25Manager:
    def __init__(self, k1=1.5, b=0.75, min_freq=5, max_vocab_size=10000, custom_dict_path=None):
        self.k1 = k1
        self.b = b
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.bm25_model = None
        self.vocabulary = {}
        self.corpus_tokens = []
        self.raw_texts = []  # 新增：原始文本列表
        self.is_fitted = False
        if custom_dict_path and os.path.exists(custom_dict_path):
            jieba.load_userdict(custom_dict_path)
    def tokenize_chinese(self, text): ...
    def fit(self, corpus_texts): ...
    def incremental_update(self, new_corpus_texts): ...
    def get_sparse_vector(self, text): ...
    def save_model(self, model_path, vocab_path): ...
    def load_model(self, model_path, vocab_path): ...
    def get_vocabulary_stats(self): ...
```

### 3.2 词汇表剪枝与增量更新（修正）
- **剪枝**：仅保留出现频率大于`min_freq`的前`max_vocab_size`高频词
- **增量更新**：维护`self.raw_texts`，每次`fit`/`incremental_update`都用**原始文本列表**重建词表，避免token当文档的bug

def fit(self, corpus_texts):
    self.raw_texts = list(corpus_texts)
    self.corpus_tokens = [self.tokenize_chinese(text) for text in self.raw_texts]
    self._build_vocabulary(self.raw_texts)
    self.bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
    self.is_fitted = True

def incremental_update(self, new_corpus_texts):
    self.raw_texts.extend(new_corpus_texts)

### 3.3 BM25权重计算（完整公式，含k1、b、归一化）
def get_sparse_vector(self, text: str) -> Dict[str, Any]:
    tokens = self.tokenize_chinese(text)
    indices, values = [], []
    avgdl = self.bm25_model.avgdl
    k1, b = self.bm25_model.k1, self.bm25_model.b
    dl = len(tokens)
    token_freq = {}
    for token in tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    for token, freq in token_freq.items():
        if token in self.vocabulary:
            idf = self.bm25_model.idf.get(token, 0)
            tf = freq
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            weight = idf * (tf * (k1 + 1)) / denom if denom > 0 else 0
            if weight > 0:
                indices.append(self.vocabulary[token])
                values.append(float(weight))
    return {"indices": indices, "values": values}


## 4. Hybrid融合与检索

### 4.1 hybrid_scale方法（补充实现）
def hybrid_scale(self, dense, sparse, alpha):
    hdense = [v * alpha for v in dense]
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]]
    }
    return hdense, hsparse
- 检索时：`hdense, hsparse = self.hybrid_scale(query_dense, query_sparse, alpha)`


## 5. 配置与导入规范

### 5.1 config.py（补充/修正）
BM25_K1 = float(os.getenv('BM25_K1', '1.5'))
BM25_B = float(os.getenv('BM25_B', '0.75'))
BM25_MIN_FREQ = int(os.getenv('BM25_MIN_FREQ', '5'))
BM25_MAX_VOCAB_SIZE = int(os.getenv('BM25_MAX_VOCAB_SIZE', '10000'))
BM25_CUSTOM_DICT_PATH = os.getenv('BM25_CUSTOM_DICT_PATH', 'dicts/warhammer40k.txt')
BM25_MODEL_PATH = os.getenv('BM25_MODEL_PATH', 'models/bm25_model.pkl')
BM25_VOCAB_PATH = os.getenv('BM25_VOCAB_PATH', 'models/bm25_vocab.json')
PINECONE_SPARSE_DIMENSION = int(os.getenv('PINECONE_SPARSE_DIMENSION', '10000'))
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
- **所有索引创建、upsert、query等均应引用配置变量，不应硬编码。**

### 5.2 导入路径规范
- 统一用绝对路径：如`from DATAUPLOD.bm25_manager import BM25Manager`
- 测试脚本加`sys.path`，保证可导入


## 6. 测试与验证（补充）

### 6.1 测试用例
- 稀疏向量正确性（indices/values长度、正值、索引范围）
- 错误处理（未训练、空语料、配置边界）
- 增量更新（词表大小变化、新词汇添加）
- hybrid_scale融合正确性
- Pinecone upsert/query异常捕获

### 6.2 示例测试脚本片段
def test_hybrid_scale():
    dense = [1.0, 2.0, 3.0]
    sparse = {"indices": [0, 2], "values": [0.5, 1.5]}
    alpha = 0.3
    hdense, hsparse = hybrid_scale(dense, sparse, alpha)
    assert all(abs(v - d * alpha) < 1e-6 for v, d in zip(hdense, dense))
    assert all(abs(v - s * (1 - alpha)) < 1e-6 for v, s in zip(hsparse["values"], sparse["values"]))


## 7. 可选：二阶段Rerank设计（补充）

### 7.1 方案说明
- 在Hybrid检索top-k结果基础上，使用Cross-Encoder模型（如`cross-encoder/ms-marco-MiniLM-L-6-v2`）对query-文档对进行二次排序，提升最终相关性。

### 7.2 伪代码
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
rerank_scores = reranker.predict([[query, doc] for doc in topk_docs])
# 按rerank_scores重新排序topk_docs
- 文档可补充"可选Rerank流程"说明，便于后续扩展。


## 8. 监控与优化建议



## 9. 风险与注意事项



## 10. 总结

本补丁式修订版在精炼结构基础上，**补充了BM25公式、词表重建、hybrid_scale、配置引用、测试、Rerank等工程关键细节**，确保设计与实现完全对齐，适用于高质量中文RAG系统的工程落地。


**如需进一步细化某一部分或补充代码实现细节，请随时告知！**


你可以将以上内容保存为 `250703-BM25_hybrid检索设计文档.md`，如需分章节下载或拆分为多文件，也可告知！


# 250703-hybrid检索设计文档

## 1. 项目概述

### 1.1 目标
在现有的RAG系统中集成BM25 sparse embedding，实现完整的hybrid检索功能，提升战锤40K规则检索的召回率和准确性。

### 1.2 实施范围
- **数据上传端**：修改`DATAUPLOD/upsert.py`，支持BM25 sparse向量生成和上传
- **检索端**：修改`vector_search.py`，支持BM25 sparse向量检索
- **配置管理**：统一BM25参数配置和词汇表管理
- **Pinecone索引**：创建支持sparse向量的索引配置

### 1.3 技术栈
- **BM25实现**：`rank_bm25`库
- **中文分词**：`jieba`库（支持自定义词典）
- **向量存储**：Pinecone（metric=dotproduct，支持sparse向量）
- **配置管理**：`config.py` + `.env`

### 1.4 关键修正点
基于ChatGPT的专业建议，本设计文档已修正以下关键问题：
- **BM25向量化逻辑**：修正token权重计算方式
- **词汇表剪枝**：添加频率阈值和大小限制
- **领域词典**：支持战锤40K专有名词优化
- **增量更新**：支持模型和词汇表的增量更新
- **Pinecone配置**：明确sparse向量索引创建要求

## 2. 系统架构设计

### 2.1 整体架构
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档输入      │    │   切片处理      │    │   向量生成      │
│   (Markdown)    │───▶│   (Chunker)     │───▶│  Dense + Sparse │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
│   检索查询      │    │   查询处理      │    │   混合检索      │
│   (User Query)  │◀───│   (Query Proc)  │◀───│  Hybrid Search  │

### 2.2 数据流设计
上传流程：
文档 → 切片 → 领域词典分词 → BM25词汇表构建(剪枝) → Dense Embedding → Sparse Embedding → Pinecone

检索流程：
查询 → 领域词典分词 → BM25稀疏向量 → Dense Embedding → Alpha加权 → Hybrid检索

### 2.3 Pinecone索引配置要求
# 创建支持sparse向量的索引
import pinecone

pinecone.create_index(
    name="40ktest-hybrid",
    dimension=1536,  # dense向量维度
    metric="dotproduct",
    pod_type="s1",
    sparse_vector_config={
        "dimension": 10000  # sparse向量维度，根据词汇表大小调整
)

## 3. 核心模块设计

### 3.1 BM25Manager类设计

#### 3.1.1 类结构
    """BM25稀疏向量管理器"""
    
        self.k1 = k1  # BM25参数k1
        self.b = b    # BM25参数b
        self.min_freq = min_freq  # 最小词频阈值
        self.max_vocab_size = max_vocab_size  # 最大词汇表大小
        self.vocabulary = {}  # 词汇表映射
        self.corpus_tokens = []  # 语料库分词结果
        
        # 加载自定义词典
        
    def tokenize_chinese(self, text):
        """中文分词（支持领域词典）"""
        
        """训练BM25模型"""
        
        """增量更新BM25模型"""
        
    def get_sparse_vector(self, text):
        """生成稀疏向量（修正后的实现）"""
        
    def save_model(self, model_path, vocab_path):
        """保存模型和词汇表"""
        
    def load_model(self, model_path, vocab_path):
        """加载模型和词汇表"""
        
    def get_vocabulary_stats(self):
        """获取词汇表统计信息"""

#### 3.1.2 核心方法实现（修正版）
def _build_vocabulary(self, corpus_texts: List[str]):
    """构建剪枝后的词汇表"""
    # 分词处理
    self.corpus_tokens = [self.tokenize_chinese(text) for text in corpus_texts]
    
    # 统计词频
    for tokens in self.corpus_tokens:
    
    # 过滤低频词
    filtered_tokens = {token: freq for token, freq in token_freq.items() 
                      if freq >= self.min_freq}
    
    # 按频率排序并限制大小
    sorted_tokens = sorted(filtered_tokens.items(), 
                          key=lambda x: x[1], reverse=True)[:self.max_vocab_size]
    
    self.vocabulary = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
    logger.info(f"词汇表构建完成，大小: {len(self.vocabulary)}")

    """
    生成正确的BM25稀疏向量
    
    Args:
        text: 输入文本
        
    Returns:
        dict: {"indices": [int], "values": [float]}
    if not self.is_fitted:
        raise ValueError("BM25模型尚未训练")
    
    indices = []
    values = []
    
    # 统计当前文本的词频
    
    # 计算每个token的BM25权重
            # 获取IDF值
            # 计算TF
            # 计算BM25权重 (简化版)
            weight = idf * tf
    

def incremental_update(self, new_corpus_texts: List[str]):
    logger.info("开始增量更新BM25模型")
    
    # 更新语料库
    new_tokens = [self.tokenize_chinese(text) for text in new_corpus_texts]
    self.corpus_tokens.extend(new_tokens)
    
    # 重新训练模型
    
    # 更新词汇表
    self._build_vocabulary([text for tokens in self.corpus_tokens for text in tokens])
    
    logger.info("BM25模型增量更新完成")

### 3.2 配置管理设计

#### 3.2.1 config.py扩展
# BM25配置

# Hybrid检索配置
HYBRID_ALPHA = float(os.getenv('HYBRID_ALPHA', '0.3'))
ENABLE_HYBRID_SEARCH = os.getenv('ENABLE_HYBRID_SEARCH', 'true').lower() == 'true'

# Pinecone配置

#### 3.2.2 .env模板
```bash
BM25_K1=1.5
BM25_B=0.75
BM25_MIN_FREQ=5
BM25_MAX_VOCAB_SIZE=10000
BM25_CUSTOM_DICT_PATH=dicts/warhammer40k.txt
BM25_MODEL_PATH=models/bm25_model.pkl
BM25_VOCAB_PATH=models/bm25_vocab.json

HYBRID_ALPHA=0.3
ENABLE_HYBRID_SEARCH=true

PINECONE_SPARSE_DIMENSION=10000

## 4. 实施步骤

### 4.1 第一阶段：BM25Manager实现

#### 4.1.1 创建BM25Manager类（修正版）
**文件位置**：`DATAUPLOD/bm25_manager.py`

import jieba
import json
import pickle
import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

    """BM25稀疏向量管理器（修正版）"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, 
                 min_freq: int = 5, max_vocab_size: int = 10000, 
                 custom_dict_path: str = None):
        
            logger.info(f"已加载自定义词典: {custom_dict_path}")
        
    def tokenize_chinese(self, text: str) -> List[str]:
        return list(jieba.cut(text))
    
        
        
        
        
    
    def fit(self, corpus_texts: List[str]):
        logger.info(f"开始训练BM25模型，语料库大小: {len(corpus_texts)}")
        
        # 构建词汇表
        self._build_vocabulary(corpus_texts)
        
        # 训练BM25模型
        
        logger.info(f"BM25模型训练完成，词汇表大小: {len(self.vocabulary)}")
    
        """生成正确的BM25稀疏向量"""
            raise ValueError("BM25模型尚未训练，请先调用fit()方法")
        
        
        
        
    
        
        
        
        
    
    def save_model(self, model_path: str, vocab_path: str):
            raise ValueError("模型尚未训练")
        
        # 创建目录
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        # 保存BM25模型
        with open(model_path, 'wb') as f:
            pickle.dump(self.bm25_model, f)
        
        # 保存词汇表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"BM25模型已保存: {model_path}, 词汇表: {vocab_path}")
    
    def load_model(self, model_path: str, vocab_path: str):
        # 加载BM25模型
        with open(model_path, 'rb') as f:
            self.bm25_model = pickle.load(f)
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)
        
        logger.info(f"BM25模型已加载: {model_path}, 词汇表大小: {len(self.vocabulary)}")
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        return {
            "vocabulary_size": len(self.vocabulary),
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "is_fitted": self.is_fitted

#### 4.1.2 依赖安装
pip3 install rank-bm25 jieba

### 4.2 第二阶段：修改upsert.py

#### 4.2.1 集成BM25Manager
# 在upsert.py中添加
from bm25_manager import BM25Manager
from config import (BM25_K1, BM25_B, BM25_MIN_FREQ, BM25_MAX_VOCAB_SIZE, 
                   BM25_CUSTOM_DICT_PATH, BM25_MODEL_PATH, BM25_VOCAB_PATH)

class UpsertManager:
    def __init__(self, environment: str = "production", chunker_version: str = "v3"):
        # ... 现有代码 ...
        self.bm25_manager = None
        
    def _initialize_services(self, enable_kg: bool, enable_pinecone: bool):
        
        if enable_pinecone and self.bm25_manager is None:
            self.bm25_manager = BM25Manager(
                k1=BM25_K1, 
                b=BM25_B,
                min_freq=BM25_MIN_FREQ,
                max_vocab_size=BM25_MAX_VOCAB_SIZE,
                custom_dict_path=BM25_CUSTOM_DICT_PATH
            
            # 尝试加载已有模型
            try:
                if os.path.exists(BM25_MODEL_PATH) and os.path.exists(BM25_VOCAB_PATH):
                    self.bm25_manager.load_model(BM25_MODEL_PATH, BM25_VOCAB_PATH)
                    logging.info("✅ BM25模型加载完成")
                else:
                    logging.info("⚠️ BM25模型文件不存在，将在首次上传时训练")
            except Exception as e:
                logging.warning(f"⚠️ BM25模型加载失败: {e}")

#### 4.2.2 修改向量上传逻辑
def process_and_upsert_document(self, file_path: str, enable_kg: bool = True, enable_pinecone: bool = True, extra_metadata: dict = None):
    # ... 现有切片代码 ...
    
    # 3. Pinecone 向量上传
    if enable_pinecone and self.pinecone_index and self.embedding_model:
        logging.info("Step 3: 正在进行向量化并上传到 Pinecone...")
        
        # 训练或更新BM25模型
        if self.bm25_manager and not self.bm25_manager.is_fitted:
            logging.info("训练BM25模型...")
            corpus_texts = [chunk['text'] for chunk in chunks]
            self.bm25_manager.fit(corpus_texts)
            
            # 保存模型
            self.bm25_manager.save_model(BM25_MODEL_PATH, BM25_VOCAB_PATH)
        
        # 构建待上传的数据
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(file_path)}-{i}"
            
                # 生成dense embedding
                dense_embedding = self.embedding_model.embed_query(chunk['text'])
                
                # 生成sparse embedding
                sparse_embedding = None
                if self.bm25_manager and self.bm25_manager.is_fitted:
                    sparse_embedding = self.bm25_manager.get_sparse_vector(chunk['text'])
                
                # 构建metadata
                pinecone_metadata = {
                    'text': chunk['text'],
                    'chunk_type': chunk.get('chunk_type', ''),
                    'content_type': chunk.get('content_type', ''),
                    'faction': chunk.get('faction', extra_metadata.get('faction', '未知')),
                    'section_heading': chunk.get('section_heading', ''),
                
                # 添加多级标题字段
                for j in range(1, 7):
                    pinecone_metadata[f'h{j}'] = chunk.get(f'h{j}', '')
                pinecone_metadata['chunk_id'] = str(i)
                pinecone_metadata['source_file'] = chunk.get('source_file', '')
                
                # 构建向量数据
                vector_data = {
                    'id': chunk_id,
                    'values': dense_embedding,
                    'metadata': pinecone_metadata
                
                # 添加sparse向量（如果可用）
                if sparse_embedding:
                    vector_data['sparse_values'] = sparse_embedding
                
                vectors_to_upsert.append(vector_data)
                
                logging.error(f"Chunk {i} 向量化失败: {e}")
                continue
        
        # 分批上传
        # ... 现有上传代码 ...

### 4.3 第三阶段：修改vector_search.py

#### 4.3.1 集成BM25Manager
# 在vector_search.py中添加
from DATAUPLOD.bm25_manager import BM25Manager
from config import (BM25_MODEL_PATH, BM25_VOCAB_PATH, BM25_K1, BM25_B, 
                   BM25_MIN_FREQ, BM25_MAX_VOCAB_SIZE, BM25_CUSTOM_DICT_PATH,
                   HYBRID_ALPHA, ENABLE_HYBRID_SEARCH)

class VectorSearch:
    def __init__(self, pinecone_api_key: str, index_name: str, openai_api_key=OPENAI_API_KEY):
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """初始化BM25管理器"""
        if ENABLE_HYBRID_SEARCH:
                
                    logger.info("✅ BM25模型加载完成")
                    logger.warning("⚠️ BM25模型文件不存在，将使用hash占位实现")
                logger.error(f"❌ BM25模型加载失败: {e}")

#### 4.3.2 修改get_sparse_embedding方法
def get_sparse_embedding(self, text: str) -> Dict[str, Any]:
    """生成sparse embedding"""
    # 优先使用BM25
            return self.bm25_manager.get_sparse_vector(text)
            logger.warning(f"BM25生成失败，使用hash占位: {e}")
    
    # 降级到hash占位实现
    from collections import Counter
    import re
    tokens = re.findall(r"\w+", text.lower())
    counter = Counter(tokens)
    indices = [abs(hash(tok)) % (2**31) for tok in counter.keys()]
    values = list(counter.values())

#### 4.3.3 修改search方法
def search(self, query: str, top_k: int = 10, alpha: float = None) -> List[Dict[str, Any]]:
    执行hybrid向量搜索
    
        query: 查询文本
        top_k: 返回结果数量
        alpha: dense/sparse加权参数，None时使用配置默认值
    if alpha is None:
        alpha = HYBRID_ALPHA
    
        # 获取dense embedding
        query_embedding = self.get_embedding(query)
        
        # 获取sparse embedding
        query_sparse = self.get_sparse_embedding(query)
        
        # 按alpha加权
        hdense, hsparse = self.hybrid_scale(query_embedding, query_sparse, alpha)
        
        # 构建查询参数
        query_params = {
            'vector': hdense,
            'top_k': top_k,
            'include_metadata': True
        
        if hsparse and hsparse.get('indices'):
            query_params['sparse_vector'] = hsparse
        
        # 执行查询
        results = self.index.query(**query_params)
        
        # 处理结果
        search_results = []
        for match in results.matches:
            text = match.metadata.get('text', '')
            if not text or len(text.strip()) < 10:
            search_results.append({
                'text': text,
                'score': match.score,
                'metadata': match.metadata
            })
        
        return search_results
        
        logger.error(f"搜索时出错：{str(e)}")
        return []

### 4.4 第四阶段：测试和验证

#### 4.4.1 创建测试脚本（修正版）
**文件位置**：`test/test_bm25_hybrid.py`

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_search import VectorSearch
from config import (PINECONE_API_KEY, PINECONE_INDEX, BM25_MIN_FREQ, 
                   BM25_MAX_VOCAB_SIZE, BM25_CUSTOM_DICT_PATH)

def test_bm25_manager():
    """测试BM25Manager"""
    print("🧪 测试BM25Manager...")
    
    # 测试数据
    corpus = [
        "战锤40K是一款桌面战棋游戏",
        "阿苏焉尼是战锤40K中的一个种族",
        "凯恩化身是阿苏焉尼的强力单位",
        "猎鹰坦克具有强大的火力支援能力"
    ]
    
    # 初始化并训练（使用配置参数）
    bm25 = BM25Manager(
    bm25.fit(corpus)
    
    # 测试稀疏向量生成
    test_text = "阿苏焉尼凯恩化身"
    sparse_vec = bm25.get_sparse_vector(test_text)
    
    print(f"✅ 稀疏向量生成成功: {sparse_vec}")
    
    # 测试词汇表统计
    stats = bm25.get_vocabulary_stats()
    print(f"📊 词汇表统计: {stats}")
    
    return bm25

def test_sparse_vector_correctness():
    """测试稀疏向量生成正确性"""
    print("🧪 测试稀疏向量正确性...")
    
    bm25 = BM25Manager()
    corpus = ["测试文本包含战锤40K相关内容"]
    
    test_text = "战锤40K测试"
    
    # 验证indices和values长度一致
    assert len(sparse_vec['indices']) == len(sparse_vec['values']), "indices和values长度不一致"
    
    # 验证values都是正数
    assert all(v > 0 for v in sparse_vec['values']), "存在非正值"
    
    # 验证indices在词汇表范围内
    max_index = max(bm25.vocabulary.values()) if bm25.vocabulary else -1
    assert all(0 <= idx <= max_index for idx in sparse_vec['indices']), "索引超出范围"
    
    print("✅ 稀疏向量正确性验证通过")

def test_incremental_update():
    """测试增量更新"""
    print("🧪 测试增量更新...")
    
    initial_corpus = ["初始语料库内容"]
    bm25.fit(initial_corpus)
    
    initial_size = len(bm25.vocabulary)
    
    # 增量更新
    new_corpus = ["新增语料库内容包含新词汇"]
    bm25.incremental_update(new_corpus)
    
    new_size = len(bm25.vocabulary)
    print(f"✅ 词汇表大小变化: {initial_size} -> {new_size}")

def test_hybrid_search():
    """测试hybrid检索"""
    print("🧪 测试Hybrid检索...")
    
    vs = VectorSearch(PINECONE_API_KEY, PINECONE_INDEX)
    
    # 测试查询
    query = "阿苏焉尼凯恩化身"
    results = vs.search(query, top_k=5, alpha=0.3)
    
    print(f"✅ 找到 {len(results)} 个结果")
    for i, result in enumerate(results, 1):
        print(f"  {i}. 分数: {result['score']:.4f}")
        print(f"     内容: {result['text'][:100]}...")
    
    return results

def test_error_handling():
    """测试错误处理"""
    print("🧪 测试错误处理...")
    
    
    # 测试未训练模型调用get_sparse_vector
        bm25.get_sparse_vector("测试文本")
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"✅ 正确捕获异常: {e}")
    
    # 测试空文本处理
        bm25.fit([])
        print("✅ 空语料库处理正常")
        print(f"⚠️ 空语料库处理异常: {e}")

if __name__ == "__main__":
    print("🚀 开始BM25 Hybrid检索测试")
    print("=" * 60)
    
    # 测试BM25Manager
    bm25 = test_bm25_manager()
    
    # 测试稀疏向量正确性
    test_sparse_vector_correctness()
    
    # 测试增量更新
    test_incremental_update()
    
    # 测试错误处理
    test_error_handling()
    
    # 测试Hybrid检索
    results = test_hybrid_search()
    
    print("✅ 所有测试完成！")

#### 4.4.2 运行测试
python3 test/test_bm25_hybrid.py

## 5. 部署和配置

### 5.1 环境配置
# 安装依赖

# 创建模型目录
mkdir -p models
mkdir -p dicts

# 创建战锤40K词典文件
cat > dicts/warhammer40k.txt << EOF
阿苏焉尼 n
凯恩化身 n
猎鹰坦克 n
战锤40K n
司战 n
斥候 n
火力支援 n
死亡之舞 n
EOF

# 配置环境变量
cp .env.template .env
# 编辑.env文件，添加BM25配置

### 5.2 数据迁移
# 1. 清空现有索引（如果需要重建）
python3 clear_pinecone_40ktest.py

# 2. 重新上传数据（会自动训练BM25模型）
python3 DATAUPLOD/upsert.py --file your_document.md

# 3. 验证hybrid检索效果

## 6. 监控和优化

### 6.1 性能监控
- **BM25模型大小**：监控词汇表大小和模型文件大小
- **检索响应时间**：对比hybrid vs dense检索的响应时间
- **召回效果**：使用评估工具对比检索效果

### 6.2 参数调优
- **BM25参数**：调整k1和b参数
- **Alpha参数**：调整dense/sparse权重
- **词汇表优化**：定期更新词汇表

## 7. 风险评估和应对

### 7.1 技术风险
- **BM25模型加载失败**：有hash占位降级方案
- **词汇表过大**：可以设置最小频率阈值
- **性能影响**：BM25计算相对轻量，影响可控

### 7.2 业务风险
- **检索效果下降**：可以通过alpha参数调整
- **系统复杂度增加**：有完整的配置开关控制

## 8. ChatGPT建议与修正总结

### 8.1 关键问题修正

#### 8.1.1 BM25向量化逻辑错误 ⭐⭐⭐⭐⭐
**问题**：原实现使用`get_scores(tokens)`返回文档得分，而非token权重
**修正**：使用`bm25_model.idf[token] * tf`计算真正的token权重
**影响**：确保sparse向量的语义正确性

#### 8.1.2 词汇表规模控制 ⭐⭐⭐⭐⭐
**问题**：全量词汇表导致维度爆炸
**修正**：添加`min_freq`和`max_vocab_size`参数进行剪枝
**影响**：显著降低存储和计算成本

#### 8.1.3 领域词典支持 ⭐⭐⭐⭐
**问题**：jieba对专有名词分词不理想
**修正**：支持自定义词典加载
**影响**：提升战锤40K专有名词识别准确性

#### 8.1.4 增量更新机制 ⭐⭐⭐⭐
**问题**：缺少模型增量更新策略
**修正**：实现`incremental_update`方法
**影响**：支持新语料的动态更新

#### 8.1.5 Pinecone配置明确化 ⭐⭐⭐⭐⭐
**问题**：缺少sparse向量索引创建说明
**修正**：明确索引配置要求和参数
**影响**：避免运行时配置错误

### 8.2 测试覆盖完善

#### 8.2.1 稀疏向量正确性测试
- 验证indices和values长度一致
- 确保values都是正值
- 检查indices在词汇表范围内

#### 8.2.2 错误处理测试
- 未训练模型调用异常
- 空语料库处理
- 配置参数边界测试

#### 8.2.3 增量更新测试
- 词汇表大小变化验证
- 新词汇添加验证
- 模型一致性验证

### 8.3 性能优化建议

#### 8.3.1 词汇表优化
- 设置合理的最小频率阈值（建议5-10）
- 限制最大词汇表大小（建议5000-10000）
- 定期清理低频词汇

#### 8.3.2 计算优化
- 缓存常用词汇的IDF值
- 批量处理稀疏向量生成
- 异步更新BM25模型

#### 8.3.3 存储优化
- 压缩词汇表存储格式
- 增量保存模型变更
- 定期备份重要数据

## 9. 实施优先级建议

### 9.1 第一阶段（必须）
1. **修正BM25向量化逻辑**：确保sparse向量正确性
2. **添加词汇表剪枝**：控制维度规模
3. **明确Pinecone配置**：避免部署错误

### 9.2 第二阶段（重要）
1. **集成领域词典**：提升专有名词识别
2. **实现增量更新**：支持动态数据更新
3. **完善测试用例**：确保系统稳定性

### 9.3 第三阶段（可选）
1. **性能优化**：缓存和异步处理
2. **监控告警**：系统运行状态监控
3. **A/B测试**：效果对比验证


本设计文档基于ChatGPT的专业建议进行了全面修正，解决了以下关键问题：

1. **BM25Manager类**：修正了向量化逻辑，添加了剪枝和增量更新
2. **配置管理**：扩展了参数配置，支持领域词典
3. **上传端集成**：支持正确的sparse向量上传
4. **检索端集成**：支持修正后的hybrid检索
5. **测试验证**：完善的测试覆盖和错误处理
6. **部署指南**：明确的Pinecone配置要求

# 【2024-07-05工程建议补充】向量上传与BM25工程实践

## 1. 向量上传异常捕获与控制流（建议已采纳）
### 背景
- 单个chunk向量化或组装可能因数据异常、模型bug等失败，若未妥善捕获，可能导致整个批次或流程中断。
### 建议内容
- for循环内每个chunk的向量化、组装、append均应包裹在try...except块内，异常只影响当前chunk，日志详细记录。
### 落地方案
- vectors_to_upsert.append(vector_data)必须放在try块内，确保只append成功的向量。
- 失败chunk记录日志并continue，主流程不中断。

## 2. 批量与并发上传（建议采纳中）
### 背景
- Pinecone等向量库API有速率和批量限制，大规模上传时需分批、并发提升吞吐量。
### 建议内容
- upsert应分批（100-500条），可用异步/线程池提升并发。
- 上传前过滤掉dense为None或sparse维度异常的条目，保证批次数据质量。
### 落地方案
- 采用批量上传，后续可引入并发（如concurrent.futures.ThreadPoolExecutor）。
- 预过滤异常条目，提升整体成功率。

## 3. 模型训练与保存时机（建议采纳中）
### 背景
- BM25Manager如每次全量fit和save_model，性能和一致性均受影响，增量场景下需优化。
### 建议内容
- 启动时优先load已有模型，新增文档用incremental_update，不要频繁全量fit和save_model。
- 定期/批量save_model，减少磁盘IO。
### 落地方案
- 支持模型增量更新和定期保存，提升效率和一致性。

## 4. 内存与状态管理（建议可选）
### 背景
- 长时间运行时，BM25Manager的raw_texts等状态会无限增长，导致内存风险。
### 建议内容
- corpus_for_bm25/raw_texts应去重或滑窗，避免内存无限增长。
- 提供清理接口，支持按日期/版本清理旧状态。
### 落地方案
- 可实现滑动窗口或定期清理，保证内存可控。

## 5. Metadata和ID设计（建议采纳中）
### 背景
- chunk_id如仅用文件名+索引，文件变动会导致ID漂移，无法幂等覆盖。
### 建议内容
- chunk_id建议加内容哈希或版本号，保证幂等upsert，避免重复和脏数据。
### 落地方案
- 推荐chunk_id = f"{basename}-{i}-{sha1(chunk['text'].encode()).hexdigest()[:8]}"，唯一标识内容。

---


本设计文档基于ChatGPT的专业建议进行了全面修正，解决了以下关键问题：

1. **BM25Manager类**：修正了向量化逻辑，添加了剪枝和增量更新
2. **配置管理**：扩展了参数配置，支持领域词典
3. **上传端集成**：支持正确的sparse向量上传
4. **检索端集成**：支持修正后的hybrid检索
5. **测试验证**：完善的测试覆盖和错误处理
6. **部署指南**：明确的Pinecone配置要求

# 250702-BM25_hybrid检索实施设计文档


在现有的RAG系统中集成BM25 sparse embedding，实现完整的hybrid检索能力，提升战锤40K规则文档的检索召回效果。

- **配置管理**：统一BM25参数配置和中文分词设置

- **中文分词**：`jieba`库
- **向量数据库**：Pinecone（需确认metric=dotproduct）
- **Python版本**：3.8+


│   (User Query)  │───▶│  Dense + Sparse │───▶│  (Hybrid)       │

文档 → 切片 → Dense Embedding + BM25 Sparse → Pinecone Upsert

查询 → Dense Embedding + BM25 Sparse → Alpha加权 → Pinecone Query

## 3. BM25模块设计

    
    def __init__(self, corpus=None, k1=1.5, b=0.75):
        初始化BM25管理器
        
            corpus: 文档集合，用于构建词汇表和IDF
            k1: BM25参数k1（词频饱和参数）
            b: BM25参数b（文档长度归一化参数）
        self.vocab = {}
        self.idf = {}
        self.avgdl = 0
        
        if corpus:
            self.fit(corpus)
    
    def fit(self, corpus):
        # 中文分词处理
        tokenized_corpus = [self._tokenize_chinese(doc) for doc in corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        # 构建词汇表映射
        self._build_vocab_mapping()
    
    def _tokenize_chinese(self, text):
        """中文分词"""
    
    def _build_vocab_mapping(self):
        """构建词汇表到索引的映射"""
        # 实现词汇表映射逻辑
        pass
    
    def encode(self, text):
        将文本编码为sparse向量
        
            
            dict: {'indices': [int], 'values': [float]}
        # 实现BM25编码逻辑

### 3.2 配置管理
# config.py 新增配置
BM25_CONFIG = {
    'k1': 1.5,           # BM25参数k1
    'b': 0.75,           # BM25参数b
    'min_idf': 0.1,      # 最小IDF阈值
    'max_features': 10000,  # 最大特征数
    'chinese_tokenizer': 'jieba'  # 中文分词器

# 中文分词配置
JIEBA_CONFIG = {
    'use_paddle': True,  # 使用paddle模式提高准确率
    'cut_all': False,    # 精确模式
    'HMM': True         # 使用HMM模型

## 4. 上传端实施设计

### 4.1 UpsertManager修改

#### 4.1.1 初始化修改
        # 现有初始化代码...
        
        # 新增BM25管理器
        self.corpus_for_bm25 = []
        
        
    
        
        # 收集所有chunk文本用于训练BM25
        if not self.corpus_for_bm25:
            logging.warning("BM25训练语料为空，将使用默认配置")
            self.bm25_manager = BM25Manager()
            self.bm25_manager = BM25Manager(corpus=self.corpus_for_bm25)
            logging.info(f"BM25模型训练完成，语料大小: {len(self.corpus_for_bm25)}")

#### 4.1.2 向量生成修改
def _generate_vectors_for_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
    为单个chunk生成dense和sparse向量
    
        chunk: 切片数据
        
        包含dense和sparse向量的字典
        # 1. 生成dense embedding
        
        # 2. 生成sparse embedding
        sparse_embedding = self.bm25_manager.encode(chunk['text'])
        
            'dense': dense_embedding,
            'sparse': sparse_embedding,
            'success': True
        logging.error(f"向量生成失败: {e}")
            'dense': None,
            'sparse': None,
            'success': False,
            'error': str(e)

#### 4.1.3 上传数据结构修改
def _build_upsert_data(self, chunk: Dict[str, Any], vectors: Dict[str, Any]) -> Dict[str, Any]:
    构建Pinecone upsert数据结构
    
        vectors: 生成的向量数据
        
        Pinecone upsert数据
    chunk_id = f"{chunk.get('source_file', 'unknown')}-{chunk.get('chunk_id', '0')}"
    
    metadata = {
        'faction': chunk.get('faction', '未知'),
        'chunk_id': str(chunk.get('chunk_id', '0')),
        'source_file': chunk.get('source_file', '')
    
        metadata[f'h{j}'] = chunk.get(f'h{j}', '')
    
    # 构建upsert数据
    upsert_data = {
        'values': vectors['dense'],
        'metadata': metadata
    
    # 添加sparse向量（如果生成成功）
    if vectors['sparse'] and vectors['success']:
        upsert_data['sparse_values'] = vectors['sparse']
    
    return upsert_data

### 4.2 批量处理优化

#### 4.2.1 语料收集
def _collect_corpus_for_bm25(self, chunks: List[Dict[str, Any]]):
    """收集BM25训练语料"""
    corpus = []
    for chunk in chunks:
        # 添加chunk文本
        corpus.append(chunk['text'])
        
        # 可选：添加标题信息
        if chunk.get('section_heading'):
            corpus.append(chunk['section_heading'])
    
    self.corpus_for_bm25.extend(corpus)
    logging.info(f"收集BM25语料: {len(corpus)} 个文档片段")

#### 4.2.2 批量向量生成
def _batch_generate_vectors(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量生成向量"""
    vectors_list = []
    
        logging.debug(f"生成向量 {i+1}/{len(chunks)}")
        vectors = self._generate_vectors_for_chunk(chunk)
        vectors_list.append(vectors)
        
        # 进度显示
        if (i + 1) % 10 == 0:
            logging.info(f"向量生成进度: {i+1}/{len(chunks)}")
    
    return vectors_list

## 5. 检索端实施设计

### 5.1 VectorSearch类修改

#### 5.1.1 初始化修改
cl 