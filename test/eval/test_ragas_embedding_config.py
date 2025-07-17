#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS Embedding配置单元测试
验证是否正确使用text-embedding-3-large模型并输出1024维度
"""

import os
import sys
import unittest
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入base_test确保安全删除
from test.base_test import TestBase

from config import EMBADDING_MODEL, get_embedding_model
from evaltool.ragas_evaluator import RAGASEvaluator
from vector_search import VectorSearch
from datasets import Dataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRAGASEmbeddingConfig(TestBase):
    """RAGAS Embedding配置测试类"""
    
    def setUp(self):
        """测试前准备"""
        super().setUp()  # 调用TestBase的setUp
        
        # 初始化向量搜索（使用测试配置）
        from config import PINECONE_API_KEY, PINECONE_INDEX
        self.vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX
        )
        
        # 初始化RAGAS评估器
        self.evaluator = RAGASEvaluator(self.vector_search)
        
        # 测试文本
        self.test_text = "测试战锤40K规则查询"
    
    def test_embedding_model_info(self):
        """测试embedding模型基本信息"""
        logger.info("测试embedding模型信息...")
        
        # 获取embedding模型信息
        embedding_model = get_embedding_model()
        
        # 测试不同长度的文本
        test_texts = [
            "短文本",
            "这是一个中等长度的测试文本，用于验证embedding模型的稳定性",
            "这是一个较长的测试文本，包含更多的词汇和语义信息，用于测试embedding模型在处理复杂文本时的表现"
        ]
        
        for i, text in enumerate(test_texts, 1):
            embedding = embedding_model.embed_query(text)
            logger.info(f"文本{i}长度: {len(text)}字符, embedding维度: {len(embedding)}")
            
            # 验证维度一致性
            self.assertEqual(len(embedding), 1024, f"文本{i}的embedding维度应该是1024，实际是{len(embedding)}")
        
        logger.info("✅ embedding模型信息测试通过")
    
    def test_base_embedding_configuration(self):
        """测试基础embedding模型配置"""
        logger.info("测试基础embedding模型...")
        
        base_embedding_model = get_embedding_model()
        base_embedding = base_embedding_model.embed_query(self.test_text)
        
        logger.info(f"基础embedding模型: {EMBADDING_MODEL}")
        logger.info(f"基础embedding维度: {len(base_embedding)}")
        
        # 验证维度是否为1024
        self.assertEqual(len(base_embedding), 1024, f"基础embedding维度应该是1024，实际是{len(base_embedding)}")
        logger.info("✅ 基础embedding模型配置正确")
    
    def test_ragas_embedding_configuration(self):
        """测试RAGAS embedding配置"""
        logger.info("测试RAGAS embedding配置...")
        
        # 测试RAGAS embedding
        ragas_embedding = self.evaluator.ragas_embeddings.embed_query(self.test_text)
        logger.info(f"RAGAS embedding维度: {len(ragas_embedding)}")
        
        # 验证维度是否为1024
        self.assertEqual(len(ragas_embedding), 1024, f"RAGAS embedding维度应该是1024，实际是{len(ragas_embedding)}")
        logger.info("✅ RAGAS embedding配置正确")
    
    def test_embedding_consistency(self):
        """测试embedding一致性"""
        logger.info("验证embedding一致性...")
        
        # 获取基础embedding和RAGAS embedding
        base_embedding_model = get_embedding_model()
        base_embedding = base_embedding_model.embed_query(self.test_text)
        ragas_embedding = self.evaluator.ragas_embeddings.embed_query(self.test_text)
        
        # 计算两个embedding的相似度
        base_norm = np.linalg.norm(base_embedding)
        ragas_norm = np.linalg.norm(ragas_embedding)
        
        # 归一化
        base_normalized = np.array(base_embedding) / base_norm
        ragas_normalized = np.array(ragas_embedding) / ragas_norm
        
        # 计算余弦相似度
        similarity = np.dot(base_normalized, ragas_normalized)
        logger.info(f"基础embedding和RAGAS embedding相似度: {similarity:.6f}")
        
        # 验证相似度接近1.0（允许小的数值误差）
        self.assertGreater(similarity, 0.9999, f"embedding相似度应该接近1.0，实际是{similarity}")
        logger.info("✅ embedding一致性验证通过")
    
    def test_ragas_metrics_embedding_config(self):
        """测试RAGAS指标embedding配置"""
        logger.info("测试RAGAS指标配置...")
        
        # 测试AnswerSimilarity指标（需要embedding）
        from ragas.metrics import AnswerSimilarity
        
        # 创建测试数据集
        test_dataset = Dataset.from_dict({
            "question": ["测试问题"],
            "contexts": [["测试上下文"]],
            "answer": ["测试答案"],
            "ground_truth": ["标准答案"]
        })
        
        # 创建AnswerSimilarity指标并配置embedding
        answer_similarity_metric = AnswerSimilarity()
        answer_similarity_metric.embeddings = self.evaluator.ragas_embeddings
        
        # 验证embedding配置成功
        self.assertIsNotNone(answer_similarity_metric.embeddings)
        logger.info("✅ RAGAS指标embedding配置正确")
    
    def test_embedding_model_name(self):
        """测试embedding模型名称"""
        logger.info("测试embedding模型名称...")
        
        # 验证模型名称
        self.assertEqual(EMBADDING_MODEL, "text-embedding-3-large", 
                        f"embedding模型应该是text-embedding-3-large，实际是{EMBADDING_MODEL}")
        logger.info(f"✅ 使用模型: {EMBADDING_MODEL}")
    
    def test_embedding_dimension_consistency(self):
        """测试embedding维度一致性"""
        logger.info("测试embedding维度一致性...")
        
        # 测试多个文本的embedding维度
        test_texts = [
            "战锤40K",
            "帝国卫队",
            "幽冥骑士",
            "AP值是什么？",
            "地形模型对移动有什么影响？"
        ]
        
        embedding_model = get_embedding_model()
        
        for text in test_texts:
            embedding = embedding_model.embed_query(text)
            self.assertEqual(len(embedding), 1024, f"文本'{text}'的embedding维度应该是1024，实际是{len(embedding)}")
        
        logger.info("✅ embedding维度一致性测试通过")
    
    def test_ragas_evaluator_initialization(self):
        """测试RAGAS评估器初始化"""
        logger.info("测试RAGAS评估器初始化...")
        
        # 验证评估器正确初始化
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.evaluator.ragas_embeddings)
        self.assertEqual(self.evaluator.vector_search, self.vector_search)
        
        # 验证embedding模型配置
        test_embedding = self.evaluator.ragas_embeddings.embed_query("测试")
        self.assertEqual(len(test_embedding), 1024)
        
        logger.info("✅ RAGAS评估器初始化测试通过")


if __name__ == '__main__':
    unittest.main() 