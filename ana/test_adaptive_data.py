#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OpenSearchConnection.adaptive_data方法的脚本
使用conn_factory生成osconn实例，处理test_chunk.json数据
"""

import json
import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from conn.conn_factory import ConnectionFactory
from config import get_qwen_embedding_model
from dataupload.bm25_manager import BM25Manager
from config import BM25_K1, BM25_B, BM25_MIN_FREQ, BM25_MAX_VOCAB_SIZE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数：测试adaptive_data方法"""
    logger.info("开始测试OpenSearchConnection.adaptive_data方法")
    
    try:
        # 1. 使用conn_factory生成OpenSearch连接实例
        logger.info("创建OpenSearch连接实例...")
        os_conn = ConnectionFactory.create_conn("opensearch")
        logger.info("✅ OpenSearch连接实例创建成功")
        
        # 2. 获取QWEN embedding模型
        logger.info("获取QWEN embedding模型...")
        qwen_model = get_qwen_embedding_model()
        logger.info("✅ QWEN embedding模型获取成功")
        
        # 3. 初始化BM25Manager
        logger.info("初始化BM25Manager...")
        userdict_path = project_root / "dict" / "bm25_vocab_freq_2508082124.json"
        model_path = project_root / "models" / "bm25_model.pkl"
        
        # 检查文件是否存在
        if not userdict_path.exists():
            raise FileNotFoundError(f"词典文件不存在: {userdict_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"BM25模型文件不存在: {model_path}")
        
        # 创建BM25Manager实例
        bm25_manager = BM25Manager(
            k1=BM25_K1,
            b=BM25_B,
            min_freq=BM25_MIN_FREQ,
            max_vocab_size=BM25_MAX_VOCAB_SIZE,
            user_dict_path=str(userdict_path)
        )
        
        # 加载模型和词汇表
        bm25_manager.load_model(str(model_path), str(userdict_path))
        logger.info("✅ BM25Manager初始化成功")
        
        # 4. 读取测试数据
        logger.info("读取测试数据...")
        test_chunk_path = project_root / "ana" / "test_chunk.json"
        with open(test_chunk_path, 'r', encoding='utf-8') as f:
            test_chunk = json.load(f)
        logger.info("✅ 测试数据读取成功")
        
        # 5. 调用adaptive_data方法
        logger.info("调用adaptive_data方法...")
        chunk_id = "test_chunk_001"
        
        result = os_conn.adaptive_data(
            chunk=test_chunk,
            chunk_id=chunk_id,
            embedding_model=qwen_model,
            bm25_manager=bm25_manager
        )
        
        logger.info("✅ adaptive_data方法调用成功")
        logger.info(f"返回结果类型: {type(result)}")
        logger.info(f"返回结果键: {list(result.keys())}")
        
        # 6. 保存输出结果
        logger.info("保存输出结果...")
        output_path = project_root / "ana" / "output.json"
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 输出结果已保存到: {output_path}")
        
        # 7. 输出结果摘要
        logger.info("=== 结果摘要 ===")
        logger.info(f"文档ID: {result.get('_id')}")
        logger.info(f"文本长度: {len(result.get('text', ''))}")
        logger.info(f"Embedding向量维度: {len(result.get('embedding', []))}")
        logger.info(f"Sparse向量特征数: {len(result.get('sparse', {}))}")
        logger.info(f"Chunk类型: {result.get('chunk_type')}")
        logger.info(f"内容类型: {result.get('content_type')}")
        logger.info(f"阵营: {result.get('faction')}")
        logger.info(f"章节标题: {result.get('section_heading')}")
        
        logger.info("🎉 测试完成！")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    main() 