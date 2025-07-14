#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGAS集成测试
验证RAGAS评估器能够正常工作，包括数据准备和指标评估
"""

import os
import sys
import tempfile
import pandas as pd
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaltool.ragas_evaluator import RAGASEvaluator, run_ragas_evaluation
from vector_search import VectorSearch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """创建测试数据"""
    # 创建测试查询
    queries = [
        "战锤40K中什么是AP值？",
        "幽冥骑士的阔步巨兽技能如何使用？",
        "地形模型对移动有什么影响？"
    ]
    
    # 创建标准答案
    gold_answers = [
        "AP值是装甲穿透值，用于降低目标的装甲保护",
        "阔步巨兽技能允许单位在移动阶段移动额外的距离",
        "地形模型会影响单位的移动距离和移动方式"
    ]
    
    return queries, gold_answers

def test_ragas_evaluator_initialization():
    """测试RAGAS评估器初始化"""
    logger.info("测试RAGAS评估器初始化...")
    
    try:
        from config import PINECONE_API_KEY, PINECONE_INDEX
        vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX
        )
        
        evaluator = RAGASEvaluator(vector_search)
        
        # 验证embedding配置
        assert hasattr(evaluator, 'ragas_embeddings'), "RAGAS评估器应该有ragas_embeddings属性"
        assert hasattr(evaluator, 'embedding_model'), "RAGAS评估器应该有embedding_model属性"
        
        # 测试embedding功能
        test_embedding = evaluator.ragas_embeddings.embed_query("测试文本")
        assert len(test_embedding) == 1024, f"embedding维度应该是1024，实际是{len(test_embedding)}"
        
        logger.info("✅ RAGAS评估器初始化测试通过")
        return evaluator
        
    except Exception as e:
        logger.error(f"RAGAS评估器初始化测试失败: {e}")
        raise

def test_dataset_preparation(evaluator):
    """测试数据集准备"""
    logger.info("测试数据集准备...")
    
    try:
        queries, gold_answers = create_test_data()
        
        # 准备数据集
        dataset = evaluator.prepare_dataset(queries, gold_answers, top_k=3)
        
        # 验证数据集结构
        assert len(dataset) == len(queries), f"数据集长度应该与查询数量一致"
        assert 'question' in dataset.column_names, "数据集应该包含question列"
        assert 'contexts' in dataset.column_names, "数据集应该包含contexts列"
        assert 'answer' in dataset.column_names, "数据集应该包含answer列"
        assert 'ground_truth' in dataset.column_names, "数据集应该包含ground_truth列"
        
        # 验证数据内容
        for i in range(len(dataset)):
            assert dataset[i]['question'] == queries[i], f"第{i}个问题不匹配"
            assert dataset[i]['ground_truth'] == gold_answers[i], f"第{i}个标准答案不匹配"
            assert isinstance(dataset[i]['contexts'], list), f"第{i}个contexts应该是列表"
            assert isinstance(dataset[i]['answer'], str), f"第{i}个answer应该是字符串"
        
        logger.info("✅ 数据集准备测试通过")
        return dataset
        
    except Exception as e:
        logger.error(f"数据集准备测试失败: {e}")
        raise

def test_metrics_evaluation(evaluator, dataset):
    """测试指标评估"""
    logger.info("测试指标评估...")
    
    try:
        # 测试检索指标
        logger.info("测试检索指标...")
        retrieval_results = evaluator.evaluate_retrieval_metrics(dataset)
        assert isinstance(retrieval_results, dict), "检索指标结果应该是字典"
        assert 'context_relevance' in retrieval_results, "应该包含context_relevance指标"
        assert 'context_recall' in retrieval_results, "应该包含context_recall指标"
        
        # 验证指标值范围
        for metric, value in retrieval_results.items():
            assert 0 <= value <= 1, f"指标{metric}的值应该在0-1之间，实际是{value}"
        
        logger.info(f"检索指标结果: {retrieval_results}")
        
        # 测试生成指标
        logger.info("测试生成指标...")
        generation_results = evaluator.evaluate_generation_metrics(dataset)
        assert isinstance(generation_results, dict), "生成指标结果应该是字典"
        assert 'faithfulness' in generation_results, "应该包含faithfulness指标"
        assert 'answer_relevancy' in generation_results, "应该包含answer_relevancy指标"
        assert 'answer_correctness' in generation_results, "应该包含answer_correctness指标"
        
        # 验证指标值范围
        for metric, value in generation_results.items():
            assert 0 <= value <= 1, f"指标{metric}的值应该在0-1之间，实际是{value}"
        
        logger.info(f"生成指标结果: {generation_results}")
        
        # 测试高级指标
        logger.info("测试高级指标...")
        advanced_results = evaluator.evaluate_advanced_metrics(dataset)
        assert isinstance(advanced_results, dict), "高级指标结果应该是字典"
        
        # 验证指标值范围
        for metric, value in advanced_results.items():
            assert 0 <= value <= 1, f"指标{metric}的值应该在0-1之间，实际是{value}"
        
        logger.info(f"高级指标结果: {advanced_results}")
        
        logger.info("✅ 指标评估测试通过")
        return retrieval_results, generation_results, advanced_results
        
    except Exception as e:
        logger.error(f"指标评估测试失败: {e}")
        raise

def test_report_generation(evaluator, dataset, results):
    """测试报告生成"""
    logger.info("测试报告生成...")
    
    try:
        # 合并所有结果
        all_results = {}
        all_results.update(results[0])  # retrieval_results
        all_results.update(results[1])  # generation_results
        all_results.update(results[2])  # advanced_results
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 生成报告
            report_path = evaluator.generate_detailed_report(dataset, all_results, temp_dir)
            
            # 验证报告文件存在
            assert os.path.exists(report_path), f"报告文件应该存在: {report_path}"
            
            # 验证报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
                assert "RAGAS Dense向量检索评估报告" in report_content, "报告应该包含标题"
                assert "评估指标结果" in report_content, "报告应该包含指标结果"
                assert "详细分析" in report_content, "报告应该包含详细分析"
            
            # 保存数据集详情
            details_path = evaluator.save_dataset_details(dataset, temp_dir)
            assert os.path.exists(details_path), f"详情文件应该存在: {details_path}"
            
            # 验证详情文件格式
            import json
            with open(details_path, 'r', encoding='utf-8') as f:
                details = json.load(f)
                assert isinstance(details, list), "详情应该是列表格式"
                assert len(details) == len(dataset), "详情长度应该与数据集一致"
        
        logger.info("✅ 报告生成测试通过")
        
    except Exception as e:
        logger.error(f"报告生成测试失败: {e}")
        raise

def test_full_evaluation_pipeline():
    """测试完整评估流程"""
    logger.info("测试完整评估流程...")
    
    try:
        # 创建测试数据文件
        queries, gold_answers = create_test_data()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建查询文件
            query_file = os.path.join(temp_dir, "test_queries.csv")
            queries_df = pd.DataFrame({"query": queries})
            queries_df.to_csv(query_file, index=False)
            
            # 创建标准答案文件
            gold_file = os.path.join(temp_dir, "test_answers.csv")
            gold_df = pd.DataFrame({
                "query": queries,
                "answer": gold_answers
            })
            gold_df.to_csv(gold_file, index=False)
            
            # 运行完整评估
            result = run_ragas_evaluation(
                query_file=query_file,
                gold_file=gold_file,
                top_k=3,
                output_dir=temp_dir
            )
            
            # 验证结果结构
            assert isinstance(result, dict), "评估结果应该是字典"
            assert 'metrics' in result, "结果应该包含metrics"
            assert 'report_path' in result, "结果应该包含report_path"
            assert 'details_path' in result, "结果应该包含details_path"
            assert 'sample_count' in result, "结果应该包含sample_count"
            assert 'evaluation_time' in result, "结果应该包含evaluation_time"
            
            # 验证指标
            metrics = result['metrics']
            assert isinstance(metrics, dict), "指标应该是字典"
            assert len(metrics) > 0, "指标不应该为空"
            
            # 验证文件存在
            assert os.path.exists(result['report_path']), f"报告文件应该存在: {result['report_path']}"
            assert os.path.exists(result['details_path']), f"详情文件应该存在: {result['details_path']}"
            
            logger.info(f"完整评估结果: {result}")
        
        logger.info("✅ 完整评估流程测试通过")
        
    except Exception as e:
        logger.error(f"完整评估流程测试失败: {e}")
        raise

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RAGAS集成测试")
    logger.info("=" * 60)
    
    try:
        # 测试1: RAGAS评估器初始化
        evaluator = test_ragas_evaluator_initialization()
        
        # 测试2: 数据集准备
        dataset = test_dataset_preparation(evaluator)
        
        # 测试3: 指标评估
        results = test_metrics_evaluation(evaluator, dataset)
        
        # 测试4: 报告生成
        test_report_generation(evaluator, dataset, results)
        
        # 测试5: 完整评估流程
        test_full_evaluation_pipeline()
        
        logger.info("=" * 60)
        logger.info("🎉 所有集成测试通过！")
        logger.info("✅ RAGAS评估器配置正确")
        logger.info("✅ 数据集准备功能正常")
        logger.info("✅ 指标评估功能正常")
        logger.info("✅ 报告生成功能正常")
        logger.info("✅ 完整评估流程正常")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ 集成测试失败: {e}")
        logger.error("=" * 60)
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        sys.exit(1) 