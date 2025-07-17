#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qProcessor模块测试类
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qProcessor import QueryProcessorFactory, QueryProcessorInterface

class TestStraightforwardProcessor(unittest.TestCase):
    """测试Straightforward处理器"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.processor = QueryProcessorFactory.create_processor("straightforward")
        self.test_query = "阿巴顿的黑色远征"
    
    def test_processor_creation(self):
        """测试处理器创建"""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor, QueryProcessorInterface)
    
    def test_processor_type(self):
        """测试处理器类型"""
        self.assertEqual(self.processor.get_type(), "straightforward")
    
    def test_is_multi_query(self):
        """测试是否多查询"""
        self.assertFalse(self.processor.is_multi_query())
    
    def test_process_single_query(self):
        """测试单查询处理"""
        result = self.processor.process(self.test_query)
        
        # 验证返回类型
        self.assertIsInstance(result, dict)
        
        # 验证返回内容
        self.assertEqual(result['text'], self.test_query)
        self.assertEqual(result['type'], 'straightforward')
        self.assertTrue(result['processed'])
        self.assertEqual(result['original_query'], self.test_query)

class TestExpanderProcessor(unittest.TestCase):
    """测试Expander处理器"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.processor = QueryProcessorFactory.create_processor("expander")
        self.test_query = "阿巴顿的黑色远征"
    
    def test_processor_creation(self):
        """测试处理器创建"""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor, QueryProcessorInterface)
    
    def test_processor_type(self):
        """测试处理器类型"""
        self.assertEqual(self.processor.get_type(), "expander")
    
    def test_is_multi_query(self):
        """测试是否多查询"""
        self.assertTrue(self.processor.is_multi_query())
    
    def test_process_multiple_queries(self):
        """测试多查询处理"""
        results = self.processor.process(self.test_query)
        
        # 验证返回类型
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 1)  # 应该生成多个查询
        
        # 验证每个结果的结构
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
            self.assertIn('type', result)
            self.assertIn('processed', result)
            self.assertIn('original_query', result)
            self.assertEqual(result['type'], 'expanded')
            self.assertTrue(result['processed'])
            self.assertEqual(result['original_query'], self.test_query)
    
    def test_expansion_methods(self):
        """测试扩写方法"""
        results = self.processor.process(self.test_query)
        
        # 检查是否包含不同的扩写方法
        methods = [result.get('expansion_method', 'unknown') for result in results]
        self.assertIn('original', methods)
        self.assertIn('_add_related_terms', methods)
        self.assertIn('_add_synonyms', methods)
        self.assertIn('_add_context_terms', methods)

class TestCOTProcessor(unittest.TestCase):
    """测试COT处理器"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.processor = QueryProcessorFactory.create_processor("cot")
        self.test_query = "阿巴顿的黑色远征有哪些关键事件？"
    
    def test_processor_creation(self):
        """测试处理器创建"""
        self.assertIsNotNone(self.processor)
        self.assertIsInstance(self.processor, QueryProcessorInterface)
    
    def test_processor_type(self):
        """测试处理器类型"""
        self.assertEqual(self.processor.get_type(), "cot")
    
    def test_is_multi_query(self):
        """测试是否多查询"""
        self.assertTrue(self.processor.is_multi_query())
    
    def test_process_cot_steps(self):
        """测试思维链步骤处理"""
        results = self.processor.process(self.test_query)
        
        # 验证返回类型
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 1)  # 应该生成多个步骤
        
        # 验证每个结果的结构
        for i, result in enumerate(results):
            self.assertIsInstance(result, dict)
            self.assertIn('text', result)
            self.assertIn('type', result)
            self.assertIn('processed', result)
            self.assertIn('original_query', result)
            self.assertIn('step', result)
            self.assertIn('step_description', result)
            self.assertEqual(result['type'], 'cot_step')
            self.assertTrue(result['processed'])
            self.assertEqual(result['original_query'], self.test_query)
            self.assertEqual(result['step'], i)
    
    def test_cot_step_sequence(self):
        """测试思维链步骤序列"""
        results = self.processor.process(self.test_query)
        
        # 验证步骤数量（通常是3步）
        self.assertEqual(len(results), 3)
        
        # 验证步骤描述
        step_descriptions = [result['step_description'] for result in results]
        self.assertIn("第一步：理解事件背景和历史", step_descriptions)
        self.assertIn("第二步：分析事件的关键参与者和地点", step_descriptions)
        self.assertIn("第三步：总结事件的影响和结果", step_descriptions)

class TestQueryProcessorFactory(unittest.TestCase):
    """测试查询处理器工厂类"""
    
    def test_create_processor(self):
        """测试处理器创建"""
        # 测试所有可用的处理器类型
        processor_types = ["straightforward", "origin", "expander", "cot"]
        
        for processor_type in processor_types:
            processor = QueryProcessorFactory.create_processor(processor_type)
            self.assertIsNotNone(processor)
            self.assertIsInstance(processor, QueryProcessorInterface)
    
    def test_create_unknown_processor(self):
        """测试创建未知处理器"""
        processor = QueryProcessorFactory.create_processor("unknown")
        self.assertIsNotNone(processor)
        # 应该返回默认的straightforward处理器
        self.assertEqual(processor.get_type(), "straightforward")
    
    def test_get_available_processors(self):
        """测试获取可用处理器列表"""
        processors = QueryProcessorFactory.get_available_processors()
        self.assertIsInstance(processors, list)
        self.assertIn("straightforward", processors)
        self.assertIn("origin", processors)
        self.assertIn("expander", processors)
        self.assertIn("cot", processors)
    
    def test_validate_processor_type(self):
        """测试处理器类型验证"""
        # 测试有效类型
        self.assertTrue(QueryProcessorFactory.validate_processor_type("expander"))
        self.assertTrue(QueryProcessorFactory.validate_processor_type("cot"))
        
        # 测试无效类型
        self.assertFalse(QueryProcessorFactory.validate_processor_type("unknown"))
        self.assertFalse(QueryProcessorFactory.validate_processor_type("invalid"))
    
    def test_get_processor_info(self):
        """测试获取处理器信息"""
        info = QueryProcessorFactory.get_processor_info("expander")
        self.assertIsInstance(info, dict)
        self.assertIn('type', info)
        self.assertIn('is_multi_query', info)
        self.assertIn('description', info)
        self.assertEqual(info['type'], 'expander')
        self.assertTrue(info['is_multi_query'])
    
    def test_unified_interface(self):
        """测试统一接口"""
        query = "阿巴顿的黑色远征"
        
        # 测试单查询处理器
        result = QueryProcessorFactory.process_query("straightforward", query)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['text'], query)
        
        # 测试多查询处理器
        results = QueryProcessorFactory.process_query("expander", query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 1)

if __name__ == "__main__":
    unittest.main() 