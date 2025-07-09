#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON配置加载器
用于加载和解析评估项词典
"""

import json
import os
from typing import Dict, List, Optional


class EvaluatorConfigLoader:
    """评估器配置加载器"""
    
    def __init__(self, config_file: str = "evaluator_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_path = os.path.join(os.path.dirname(__file__), self.config_file)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_evaluator_config(self, evaluator_name: str) -> Optional[Dict]:
        """获取指定评估器的配置"""
        return self.config.get('evaluators', {}).get(evaluator_name)
    
    def get_all_evaluators(self) -> List[str]:
        """获取所有评估器名称"""
        return list(self.config.get('evaluators', {}).keys())
    
    def get_evaluation_groups(self) -> Dict:
        """获取评估组配置"""
        return self.config.get('evaluation_groups', {})
    
    def can_evaluate(self, evaluator_name: str, args) -> bool:
        """检查评估器是否可以执行"""
        evaluator_config = self.get_evaluator_config(evaluator_name)
        if not evaluator_config:
            return False
        
        conditions = evaluator_config.get('conditions', {})
        
        # 检查是否总是执行
        if conditions.get('always_execute', False):
            return True
        
        # 检查是否需要gold文件
        if conditions.get('requires_gold_file', False):
            if not args.gold or not os.path.exists(args.gold):
                return False
        
        # 检查是否需要特定的检索引擎
        required_engine = conditions.get('requires_search_engine')
        if required_engine and args.se != required_engine:
            return False
        
        return True
    
    def get_available_evaluators(self, args) -> List[str]:
        """根据参数获取可执行的评估器"""
        available_evaluators = []
        
        for evaluator_name in self.get_all_evaluators():
            if self.can_evaluate(evaluator_name, args):
                available_evaluators.append(evaluator_name)
        
        return available_evaluators
    
    def get_evaluators_by_groups(self, args) -> List[str]:
        """根据评估组获取可执行的评估器"""
        available_evaluators = []
        groups = self.get_evaluation_groups()
        
        for group_name, group_config in groups.items():
            group_conditions = group_config.get('conditions', {})
            
            # 检查组条件
            can_execute_group = True
            
            if group_conditions.get('requires_gold_file', False):
                if not args.gold or not os.path.exists(args.gold):
                    can_execute_group = False
            
            required_engine = group_conditions.get('requires_search_engine')
            if required_engine and args.se != required_engine:
                can_execute_group = False
            
            if can_execute_group:
                group_evaluators = group_config.get('evaluators', [])
                for evaluator_name in group_evaluators:
                    if self.can_evaluate(evaluator_name, args):
                        available_evaluators.append(evaluator_name)
        
        return available_evaluators
    
    def get_visualization_config(self, evaluator_name: str) -> Optional[Dict]:
        """获取评估器的可视化配置"""
        evaluator_config = self.get_evaluator_config(evaluator_name)
        if evaluator_config:
            return evaluator_config.get('visualization')
        return None
    
    def get_evaluator_description(self, evaluator_name: str) -> str:
        """获取评估器描述"""
        evaluator_config = self.get_evaluator_config(evaluator_name)
        if evaluator_config:
            return evaluator_config.get('description', '')
        return ''
    
    def print_evaluator_info(self, args):
        """打印评估器信息"""
        print("=== 评估项词典信息 ===")
        print(f"配置文件: {self.config_file}")
        print(f"总评估器数量: {len(self.get_all_evaluators())}")
        print()
        
        print("所有评估器:")
        for evaluator_name in self.get_all_evaluators():
            config = self.get_evaluator_config(evaluator_name)
            can_execute = self.can_evaluate(evaluator_name, args)
            status = "✓" if can_execute else "✗"
            print(f"  {status} {evaluator_name}: {config.get('description', '')}")
        
        print()
        print("可执行的评估器:")
        available = self.get_available_evaluators(args)
        for evaluator_name in available:
            description = self.get_evaluator_description(evaluator_name)
            print(f"  - {evaluator_name}: {description}")


# 全局配置加载器实例
config_loader = EvaluatorConfigLoader() 