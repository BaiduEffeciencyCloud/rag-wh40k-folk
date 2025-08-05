#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的意图预测接口
封装特征提取和预测逻辑，提供统一的 predict() 方法
"""

import os
import sys
import yaml
import json
import argparse
import joblib
import numpy as np
from typing import Dict, Any, List

from .model.intent_classifier import IntentClassifier
from .feature.factory import FeatureExtractorFactory


class IntentPredictor:
    """统一的意图预测接口"""
    
    def __init__(self, config_path: str, feature_path: str, model_path: str):
        """
        初始化预测器
        
        Args:
            config_path: 配置文件路径
            feature_path: 特征提取器路径
            model_path: 模型文件路径
        """
        self.config_path = config_path
        self.feature_path = feature_path
        self.model_path = model_path
        self.config = None
        self.feature_extractor = None
        self.intent_classifier = None
        self.use_enhanced = False
        self._load_models()
    
    @classmethod
    def from_models_dir(cls, models_dir: str = "models"):
        """
        从导出的 models 目录加载（用于 adp.py）
        
        Args:
            models_dir: models 目录路径
            
        Returns:
            IntentPredictor 实例
        """
        config_path = os.path.join(models_dir, "config.yaml")
        feature_path = os.path.join(models_dir, "intent_tfidf.pkl")
        model_path = os.path.join(models_dir, "intent_classifier.pkl")
        return cls(config_path, feature_path, model_path)
    
    @classmethod
    def from_training_output(cls, version_path: str):
        """
        从训练输出目录加载（用于 run_evaluation.py）
        
        Args:
            version_path: 训练输出目录路径
            
        Returns:
            IntentPredictor 实例
        """
        config_path = os.path.join(version_path, "config", "config.yaml")
        feature_path = os.path.join(version_path, "feature", "intent_tfidf.pkl")
        model_path = os.path.join(version_path, "model", "intent_classifier.pkl")
        return cls(config_path, feature_path, model_path)
    
    def predict(self, query: str) -> Dict[str, Any]:
        """
        统一的预测接口
        
        Args:
            query: 输入查询
            
        Returns:
            预测结果字典，包含 intent, confidence, probabilities
        """
        # 特征提取
        features = self._extract_features(query)
        
        # 意图预测
        intent_probabilities = self.intent_classifier.predict_proba(features)[0]
        max_prob_idx = np.argmax(intent_probabilities)
        intent_prediction = self.intent_classifier.classes_[max_prob_idx]
        max_confidence = np.max(intent_probabilities)
        
        return {
            "intent": intent_prediction,
            "confidence": max_confidence,
            "probabilities": intent_probabilities.tolist()
        }
    
    def _load_models(self):
        """加载模型和特征提取器"""
        # 1. 加载配置文件
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 2. 使用工厂方法创建特征提取器
        self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(self.config)
        
        # 3. 确定是否使用增强特征
        self.use_enhanced = FeatureExtractorFactory.is_enhanced_feature_extractor(self.config)
        
        # 4. 加载特征提取器
        if self.use_enhanced:
            self.feature_extractor.load_enhanced_feature_extractor(self.feature_path)
        else:
            self.feature_extractor.load(self.feature_path)
        
        # 5. 加载分类器
        self.intent_classifier = IntentClassifier(self.config)
        self.intent_classifier.load_model(self.model_path)
    
    def _extract_features(self, query: str) -> np.ndarray:
        """内部特征提取，对外透明"""
        if self.use_enhanced:
            # 使用增强特征（与评估时相同）
            enhanced_features = self.feature_extractor.get_enhanced_intent_features(query)
            return enhanced_features.flatten().reshape(1, -1)
        else:
            # 使用基础特征（与评估时相同）
            return self.feature_extractor.transform([query])


def find_latest_version():
    """找到最新的模型版本"""
    model_output_dir = "intent_training/model_output"
    if not os.path.exists(model_output_dir):
        raise FileNotFoundError(f"模型输出目录不存在: {model_output_dir}")
    
    # 获取所有时间戳目录
    versions = [d for d in os.listdir(model_output_dir) 
               if os.path.isdir(os.path.join(model_output_dir, d)) and d.isdigit()]
    
    if not versions:
        raise FileNotFoundError("没有找到任何模型版本")
    
    # 按时间戳排序，返回最新的
    latest_version = sorted(versions)[-1]
    return os.path.join(model_output_dir, latest_version)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统一的意图预测接口")
    parser.add_argument("--query", required=True, help="用户查询")
    parser.add_argument("--mode", choices=["eva", "online"], required=True, help="使用模式")
    parser.add_argument("--ver", help="模型版本 (head 或时间戳)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "online":
            # 使用 models/ 目录
            predictor = IntentPredictor.from_models_dir("models")
        elif args.mode == "eva":
            if not args.ver:
                raise ValueError("评估模式需要指定 --ver 参数")
            
            if args.ver == "head":
                # 找到最新时间戳
                version_path = find_latest_version()
            else:
                # 使用指定时间戳
                version_path = f"intent_training/model_output/{args.ver}"
            
            predictor = IntentPredictor.from_training_output(version_path)
        
        # 执行预测
        result = predictor.predict(args.query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 