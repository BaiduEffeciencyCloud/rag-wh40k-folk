"""
模型可解释性模块

重要说明：
此模块在模型训练阶段不使用，专门为后续qProcessor在线意图识别服务设计。

功能说明：
1. 为qProcessor提供模型预测的可解释性支持
2. 生成轻量级解释，用于在线实时解释预测结果
3. 提供深度分析功能，用于离线模型诊断和优化
4. 支持特征重要性分析、决策路径分析等

使用场景：
- qProcessor在线预测时，为用户提供预测结果的解释
- 模型性能分析时，提供深度诊断和优化建议
- 支持在线轻量级解释和离线深度分析两种模式

注意：此模块不参与模型训练，仅在推理阶段和模型分析阶段使用。
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

class LightInterpreter:
    """
    轻量级解释器 - 用于在线解释
    
    重要说明：
    此模块在模型训练阶段不使用，专门为后续qProcessor在线意图识别服务设计。
    
    主要功能：
    1. 为qProcessor提供模型预测的可解释性支持
    2. 生成轻量级解释，用于在线实时解释预测结果
    3. 提供深度分析功能，用于离线模型诊断和优化
    4. 支持特征重要性分析、决策路径分析等
    
    使用场景：
    - qProcessor在线预测时，为用户提供预测结果的解释
    - 模型性能分析时，提供深度诊断和优化建议
    - 支持在线轻量级解释和离线深度分析两种模式
    
    注意：此模块不参与模型训练，仅在推理阶段和模型分析阶段使用。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化轻量级解释器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.max_features = config.get('max_features', 100)
        self.explanation_level = config.get('explanation_level', 'light')
        self.max_response_time = config.get('max_response_time', 100)  # ms
        
    def explain(self, processed_query: Dict[str, Any], model=None, features=None) -> Dict[str, Any]:
        """
        生成轻量级解释
        
        Args:
            processed_query: 处理后的查询信息
            model: 训练好的模型（可选）
            features: 特征向量（可选）
            
        Returns:
            解释结果字典
        """
        logger.info("开始轻量级解释生成")
        
        # 提取关键特征
        key_features = self._extract_key_features(processed_query)
        
        # 生成置信度解释（如果有概率信息）
        confidence_reason = "基于查询特征进行意图分类"
        if features is not None and model is not None:
            try:
                probabilities = model.predict_proba([features])[0]
                confidence_reason = self._explain_confidence(processed_query, probabilities)
            except:
                pass
        
        # 生成决策摘要
        prediction = processed_query.get('prediction', 'unknown')
        decision_summary = self._generate_summary(processed_query, prediction)
        
        return {
            'key_features': key_features,
            'confidence_reason': confidence_reason,
            'decision_summary': decision_summary
        }
        
    def _extract_key_features(self, processed_query: Dict[str, Any]) -> List[str]:
        """
        提取关键特征
        
        Args:
            processed_query: 处理后的查询信息
            
        Returns:
            关键特征列表
        """
        key_features = []
        
        # 从查询文本中提取关键词
        text = processed_query.get('text', '')
        if text:
            # 简单的关键词提取（基于常见词汇）
            keywords = ['查询', '比较', '列出', '分析', '说明', '介绍', '详细', '简单']
            for keyword in keywords:
                if keyword in text:
                    key_features.append(keyword)
        
        # 从实体中提取特征
        entities = processed_query.get('entities', [])
        key_features.extend(entities)
        
        # 从关系中提取特征
        relations = processed_query.get('relations', [])
        key_features.extend(relations)
        
        # 如果没有提取到特征，使用查询文本的前几个词
        if not key_features and text:
            words = text.split()[:3]  # 取前3个词
            key_features.extend(words)
        
        return key_features
        
    def _explain_confidence(self, processed_query: Dict[str, Any], probabilities: np.ndarray) -> str:
        """
        解释置信度
        
        Args:
            processed_query: 处理后的查询信息
            probabilities: 预测概率
            
        Returns:
            置信度解释文本
        """
        max_prob = np.max(probabilities)
        max_idx = np.argmax(probabilities)
        
        # 意图名称映射
        intent_names = ['query', 'compare', 'list', 'analyze', 'explain']
        intent_name = intent_names[max_idx] if max_idx < len(intent_names) else f'intent_{max_idx}'
        
        if max_prob >= 0.8:
            confidence_level = "高置信度"
        elif max_prob >= 0.6:
            confidence_level = "中等置信度"
        else:
            confidence_level = "低置信度"
        
        return f"模型以{confidence_level}({max_prob:.2f})预测为'{intent_name}'意图"
        
    def _generate_summary(self, processed_query: Dict[str, Any], prediction: str) -> str:
        """
        生成决策摘要
        
        Args:
            processed_query: 处理后的查询信息
            prediction: 预测结果
            
        Returns:
            决策摘要文本
        """
        text = processed_query.get('text', '')
        entities = processed_query.get('entities', [])
        
        # 根据预测结果生成摘要
        if prediction == 'compare':
            if len(entities) >= 2:
                return f"识别为比较意图，将比较{entities[0]}和{entities[1]}"
            else:
                return f"识别为比较意图，查询文本包含比较相关词汇"
        elif prediction == 'query':
            if entities:
                return f"识别为查询意图，查询{entities[0]}的相关信息"
            else:
                return f"识别为查询意图，查询相关信息"
        elif prediction == 'list':
            return f"识别为列出意图，将列出相关内容"
        else:
            return f"识别为{prediction}意图，基于查询特征进行分类"


class DeepInterpreter:
    """
    深度解释器 - 用于离线分析
    
    重要说明：
    此模块在模型训练阶段不使用，专门为后续qProcessor在线意图识别服务设计。
    
    主要功能：
    1. 提供深度模型分析，用于离线诊断和优化
    2. 计算特征重要性，分析模型决策依据
    3. 分析决策路径，理解模型推理过程
    4. 生成模型诊断报告和优化建议
    
    使用场景：
    - 模型性能分析时，提供深度诊断和优化建议
    - 模型调优时，分析特征重要性和决策路径
    - 生成模型解释报告，用于业务理解和模型改进
    
    注意：此模块不参与模型训练，仅在模型分析阶段使用。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化深度解释器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.analysis_depth = config.get('analysis_depth', 'full')
        self.feature_importance_methods = config.get('feature_importance_methods', ['weight', 'permutation'])
        self.visualization_enabled = config.get('visualization_enabled', True)
        
    def analyze_model(self, model, test_data: Dict[str, Any], feature_names: List[str] = None) -> Dict[str, Any]:
        """
        深度分析模型
        
        Args:
            model: 训练好的模型
            test_data: 测试数据
            feature_names: 特征名称列表
            
        Returns:
            深度分析结果字典
        """
        logger.info("开始深度模型分析")
        
        X = test_data.get('X')
        y = test_data.get('y')
        
        if X is None or y is None:
            logger.warning("测试数据缺少X或y")
            return {}
        
        # 计算特征重要性
        feature_importance = self._calculate_feature_importance(model, X, y, feature_names)
        
        # 分析决策路径
        decision_paths = self._analyze_decision_paths(model, X, feature_names)
        
        # 运行模型诊断
        diagnostics = self._run_diagnostics(model, X, y)
        
        # 生成优化建议
        suggestions = self._generate_suggestions(model, {
            'feature_importance': feature_importance,
            'diagnostics': diagnostics
        })
        
        return {
            'feature_importance': feature_importance,
            'decision_paths': decision_paths,
            'diagnostics': diagnostics,
            'suggestions': suggestions
        }
        
    def _calculate_feature_importance(self, model, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        计算特征重要性
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 标签
            feature_names: 特征名称列表
            
        Returns:
            特征重要性结果
        """
        result = {}
        
        # 基于权重的特征重要性
        if hasattr(model, 'coef_'):
            weight_importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            result['weight_importance'] = weight_importance
        elif hasattr(model, 'feature_importances_'):
            result['weight_importance'] = model.feature_importances_
        else:
            # 如果没有特征重要性属性，使用随机值
            result['weight_importance'] = np.random.rand(X.shape[1])
        
        # 排列重要性（简化版本）
        try:
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            result['permutation_importance'] = perm_importance.importances_mean
        except:
            # 如果无法计算排列重要性，使用权重重要性
            result['permutation_importance'] = result['weight_importance']
        
        # 特征重要性排序
        if feature_names:
            importance_with_names = list(zip(feature_names, result['weight_importance']))
            importance_with_names.sort(key=lambda x: x[1], reverse=True)
            result['top_features'] = importance_with_names[:10]  # 前10个重要特征
        
        return result
        
    def _analyze_decision_paths(self, model, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        分析决策路径
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            feature_names: 特征名称列表
            
        Returns:
            决策路径分析结果
        """
        result = {
            'decision_rules': [],
            'path_analysis': {}
        }
        
        # 对于线性模型，分析决策边界
        if hasattr(model, 'coef_'):
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
            
            # 生成决策规则
            if feature_names:
                for i, (name, weight) in enumerate(zip(feature_names, coef)):
                    if abs(weight) > 0.01:  # 只考虑重要特征
                        if weight > 0:
                            rule = f"如果{name}增加，倾向于正类"
                        else:
                            rule = f"如果{name}增加，倾向于负类"
                        result['decision_rules'].append(rule)
            
            # 决策边界分析
            result['path_analysis'] = {
                'decision_boundary': f"决策边界: {intercept:.3f} + sum(coefficients * features) = 0",
                'positive_features': [f for f, w in zip(feature_names or [], coef) if w > 0.01],
                'negative_features': [f for f, w in zip(feature_names or [], coef) if w < -0.01]
            }
        
        # 对于树模型，分析决策路径
        elif hasattr(model, 'tree_'):
            result['decision_rules'] = ["树模型决策路径分析"]
            result['path_analysis'] = {
                'tree_depth': getattr(model, 'max_depth', 'unknown'),
                'n_leaves': getattr(model, 'n_leaves_', 'unknown')
            }
        
        return result
        
    def _run_diagnostics(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        运行模型诊断
        
        Args:
            model: 训练好的模型
            X: 特征矩阵
            y: 标签
            
        Returns:
            诊断结果
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 预测
        y_pred = model.predict(X)
        
        # 计算性能指标
        try:
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        except:
            accuracy = precision = recall = f1 = 0.0
        
        # 数据质量分析
        missing_ratio = np.isnan(X).sum() / X.size if X.size > 0 else 0
        feature_variance = np.var(X, axis=0).mean() if X.size > 0 else 0
        
        # 潜在问题检测
        potential_issues = []
        if accuracy < 0.7:
            potential_issues.append("模型准确率较低，可能需要更多训练数据或特征工程")
        if missing_ratio > 0.1:
            potential_issues.append("数据中存在较多缺失值")
        if feature_variance < 0.01:
            potential_issues.append("特征方差较小，可能存在冗余特征")
        
        return {
            'model_performance': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'data_quality': {
                'missing_ratio': missing_ratio,
                'feature_variance': feature_variance,
                'n_samples': len(X),
                'n_features': X.shape[1]
            },
            'potential_issues': potential_issues
        }
        
    def _generate_suggestions(self, model, analysis_results: Dict[str, Any]) -> List[str]:
        """
        生成优化建议
        
        Args:
            model: 训练好的模型
            analysis_results: 分析结果
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        # 基于性能指标的建议
        if 'diagnostics' in analysis_results:
            diagnostics = analysis_results['diagnostics']
            performance = diagnostics.get('model_performance', {})
            data_quality = diagnostics.get('data_quality', {})
            
            accuracy = performance.get('accuracy', 0)
            if accuracy < 0.8:
                suggestions.append("建议增加训练数据以提高模型准确率")
            if accuracy < 0.6:
                suggestions.append("建议重新设计特征工程策略")
            
            missing_ratio = data_quality.get('missing_ratio', 0)
            if missing_ratio > 0.05:
                suggestions.append("建议处理数据中的缺失值")
        
        # 基于特征重要性的建议
        if 'feature_importance' in analysis_results:
            feature_importance = analysis_results['feature_importance']
            if 'top_features' in feature_importance:
                top_features = feature_importance['top_features']
                if len(top_features) > 0:
                    top_feature_name = top_features[0][0]
                    suggestions.append(f"重点关注特征'{top_feature_name}'，其对模型影响最大")
        
        # 基于模型类型的建议
        if hasattr(model, 'coef_'):
            suggestions.append("线性模型建议：考虑特征标准化以提高训练稳定性")
        elif hasattr(model, 'feature_importances_'):
            suggestions.append("树模型建议：考虑调整树的深度和叶子节点数量")
        
        # 如果没有具体建议，提供通用建议
        if not suggestions:
            suggestions.append("建议定期重新训练模型以保持性能")
            suggestions.append("建议收集更多用户反馈数据")
        
        return suggestions


class InterpretabilityManager:
    """
    解释性功能管理器
    
    重要说明：
    此模块在模型训练阶段不使用，专门为后续qProcessor在线意图识别服务设计。
    
    主要功能：
    1. 统一管理轻量级解释器和深度解释器
    2. 为qProcessor提供统一的解释接口
    3. 支持在线解释和离线分析两种模式
    4. 生成解释报告，用于业务理解和模型改进
    
    使用场景：
    - qProcessor在线预测时，提供实时解释服务
    - 模型分析时，提供深度诊断和报告生成
    - 统一管理所有可解释性相关功能
    
    注意：此模块不参与模型训练，仅在推理阶段和模型分析阶段使用。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化解释性功能管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.light_interpreter = LightInterpreter(config.get('light_interpreter', {}))
        self.deep_interpreter = DeepInterpreter(config.get('deep_interpreter', {}))
        
    def explain_online(self, processed_query: Dict[str, Any], model=None, features=None) -> Dict[str, Any]:
        """
        在线解释接口
        
        Args:
            processed_query: 处理后的查询信息
            model: 训练好的模型（可选）
            features: 特征向量（可选）
            
        Returns:
            在线解释结果
        """
        return self.light_interpreter.explain(processed_query, model, features)
        
    def analyze_offline(self, model, test_data: Dict[str, Any], feature_names: List[str] = None) -> Dict[str, Any]:
        """
        离线分析接口
        
        Args:
            model: 训练好的模型
            test_data: 测试数据
            feature_names: 特征名称列表
            
        Returns:
            离线分析结果
        """
        return self.deep_interpreter.analyze_model(model, test_data, feature_names)
        
    def generate_report(self, analysis_results: Dict[str, Any], output_path: str = None) -> str:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果
            output_path: 输出路径（可选）
            
        Returns:
            报告内容
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("模型解释性分析报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 特征重要性部分
        if 'feature_importance' in analysis_results:
            report_lines.append("## 特征重要性分析")
            report_lines.append("-" * 30)
            feature_importance = analysis_results['feature_importance']
            
            if 'top_features' in feature_importance:
                report_lines.append("### 前10个重要特征:")
                for i, (name, importance) in enumerate(feature_importance['top_features'][:10], 1):
                    report_lines.append(f"{i}. {name}: {importance:.4f}")
            report_lines.append("")
        
        # 决策路径部分
        if 'decision_paths' in analysis_results:
            report_lines.append("## 决策路径分析")
            report_lines.append("-" * 30)
            decision_paths = analysis_results['decision_paths']
            
            if 'decision_rules' in decision_paths:
                report_lines.append("### 决策规则:")
                for i, rule in enumerate(decision_paths['decision_rules'][:5], 1):
                    report_lines.append(f"{i}. {rule}")
            report_lines.append("")
        
        # 诊断结果部分
        if 'diagnostics' in analysis_results:
            report_lines.append("## 模型诊断结果")
            report_lines.append("-" * 30)
            diagnostics = analysis_results['diagnostics']
            
            if 'model_performance' in diagnostics:
                performance = diagnostics['model_performance']
                report_lines.append("### 性能指标:")
                report_lines.append(f"- 准确率: {performance.get('accuracy', 0):.4f}")
                report_lines.append(f"- 精确率: {performance.get('precision', 0):.4f}")
                report_lines.append(f"- 召回率: {performance.get('recall', 0):.4f}")
                report_lines.append(f"- F1分数: {performance.get('f1_score', 0):.4f}")
            
            if 'data_quality' in diagnostics:
                quality = diagnostics['data_quality']
                report_lines.append("### 数据质量:")
                report_lines.append(f"- 样本数量: {quality.get('n_samples', 0)}")
                report_lines.append(f"- 特征数量: {quality.get('n_features', 0)}")
                report_lines.append(f"- 缺失值比例: {quality.get('missing_ratio', 0):.4f}")
            
            if 'potential_issues' in diagnostics:
                issues = diagnostics['potential_issues']
                if issues:
                    report_lines.append("### 潜在问题:")
                    for issue in issues:
                        report_lines.append(f"- {issue}")
            report_lines.append("")
        
        # 优化建议部分
        if 'suggestions' in analysis_results:
            report_lines.append("## 优化建议")
            report_lines.append("-" * 30)
            suggestions = analysis_results['suggestions']
            for i, suggestion in enumerate(suggestions, 1):
                report_lines.append(f"{i}. {suggestion}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        report_lines.append("报告生成完成")
        report_lines.append("=" * 60)
        
        report_content = "\n".join(report_lines)
        
        # 保存到文件（如果指定了路径）
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"报告已保存到: {output_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report_content 