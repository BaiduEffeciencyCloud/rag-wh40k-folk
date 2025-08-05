import argparse
import os
import sys
import yaml
import joblib
import shutil
from datetime import datetime
import jieba

# 修复导入路径
from intent_training.data.data_manager import DataManager
from intent_training.evaluation.evaluator import Evaluator
from intent_training.model.intent_classifier import IntentClassifier
from intent_training.model.sequence_slot_filler import SequenceSlotFiller
from intent_training.feature.factory import FeatureExtractorFactory

class IntegratedModel:
    """一个包装类，用于将意图和槽位模型统一成Evaluator所需的单一predict接口"""
    def __init__(self, intent_clf, slot_filler, feature_extractor, use_enhanced=False):
        self.intent_clf = intent_clf
        self.slot_filler = slot_filler
        self.feature_extractor = feature_extractor
        self.use_enhanced = use_enhanced
        self.tokenized_texts = None

    def predict(self, texts, tokenized_texts=None):
        """
        预测意图和槽位标签
        
        Args:
            texts: 输入文本列表
            tokenized_texts: 预定义的分词结果列表，如果为None则使用jieba分词
            
        Returns:
            tuple: (intent_preds, slot_preds, intent_confidences) 意图预测结果、槽位预测结果和意图置信度
        """
        # 意图预测
        if self.use_enhanced:
            # 使用增强特征预测意图
            intent_preds = []
            intent_confidences = []
            for text in texts:
                # 获取增强特征
                enhanced_features = self.feature_extractor.get_enhanced_intent_features(text)
                enhanced_features = enhanced_features.flatten().reshape(1, -1)
                # 使用加载的分类器进行预测
                intent = self.intent_clf.predict(enhanced_features)[0]
                # 获取预测概率
                probabilities = self.intent_clf.predict_proba(enhanced_features)[0]
                # 创建置信度字典
                confidence_dict = {}
                for i, class_name in enumerate(self.intent_clf.classes_):
                    confidence_dict[class_name] = float(probabilities[i])
                intent_preds.append(intent)
                intent_confidences.append(confidence_dict)
        else:
            # 使用基础特征预测意图
            intent_preds = self.intent_clf.predict(self.feature_extractor.transform(texts))
            # 获取预测概率
            probabilities = self.intent_clf.predict_proba(self.feature_extractor.transform(texts))
            # 创建置信度字典列表
            intent_confidences = []
            for prob in probabilities:
                confidence_dict = {}
                for i, class_name in enumerate(self.intent_clf.classes_):
                    confidence_dict[class_name] = float(prob[i])
                intent_confidences.append(confidence_dict)
        
        # 槽位预测
        slot_preds_data = [{"query": text, "intent": intent} for text, intent in zip(texts, intent_preds)]
        slot_results = [self.slot_filler.predict(query=item['query'], intent=item['intent']) for item in slot_preds_data]

        # 将字符位置的槽位信息转换为词级别的BIO标注
        formatted_slot_preds = []
        for i, (text, result) in enumerate(zip(texts, slot_results)):
            # 使用预定义的分词结果或jieba分词
            if tokenized_texts and i < len(tokenized_texts):
                tokens = tokenized_texts[i]
            elif self.tokenized_texts and i < len(self.tokenized_texts):
                tokens = self.tokenized_texts[i]
            else:
                tokens = list(jieba.cut(text))
            
            token_labels = ['O'] * len(tokens)
            
            # 从SlotFillingResult中提取槽位信息
            for slot_name, slot in result.slots.items():
                slot_value = slot.value
                start_pos = slot.start_pos
                end_pos = slot.end_pos
                
                # 将字符位置映射到词位置
                char_pos = 0
                token_start_idx = -1
                token_end_idx = -1
                
                for j, token in enumerate(tokens):
                    token_len = len(token)
                    # 检查当前词是否包含槽位的开始位置
                    if token_start_idx == -1 and start_pos >= char_pos and start_pos < char_pos + token_len:
                        token_start_idx = j
                    # 检查当前词是否包含槽位的结束位置
                    if end_pos > char_pos and end_pos <= char_pos + token_len:
                        token_end_idx = j
                        break
                    char_pos += token_len
                
                # 如果找到了匹配的词位置，进行标注
                if token_start_idx != -1 and token_end_idx != -1:
                    # 标记B标签
                    token_labels[token_start_idx] = f'B-{slot_name}'
                    # 标记I标签
                    for j in range(token_start_idx + 1, token_end_idx + 1):
                        if j < len(token_labels):
                            token_labels[j] = f'I-{slot_name}'
            
            formatted_slot_preds.append(token_labels)

        return intent_preds, formatted_slot_preds, intent_confidences

def find_model_version_path(model_root, version):
    """根据version参数查找模型路径"""
    if not os.path.exists(model_root):
        return None
        
    if version == "head":
        all_versions = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
        if not all_versions:
            return None
        latest_version = sorted(all_versions)[-1]
        return os.path.join(model_root, latest_version)
    else:
        # 简单验证是否是时间戳格式
        if version.isdigit() and len(version) == 12:
            path = os.path.join(model_root, version)
            if os.path.exists(path):
                return path
    return None

def load_dependencies(version_path):
    """从指定版本路径加载模型、特征和配置"""
    try:
        # 加载配置
        config_path = os.path.join(version_path, "config", "config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 使用工厂方法创建特征提取器
        feature_extractor = FeatureExtractorFactory.create_feature_extractor(config)
        
        # 检查是否使用增强特征
        use_enhanced = FeatureExtractorFactory.is_enhanced_feature_extractor(config)
        
        if use_enhanced:
            # 加载增强特征提取器（使用现有命名约定）
            intent_tfidf_name = config.get('intent_feature', {}).get('intent_tfidf', 'intent_tfidf.pkl')
            feature_path = os.path.join(version_path, "feature", intent_tfidf_name)
            feature_extractor.load_enhanced_feature_extractor(feature_path)
            print(f"加载增强特征提取器: {intent_tfidf_name}")
        else:
            # 加载基础特征提取器
            feature_path = os.path.join(version_path, "feature", config['intent_feature']['intent_tfidf'])
            feature_extractor.load(feature_path)
            print(f"加载基础特征提取器: {config['intent_feature']['intent_tfidf']}")

        # 加载模型
        model_path = os.path.join(version_path, "model")
        
        # 无论基础还是增强特征，都需要加载单独的意图分类器
        intent_clf = IntentClassifier(config)
        intent_clf.load_model(os.path.join(model_path, config['model']['intent_classifier']))
        print(f"加载意图分类器: {config['model']['intent_classifier']}")
        
        slot_filler = SequenceSlotFiller(config)
        slot_filler.load_model(os.path.join(model_path, config['model']['sequence_slot_filler']))

        print("模型及依赖加载成功！")
        return {
            "config": config,
            "feature_extractor": feature_extractor,
            "intent_clf": intent_clf,
            "slot_filler": slot_filler,
            "use_enhanced": use_enhanced
        }
    except Exception as e:
        print(f"错误: 加载依赖失败 - {e}")
        return None

def load_golden_test_data(config):
    """加载黄金测试集"""
    try:
        data_manager = DataManager(config)
        # 路径相对于项目根目录，脚本在intent_training下，所以需要../
        intent_file = os.path.join("intent_training/data", "golden_test", "intent_test.csv")
        slot_file = os.path.join("intent_training/data", "golden_test", "slot_test.conll")
        
        # 假设 DataManager 能够处理相对于项目根目录的路径
        intent_df = data_manager.load_intent_data(intent_file)
        slot_sentences, slot_labels = data_manager.load_slot_data(slot_file)
        
        # 将意图和槽位数据合并对齐
        # 注意：此处的对齐逻辑是一个简化实现
        texts = intent_df['query'].tolist()
        true_intents = intent_df['intent'].tolist()
        
        # 假设slot_sentences和texts是一一对应的
        # 在真实场景中，需要更鲁棒的对齐机制
        if len(texts) != len(slot_sentences):
             print(f"警告: 黄金测试集的意图样本数({len(texts)})与槽位样本数({len(slot_sentences)})不一致。")
             # 此处可以决定是截断、填充还是报错
             # 暂时返回None表示失败
             return None

        # 提取CoNLL文件中的分词信息
        tokenized_texts = _extract_tokenized_sentences_from_conll(slot_file)
        
        # 验证分词信息与槽位标签数量一致
        if len(tokenized_texts) != len(slot_labels):
            print(f"警告: 分词信息数量({len(tokenized_texts)})与槽位标签数量({len(slot_labels)})不一致。")
            return None
        
        for i, (tokens, labels) in enumerate(zip(tokenized_texts, slot_labels)):
            if len(tokens) != len(labels):
                print(f"警告: 第{i+1}个样本的分词数量({len(tokens)})与标签数量({len(labels)})不一致。")
                return None

        print("黄金测试集加载成功！")
        return texts, true_intents, slot_labels, tokenized_texts
        
    except FileNotFoundError as e:
        print(f"错误: 黄金测试集文件未找到 - {e}")
        return None
    except Exception as e:
        print(f"错误: 加载黄金测试集失败 - {e}")
        return None


def _extract_tokenized_sentences_from_conll(conll_file_path):
    """
    从CoNLL文件中提取分词信息
    
    Args:
        conll_file_path: CoNLL文件路径
        
    Returns:
        List[List[str]]: 每个句子的分词列表
    """
    tokenized_sentences = []
    current_sentence = []
    
    try:
        with open(conll_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 空行表示句子结束
                if not line:
                    if current_sentence:
                        tokenized_sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                # 解析每行：词\t标签
                parts = line.split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    current_sentence.append(token)
            
            # 处理最后一个句子
            if current_sentence:
                tokenized_sentences.append(current_sentence)
                
    except Exception as e:
        print(f"错误: 解析CoNLL文件失败 - {e}")
        return []
    
    return tokenized_sentences


def handle_model_export(config, evaluator, version_path):
    """
    根据评估结果条件导出模型
    
    Args:
        config: 配置字典
        evaluator: 评估器实例
        version_path: 模型版本路径
    """
    
    evaluation_config = config.get('evaluation', {})
    if not evaluation_config.get('run_evaluation', False):
        print("评估导出功能未启用")
        return
    
    export_config = evaluation_config.get('export', {})
    if not export_config.get('enabled', False):
        print("模型导出功能未启用")
        return
    
    export_dir = export_config.get('export_dir', 'dict/')
    override = export_config.get('override', True)
    
    # 创建导出目录
    os.makedirs(export_dir, exist_ok=True)
    
    # 检查意图分类器导出
    intent_config = evaluation_config.get('intent_classifier', {})
    if intent_config.get('enabled', False):
        intent_accuracy = evaluator.last_intent_accuracy
        threshold = intent_config.get('threshold', 0.9)
        
        if intent_accuracy >= threshold:
            print(f"意图分类器准确率: {intent_accuracy:.4f} 超过阈值 {threshold}，准备导出")
            export_model('intent_classifier', version_path, export_dir, override, config)
            export_feature_extractor(version_path, export_dir, override, config)
            export_config_file(version_path, export_dir, override, config)
        else:
            print(f"意图分类器准确率: {intent_accuracy:.4f} 未达到阈值 {threshold}")
    
    # 检查槽位填充器导出
    slot_config = evaluation_config.get('sequence_slot_filler', {})
    if slot_config.get('enabled', False):
        f1_score = evaluator.last_slot_f1_report.get('f1_score', 0.0)
        threshold = slot_config.get('threshold', 0.5)
        
        if f1_score >= threshold:
            print(f"槽位填充器F1分数: {f1_score:.4f} 超过阈值 {threshold}，准备导出")
            export_model('sequence_slot_filler', version_path, export_dir, override, config)
        else:
            print(f"槽位填充器F1分数: {f1_score:.4f} 未达到阈值 {threshold}")


def export_model(model_type, version_path, export_dir, override, config):
    """
    导出模型到指定目录
    
    Args:
        model_type: 模型类型
        version_path: 模型版本路径
        export_dir: 导出目录
        override: 是否覆盖
        config: 配置字典
    """
    model_filename = f"{model_type}.pkl"
    source_path = os.path.join(version_path, "model", model_filename)
    target_path = os.path.join(export_dir, model_filename)
    
    if not os.path.exists(source_path):
        print(f"源模型文件不存在: {source_path}")
        return
    
    if os.path.exists(target_path) and not override:
        print(f"目标文件已存在且不允许覆盖: {target_path}")
        return
    
    if os.path.exists(target_path) and override:
        os.remove(target_path)
        print(f"删除已存在的目标文件: {target_path}")
    
    shutil.copy2(source_path, target_path)
    print(f"模型导出成功: {source_path} -> {target_path}")


def export_feature_extractor(version_path, export_dir, override, config):
    """
    导出特征提取器到指定目录
    
    Args:
        version_path: 模型版本路径
        export_dir: 导出目录
        override: 是否覆盖
        config: 配置字典
    """
    feature_extractor_name = config.get('intent_feature', {}).get('intent_tfidf', 'intent_tfidf.pkl')
    source_path = os.path.join(version_path, "feature", feature_extractor_name)
    target_path = os.path.join(export_dir, feature_extractor_name)
    
    if not os.path.exists(source_path):
        print(f"特征提取器文件不存在: {source_path}")
        return
    
    if os.path.exists(target_path) and not override:
        print(f"特征提取器文件已存在且不允许覆盖: {target_path}")
        return
    
    if os.path.exists(target_path) and override:
        os.remove(target_path)
        print(f"删除已存在的特征提取器文件: {target_path}")
    
    shutil.copy2(source_path, target_path)
    print(f"特征提取器导出成功: {source_path} -> {target_path}")


def export_config_file(version_path, export_dir, override, config):
    """
    导出配置文件到指定目录
    
    Args:
        version_path: 模型版本路径
        export_dir: 导出目录
        override: 是否覆盖
        config: 配置字典
    """
    source_path = os.path.join(version_path, "config", "config.yaml")
    target_path = os.path.join(export_dir, "config.yaml")
    
    if not os.path.exists(source_path):
        print(f"配置文件不存在: {source_path}")
        return
    
    if os.path.exists(target_path) and not override:
        print(f"配置文件已存在且不允许覆盖: {target_path}")
        return
    
    if os.path.exists(target_path) and override:
        os.remove(target_path)
        print(f"删除已存在的配置文件: {target_path}")
    
    shutil.copy2(source_path, target_path)
    print(f"配置文件导出成功: {source_path} -> {target_path}")


def main():
    parser = argparse.ArgumentParser(description="对指定的模型版本进行深度评估")
    parser.add_argument("--ver", "-v", default="head",
                        help="要评估的模型版本 (时间戳 YYMMDDHHMMSS 或 'head' 表示最新)")
    args = parser.parse_args()

    # 1. 查找模型路径
    model_root = "intent_training/model_output" # 路径相对于intent_training
    version_path = find_model_version_path(model_root, args.ver)
    if not version_path:
        print(f"错误: 找不到模型版本 '{args.ver}'")
        sys.exit(1)

    # 2. 加载依赖
    dependencies = load_dependencies(version_path)
    if not dependencies:
        sys.exit(1)

    # 3. 加载黄金测试集
    test_data = load_golden_test_data(dependencies['config'])
    if not test_data:
        sys.exit(1)

    # 4. 封装模型并初始化评估器
    # 我们需要在加载的config基础上，动态修改报告的输出路径
    eval_config = dependencies['config'].copy()
    report_ts = datetime.now().strftime("%y%m%d%H%M%S")
    report_dir = os.path.join(version_path, "report")
    # 创建report目录
    os.makedirs(report_dir, exist_ok=True)
    eval_config['summary_report_path'] = os.path.join(report_dir, f"{report_ts}_summary.md")
    eval_config['detailed_report_path'] = os.path.join(report_dir, f"{report_ts}_detailed.json")
    
    integrated_model = IntegratedModel(
        dependencies['intent_clf'],
        dependencies['slot_filler'],
        dependencies['feature_extractor'],
        dependencies['use_enhanced']
    )
    evaluator = Evaluator(integrated_model, eval_config, mode="full_analysis")

    # 5. 执行评估
    print(f"正在对模型版本 '{os.path.basename(version_path)}' 进行评估...")
    texts, true_intents, true_slots, tokenized_texts = test_data
    
    # 将分词信息传递给模型
    integrated_model.tokenized_texts = tokenized_texts
    
    all_metrics = evaluator.run(texts, true_intents, true_slots)

    print("\n评估完成！")
    print(f"  - 摘要报告已生成: {eval_config['summary_report_path']}")
    print(f"  - 详细报告已生成: {eval_config['detailed_report_path']}")
    print("\n主要指标:")
    print(f"  - Frame Accuracy: {all_metrics:.4f}")
    
    # 6. 条件导出模型
    handle_model_export(dependencies['config'], evaluator, version_path)

if __name__ == "__main__":
    main() 