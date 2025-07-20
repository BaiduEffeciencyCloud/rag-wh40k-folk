import os
from datetime import datetime
import shutil
from typing import List, Dict

class IntentTraining:
    def __init__(self, model_output_root, raw_dir, config_path):
        """
        初始化pipeline归档主流程。
        - model_output_root: 归档根目录（如 intent_training/model_output/）
        - raw_dir: 原始样本文件目录（如 data/raw/）
        - config_path: 本次训练用到的配置文件路径
        自动生成时间戳归档目录和各子目录。
        """
        self.model_output_root = model_output_root
        self.raw_dir = raw_dir
        self.config_path = config_path
        self.ts = datetime.now().strftime("%y%m%d%H%M%S")
        self.archive_dir = os.path.join(self.model_output_root, self.ts)
        self.dirs = {
            'raw': os.path.join(self.archive_dir, 'raw'),
            'split': os.path.join(self.archive_dir, 'split'),
            'feature': os.path.join(self.archive_dir, 'feature'),
            'model': os.path.join(self.archive_dir, 'model'),
            'report': os.path.join(self.archive_dir, 'report'),
            'config': os.path.join(self.archive_dir, 'config'),
        }
        self._init_logger()

    def _extract_slots_from_conll(self, sentence: List[str], labels: List[str]) -> Dict[str, str]:
        """
        从CoNLL格式的标签中提取槽位信息
        
        Args:
            sentence: 句子词列表
            labels: 标签列表
            
        Returns:
            槽位字典 {slot_name: slot_value}
        """
        slots = {}
        current_slot = None
        current_value = []
        current_start = 0
        
        for i, (word, label) in enumerate(zip(sentence, labels)):
            if label.startswith('B-'):
                # 保存之前的槽位
                if current_slot and current_value:
                    slots[current_slot] = ''.join(current_value)
                
                # 开始新槽位
                current_slot = label[2:]  # 去掉'B-'前缀
                current_value = [word]
                current_start = i
                
            elif label.startswith('I-') and current_slot:
                # 继续当前槽位
                if label[2:] == current_slot:  # 确保是同一个槽位类型
                    current_value.append(word)
                else:
                    # 槽位类型不匹配，结束当前槽位
                    if current_value:
                        slots[current_slot] = ''.join(current_value)
                    current_slot = None
                    current_value = []
                    
            else:
                # O标签或其他，结束当前槽位
                if current_slot and current_value:
                    slots[current_slot] = ''.join(current_value)
                current_slot = None
                current_value = []
        
        # 处理最后一个槽位
        if current_slot and current_value:
            slots[current_slot] = ''.join(current_value)
        
        return slots

    def archive_raw(self):
        """
        归档本次训练用到的原始样本文件。
        输入：self.raw_dir 目录下的所有原始数据文件（如 intent_sample.csv、slot_sample.conll）。
        输出：将所有原始数据文件复制到 self.dirs['raw'] 目录。
        边界：
            - 若 raw_dir 为空目录，则 raw/ 目录为空。
            - 若原始文件缺失，归档后无对应文件。
        异常兜底：
            - 目录不存在时自动创建。
            - 文件不存在时跳过，不抛异常。
        典型调用：self.archive_raw()
        失败行为：仅影响单个文件归档，不影响主流程。
        """
        os.makedirs(self.dirs['raw'], exist_ok=True)
        if not os.path.isdir(self.raw_dir):
            self.logger.warning(f"原始数据目录不存在: {self.raw_dir}")
            return
        for fname in os.listdir(self.raw_dir):
            src = os.path.join(self.raw_dir, fname)
            dst = os.path.join(self.dirs['raw'], fname)
            try:
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    self.logger.info(f"归档原始文件: {fname}")
            except Exception as e:
                self.logger.error(f"归档原始文件失败: {fname}, {e}")

    def archive_split(self):
        """
        归档数据切分产物。
        输入：数据切分后生成的文件（如 intent_train.csv、slot_train.conll 等），假定已生成在指定临时目录。
        输出：将所有切分数据文件复制到 self.dirs['split'] 目录。
        边界：
            - 若无切分产物，split/ 目录为空。
        异常兜底：
            - 目录不存在时自动创建。
            - 文件不存在时跳过，不抛异常。
        典型调用：self.archive_split()
        失败行为：仅影响单个文件归档，不影响主流程。
        """
        os.makedirs(self.dirs['split'], exist_ok=True)
        # 真实实现后补充

    def archive_feature(self):
        """
        归档特征工程产物。
        输入：特征工程生成的文件（如 tfidf.pkl、vocab.json、feature_config.json 等），假定已生成在指定临时目录。
        输出：将所有特征相关文件复制到 self.dirs['feature'] 目录。
        边界：
            - 若无特征产物，feature/ 目录为空。
        异常兜底：
            - 目录不存在时自动创建。
            - 文件不存在时跳过，不抛异常。
        典型调用：self.archive_feature()
        失败行为：仅影响单个文件归档，不影响主流程。
        """
        os.makedirs(self.dirs['feature'], exist_ok=True)
        # 真实实现后补充

    def archive_model(self):
        """
        归档训练好的模型文件。
        输入：模型训练产物（如 intent_classifier.pkl、slot_filler.pkl、label_map.json 等），假定已生成在指定临时目录。
        输出：将所有模型相关文件复制到 self.dirs['model'] 目录。
        边界：
            - 若无模型产物，model/ 目录为空。
        异常兜底：
            - 目录不存在时自动创建。
            - 文件不存在时跳过，不抛异常。
        典型调用：self.archive_model()
        失败行为：仅影响单个文件归档，不影响主流程。
        """
        os.makedirs(self.dirs['model'], exist_ok=True)
        # 真实实现后补充

    def archive_report(self):
        """
        归档评估报告。
        输入：评估产物（如 report.json、insight.md、混淆矩阵等），假定已生成在指定临时目录。
        输出：将所有评估相关文件复制到 self.dirs['report'] 目录。
        边界：
            - 若无评估产物，report/ 目录为空。
        异常兜底：
            - 目录不存在时自动创建。
            - 文件不存在时跳过，不抛异常。
        典型调用：self.archive_report()
        失败行为：仅影响单个文件归档，不影响主流程。
        """
        os.makedirs(self.dirs['report'], exist_ok=True)
        # 真实实现后补充

    def archive_config(self):
        """
        归档本次训练用到的配置文件。
        输入：self.config_path 指定的配置文件（如 config.yaml）。
        输出：将配置文件复制到 self.dirs['config'] 目录。
        边界：
            - 若配置文件不存在，config/ 目录为空。
        异常兜底：
            - 目录不存在时自动创建。
            - 文件不存在时跳过，不抛异常。
        典型调用：self.archive_config()
        失败行为：仅影响配置文件归档，不影响主流程。
        """
        os.makedirs(self.dirs['config'], exist_ok=True)
        if not os.path.isfile(self.config_path):
            self.logger.warning(f"配置文件不存在: {self.config_path}")
            return
        try:
            shutil.copy2(self.config_path, os.path.join(self.dirs['config'], os.path.basename(self.config_path)))
            self.logger.info(f"归档配置文件: {self.config_path}")
        except Exception as e:
            self.logger.error(f"归档配置文件失败: {self.config_path}, {e}")


    def _init_logger(self):
        """
        初始化pipeline日志，日志文件与归档目录同级，便于归档和溯源。
        """
        import logging
        os.makedirs(self.archive_dir, exist_ok=True)
        log_path = os.path.join(self.archive_dir, "pipeline.log")
        self.logger = logging.getLogger(f"pipeline_{self.ts}")
        self.logger.setLevel(logging.INFO)
        # 优雅地移除旧的handler，先close再clear，防止资源泄漏
        if self.logger.hasHandlers():
            for handler in self.logger.handlers:
                try:
                    handler.close()
                except Exception:
                    pass
            self.logger.handlers.clear()
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def run(self):
        """
        串联整个离线训练数据流，依次执行各阶段并归档产物。
        1. 数据加载与清洗
        2. 数据切分
        3. 特征工程
        4. 模型训练
        5. 归档所有阶段产物
        日志文件 pipeline.log 会记录每一步归档和异常。
        """
        import yaml
        import json
        import pandas as pd
        from intent_training.data.data_manager import DataManager
        from intent_training.feature.feature_engineering import IntentFeatureExtractor
        from intent_training.model.intent_classifier import IntentClassifier
        from intent_training.model.slot_filler import SlotFiller
        import joblib

        # 1. 加载配置
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return
        self.logger.info(f"[pipeline] 加载配置完成: {self.config_path}")

        # 2. 数据加载与清洗
        data_manager = DataManager(config)
        intent_file = config.get('data', {}).get('intent_file', 'intent_sample.csv')
        slot_file = config.get('data', {}).get('slot_file', 'slot_sample.conll')
        intent_path = os.path.join(self.raw_dir, intent_file)
        slot_path = os.path.join(self.raw_dir, slot_file)
        try:
            intent_df = data_manager.load_intent_data(intent_path)
            slot_sentences, slot_labels = data_manager.load_slot_data(slot_path)
        except FileNotFoundError as e:
            self.logger.error(str(e))
            return
        self.archive_raw()

        # 3. 数据切分
        split_result = data_manager.split_data('both')
        split_dir = self.dirs['split']
        os.makedirs(split_dir, exist_ok=True)
        for name in ['train', 'val', 'test']:
            X, y = split_result['intent'][name]
            df = pd.DataFrame({'query': X, 'intent': y})
            df.to_csv(os.path.join(split_dir, f"intent_{name}.csv"), index=False)
        for name in ['train', 'val', 'test']:
            sents, labels = split_result['slot'][name]
            with open(os.path.join(split_dir, f"slot_{name}.conll"), "w", encoding="utf-8") as f:
                for sent, label_seq in zip(sents, labels):
                    for w, l in zip(sent, label_seq):
                        f.write(f"{w}\t{l}\n")
                    f.write("\n")
        self.archive_split()

        # 4. 特征工程
        feature_dir = self.dirs['feature']
        os.makedirs(feature_dir, exist_ok=True)
        intent_feat = IntentFeatureExtractor(config)
        X_train = split_result['intent']['train'][0]
        intent_feat.fit(X_train)
        intent_tfidf_name = config.get('feature', {}).get('intent_tfidf', 'intent_tfidf.pkl')
        joblib.dump(intent_feat.tfidf_vectorizer, os.path.join(feature_dir, intent_tfidf_name))
        self.archive_feature()

        # 5. 模型训练
        model_dir = self.dirs['model']
        os.makedirs(model_dir, exist_ok=True)
        y_train = split_result['intent']['train'][1]
        X_train_vec = intent_feat.transform(X_train)
        intent_clf = IntentClassifier(config)
        intent_clf.fit(X_train_vec, y_train)
        intent_clf_name = config.get('model', {}).get('intent_classifier', 'intent_classifier.pkl')
        intent_clf.save_model(os.path.join(model_dir, intent_clf_name))
        
        # 槽位填充器训练 - 使用意图分类器预测真实意图
        slot_train_sents, slot_train_labels = split_result['slot']['train']
        slotfiller = SlotFiller(config)
        slot_train_data = []
        
        for sent, label_seq in zip(slot_train_sents, slot_train_labels):
            query = "".join(sent)
            
            # 使用意图分类器预测意图
            query_vec = intent_feat.transform([query])
            predicted_intent = intent_clf.predict(query_vec)[0]
            
            # 从CoNLL标签中提取槽位信息
            extracted_slots = self._extract_slots_from_conll(sent, label_seq)
            
            slot_train_data.append({
                "query": query, 
                "intent": predicted_intent,  # 使用预测的意图
                "slots": extracted_slots
            })
        
        slotfiller.train(slot_train_data)
        slotfiller.save_model(model_dir)
        self.archive_model()

        # 6. 配置归档
        self.logger.info("[pipeline] 归档配置文件阶段开始")
        self.archive_config()
        # 保证report目录被创建
        self.archive_report()
        self.logger.info("[pipeline] 归档主流程结束")
        for handler in self.logger.handlers:
            handler.flush()
            if hasattr(handler, 'close'):
                handler.close()


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="意图识别训练pipeline执行入口")
    parser.add_argument("--raw", "-r", default="intent_training/data/raw", 
                       help="原始数据目录路径 (默认: intent_training/data/raw)")
    parser.add_argument("--output", "-o", default="intent_training/model_output", 
                       help="模型输出根目录 (默认: intent_training/model_output)")
    parser.add_argument("--config", "-c", default="intent_training/config.yaml", 
                       help="配置文件路径 (默认: intent_training/config.yaml)")
    
    args = parser.parse_args()
    
    # 检查必要文件是否存在
    if not os.path.exists(args.raw):
        print(f"错误: 原始数据目录不存在: {args.raw}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"开始执行意图识别训练pipeline...")
    print(f"原始数据目录: {args.raw}")
    print(f"输出目录: {args.output}")
    print(f"配置文件: {args.config}")
    
    try:
        pipeline = IntentTraining(
            model_output_root=args.output,
            raw_dir=args.raw,
            config_path=args.config
        )
        pipeline.run()
        print("pipeline执行完成!")
    except Exception as e:
        print(f"pipeline执行失败: {e}")
        sys.exit(1)