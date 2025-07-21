import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger(__name__)

class DataManager:
    """数据管理模块"""
    
    def __init__(self, config: Dict):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典，包含test_size, val_size, random_state, stratify等参数
        """
        self.config = config
        self.intent_data = None
        self.slot_data = None
        
        # 验证配置参数
        self._validate_config()
        
    def load_intent_data(self, file_path: str) -> pd.DataFrame:
        """
        加载意图识别数据
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            加载并清洗后的DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误或缺少必需列
        """
        logger.info(f"加载意图识别数据: {file_path}")
        
        # 文件存在性检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # CSV文件读取
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"CSV文件读取失败: {e}")
        
        # 必需列验证
        required_columns = ['query', 'intent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必需列: {missing_columns}")
        
        # 数据清洗
        cleaned_df = self._clean_intent_data(df)
        
        # 保存到实例变量
        self.intent_data = cleaned_df
        
        return cleaned_df
    
    def _preprocess_conll_format(self, file_path: str) -> str:
        """
        CoNLL格式预处理：自动修正各种格式错误，并记录每一条修改日志
        """
        logger.info(f"开始CoNLL格式预处理: {file_path}")
        processed_lines = []
        current_sentence_lines = []
        def process_sentence_lines(sentence_lines):
            if not sentence_lines:
                return []
            processed_sentence = []
            for i, line in enumerate(sentence_lines):
                raw_line = line
                line = line.strip()
                # 1. 处理只有label没有token的行
                parts_by_space = line.split()
                if len(parts_by_space) == 1:
                    word = parts_by_space[0]
                    if word.startswith('B-') or word.startswith('I-') or word.startswith('O'):
                        logger.warning(f"删除只有label没有token的行: 原始行='{raw_line}'")
                        continue
                    elif word in ['只有label没有token', 'B-UNIT_TYPE']:
                        logger.warning(f"删除描述性文本行: 原始行='{raw_line}'")
                        continue
                    else:
                        logger.info(f"为只有token的行添加O标签: 原始行='{raw_line}' -> 修正后='{word}\tO'")
                        processed_sentence.append(f"{word}\tO")
                        continue
                # 2. 统一分隔符：将多个空格转换为制表符
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        logger.info(f"空格分隔符转制表符: 原始行='{raw_line}' -> 修正后='{parts[0]}\t{parts[1]}'")
                if len(parts) == 0:
                    continue
                elif len(parts) == 1:
                    token = parts[0]
                    logger.info(f"为只有token的行添加O标签: 原始行='{raw_line}' -> 修正后='{token}\tO'")
                    processed_sentence.append(f"{token}\tO")
                elif len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    orig_label = label
                    # 3. 将0标签转换为O
                    if label == '0':
                        logger.info(f"0标签转O: 原始行='{raw_line}' -> 修正后='{token}\tO'")
                        label = 'O'
                    # 4. 处理缺少前缀的标签
                    if not label.startswith('B-') and not label.startswith('I-') and not label.startswith('O'):
                        logger.info(f"缺少前缀标签加B-: 原始行='{raw_line}' -> 修正后='{token}\tB-{label}'")
                        label = f"B-{label}"
                    # 5. 处理不匹配的I标签（前面没有B或I标签或实体类型不一致时，改为B标签）
                    if label.startswith('I-'):
                        entity_type = label[2:]
                        should_convert = False
                        if len(processed_sentence) == 0:
                            should_convert = True
                        else:
                            prev_line = processed_sentence[-1].strip()
                            if prev_line:
                                prev_parts = prev_line.split('\t')
                                if len(prev_parts) >= 2:
                                    prev_label = prev_parts[1]
                                    if prev_label.startswith('B-') or prev_label.startswith('I-'):
                                        prev_entity_type = prev_label[2:]
                                        if prev_entity_type != entity_type:
                                            should_convert = True
                                    else:
                                        should_convert = True
                        if should_convert:
                            logger.info(f"I标签不连续或实体类型不一致转B-: 原始行='{raw_line}' -> 修正后='{token}\tB-{entity_type}'")
                            label = f"B-{entity_type}"
                    processed_sentence.append(f"{token}\t{label}")
                else:
                    processed_sentence.append(line)
            return processed_sentence
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    if current_sentence_lines:
                        processed_sentence = process_sentence_lines(current_sentence_lines)
                        processed_lines.extend(processed_sentence)
                        processed_lines.append('')
                        current_sentence_lines = []
                else:
                    current_sentence_lines.append(line)
            if current_sentence_lines:
                processed_sentence = process_sentence_lines(current_sentence_lines)
                processed_lines.extend(processed_sentence)
        return '\n'.join(processed_lines)

    def overwrite_conll_file(self, file_path: str, fixed_content: str) -> None:
        """
        用修正后的内容原子性覆盖CoNLL文件，保留.bak备份。
        如果校验失败则抛出异常，原文件不被破坏。
        Args:
            file_path: 原始CoNLL文件路径
            fixed_content: 修正后的内容字符串
        """
        logger.info(f"[overwrite_conll_file] 开始备份并覆盖: {file_path}")
        bak_path = file_path + ".bak"
        # 1. 写入.bak文件
        with open(bak_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        logger.info(f"[overwrite_conll_file] 修正内容已写入备份: {bak_path}")
        # 2. 校验（尝试重新解析，确保无格式错误）
        try:
            for line in fixed_content.split('\n'):
                if line.strip() == '':
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    raise ValueError(f"修正后仍有格式错误: {line}")
        except Exception as e:
            logger.error(f"[overwrite_conll_file] 校验失败，保留.bak: {e}")
            raise
        # 3. 原子性覆盖原文件
        try:
            os.replace(bak_path, file_path)
            logger.info(f"[overwrite_conll_file] 已原子性覆盖原文件: {file_path}")
        except Exception as e:
            logger.error(f"[overwrite_conll_file] 覆盖原文件失败: {e}")
            raise

    def load_slot_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        加载槽位填充数据
        Args:
            file_path: CoNLL文件路径
        Returns:
            (sentences, labels) 句子和标签列表
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误或数据质量问题
        """
        logger.info(f"加载槽位填充数据: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        # 修正并覆盖原文件，获取修正后的内容
        processed_content = self._preprocess_conll_format(file_path)
        # 修正后覆写原文件
        self.overwrite_conll_file(file_path, processed_content)
        # 解析CoNLL格式
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
        try:
            for line in processed_content.split('\n'):
                line = line.strip()
                if line == '':
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_sentence.append(token)
                        current_labels.append(label)
                    else:
                        raise ValueError(f"CoNLL格式错误: {line}")
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
        except Exception as e:
            raise ValueError(f"CoNLL文件解析失败: {e}")
        # 检查是否有有效数据
        if not sentences:
            raise ValueError("CoNLL文件中没有有效的句子数据")
        # 验证数据格式
        self._validate_slot_data(sentences, labels)
        # 清洗数据
        cleaned_sentences, cleaned_labels = self._clean_slot_data(sentences, labels)
        # 检查清洗后的数据质量
        if not cleaned_sentences:
            raise ValueError("数据清洗后没有有效的句子，可能存在严重的数据质量问题")
        # 检查清洗前后数据量差异
        original_count = len(sentences)
        cleaned_count = len(cleaned_sentences)
        if cleaned_count < original_count * 0.5:
            raise ValueError(f"数据清洗后损失过多数据: 原始{original_count}句，清洗后{cleaned_count}句")
        self.slot_data = (cleaned_sentences, cleaned_labels)
        return cleaned_sentences, cleaned_labels

    def _clean_intent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗意图识别数据
        
        Args:
            df: 原始DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        # 复制数据避免修改原始数据
        cleaned_df = df.copy()
        
        # 空值处理
        cleaned_df = cleaned_df.dropna(subset=['query', 'intent'])
        
        # 去除空字符串
        cleaned_df = cleaned_df[cleaned_df['query'].str.strip() != '']
        cleaned_df = cleaned_df[cleaned_df['intent'].str.strip() != '']
        
        # 长度过滤（去除过短的查询）
        cleaned_df = cleaned_df[cleaned_df['query'].str.len() >= 2]
        
        # 去重逻辑
        cleaned_df = cleaned_df.drop_duplicates(subset=['query', 'intent'])
        
        # 重置索引
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        return cleaned_df
    
    def _clean_slot_data(self, sentences: List[List[str]], labels: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        清洗槽位填充数据
        Args:
            sentences: 句子列表
            labels: 标签列表
        Returns:
            (cleaned_sentences, cleaned_labels) 清洗后的句子和标签列表
        Raises:
            ValueError: 数据质量问题严重
        """
        cleaned_sentences = []
        cleaned_labels = []
        error_count = 0
        for i, (sentence, label_seq) in enumerate(zip(sentences, labels)):
            # 空句子过滤
            if not sentence or len(sentence) == 0:
                error_count += 1
                continue
            # 长度过滤（去除过短的句子）
            if len(sentence) < 2:
                error_count += 1
                continue
            # 长度一致性检查 - 这是严重错误，应该抛出异常
            if len(sentence) != len(label_seq):
                raise ValueError(f"第{i}个句子长度({len(sentence)})与标签长度({len(label_seq)})不一致")
            # 去除空token
            filtered_sentence = []
            filtered_labels = []
            for token, label in zip(sentence, label_seq):
                if token.strip():  # 非空token
                    filtered_sentence.append(token.strip())
                    filtered_labels.append(label)
            # 过滤后再次检查长度
            if len(filtered_sentence) >= 2:
                cleaned_sentences.append(filtered_sentence)
                cleaned_labels.append(filtered_labels)
            else:
                error_count += 1
        # 如果错误率过高，抛出异常
        total_count = len(sentences)
        if total_count > 0 and error_count / total_count > 0.3:
            raise ValueError(f"数据质量过差: 总{total_count}句，错误{error_count}句，错误率{error_count/total_count:.1%}")
        return cleaned_sentences, cleaned_labels
    
    def _validate_config(self):
        """
        验证配置参数
        Raises:
            ValueError: 配置参数无效
        """
        # 支持分组后的config结构
        data_cfg = self.config.get('data', self.config)
        test_size = data_cfg.get('test_size', 0.2)
        val_size = data_cfg.get('val_size', 0.1)
        random_state = data_cfg.get('random_state', 42)
        stratify = data_cfg.get('stratify', True)
        # 参数范围校验
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be in range (0, 1), got {test_size}")
        if not (0 <= val_size < 1):
            raise ValueError(f"val_size must be in range [0, 1), got {val_size}")
        if test_size + val_size >= 1:
            raise ValueError(f"test_size + val_size must be less than 1, got {test_size + val_size}")
        if not isinstance(random_state, int):
            raise ValueError(f"random_state must be an integer, got {type(random_state)}")
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify
    
    def _validate_slot_data(self, sentences: List[List[str]], labels: List[List[str]]):
        """
        验证槽位填充数据
        
        Args:
            sentences: 句子列表
            labels: 标签列表
            
        Raises:
            ValueError: 数据格式错误
        """
        # 句子和标签数量一致性检查
        if len(sentences) != len(labels):
            raise ValueError(f"句子数量({len(sentences)})与标签数量({len(labels)})不一致")
        
        # 长度一致性检查和BIO标签格式检查
        valid_labels = {'O'}  # 基础标签
        
        for i, (sentence, label_seq) in enumerate(zip(sentences, labels)):
            # 长度一致性检查
            if len(sentence) != len(label_seq):
                raise ValueError(f"第{i}个句子长度({len(sentence)})与标签长度({len(label_seq)})不一致")
            
            # BIO标签格式检查
            for j, label in enumerate(label_seq):
                if label.startswith('B-') or label.startswith('I-'):
                    # 提取实体类型
                    entity_type = label[2:]
                    valid_labels.add(f'B-{entity_type}')
                    valid_labels.add(f'I-{entity_type}')
                elif label != 'O':
                    raise ValueError(f"第{i}个句子第{j}个标签格式错误: {label}")
        
        # 检查BIO标签的连续性
        for i, label_seq in enumerate(labels):
            for j in range(len(label_seq)):
                if label_seq[j].startswith('I-'):
                    # I标签前面必须有B标签或I标签
                    if j == 0 or (not label_seq[j-1].startswith('B-') and not label_seq[j-1].startswith('I-')):
                        raise ValueError(f"第{i}个句子第{j}个I标签前缺少B标签或I标签")
                    # I标签的实体类型必须与前面的B标签一致
                    if j > 0 and label_seq[j-1].startswith('B-'):
                        if label_seq[j-1][2:] != label_seq[j][2:]:
                            raise ValueError(f"第{i}个句子第{j}个I标签实体类型与前面B标签不一致")
    
    def split_data(self, mode: str = 'both'):
        """
        切分意图和槽位数据为训练集、验证集和测试集。
        支持 'intent', 'slot', 'both' 三种模式。
        """
        result = {}
        if mode in ['intent', 'both']:
            result['intent'] = self._split_intent_data()
        if mode in ['slot', 'both']:
            result['slot'] = self._split_slot_data()
        
        # 存储到实例变量中
        if 'intent' in result and 'slot' in result:
            self.train_data = (result['intent']['train'][0], result['intent']['train'][1], result['slot']['train'][1])
            self.test_data = (result['intent']['test'][0], result['intent']['test'][1], result['slot']['test'][1])
            
        return result

    def _split_intent_data(self):
        """
        切分意图识别数据。
        优先尝试分层抽样，如果失败（如样本太少），则自动降级为普通随机抽样。
        """
        X = self.intent_data['query'].tolist()
        y = self.intent_data['intent'].tolist()
        
        test_size = self.config.get('data', {}).get('test_size', 0.2)
        val_size_of_train = self.config.get('data', {}).get('val_size', 0.2)

        def split_with_stratify(features, labels, stratify_labels, **kwargs):
            return train_test_split(features, labels, stratify=stratify_labels, **kwargs)

        def split_without_stratify(features, labels, **kwargs):
            return train_test_split(features, labels, stratify=None, **kwargs)

        # 第一次切分：分离出测试集
        try:
            X_temp, X_test, y_temp, y_test = split_with_stratify(
                X, y, stratify_labels=y, test_size=test_size, random_state=42
            )
        except ValueError:
            logging.warning("分层抽样失败（测试集划分），可能是由于某些类别样本数过少。自动切换到普通随机抽样。")
            X_temp, X_test, y_temp, y_test = split_without_stratify(
                X, y, test_size=test_size, random_state=42
            )

        # 第二次切分：从剩余数据中分离出训练集和验证集
        try:
            X_train, X_val, y_train, y_val = split_with_stratify(
                X_temp, y_temp, stratify_labels=y_temp, test_size=val_size_of_train, random_state=42
            )
        except ValueError:
            logging.warning("分层抽样失败（验证集划分），可能是由于某些类别样本数过少。自动切换到普通随机抽样。")
            X_train, X_val, y_train, y_val = split_without_stratify(
                X_temp, y_temp, test_size=val_size_of_train, random_state=42
            )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def _split_slot_data(self):
        """
        辅助方法：切分槽位数据。
        为了与意图数据保持样本对齐，这里我们不使用随机切分，
        而是假设意图数据已经切分好，并返回同样结构的数据。
        """
        sentences, labels = self.slot_data
        
        # Note: This logic might need to be revisited to perfectly align with intent splits.
        # For now, it performs a simple random split.
        num_samples = len(self.slot_data[0])
        indices = list(range(num_samples))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        test_size_abs = int(num_samples * self.config.get('data', {}).get('test_size', 0.2))
        val_size_abs = int(num_samples * self.config.get('data', {}).get('val_size', 0.2))

        test_indices = indices[:test_size_abs]
        val_indices = indices[test_size_abs:test_size_abs + val_size_abs]
        train_indices = indices[test_size_abs + val_size_abs:]

        X_train = [sentences[i] for i in train_indices]
        y_train = [labels[i] for i in train_indices]
        X_val = [sentences[i] for i in val_indices]
        y_val = [labels[i] for i in val_indices]
        X_test = [sentences[i] for i in test_indices]
        y_test = [labels[i] for i in test_indices]

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def _calculate_intent_metadata(self, X: List[str], y: List[str]) -> Dict:
        """
        计算意图识别数据元数据
        
        Args:
            X: 查询列表
            y: 意图列表
            
        Returns:
            元数据字典
        """
        # 样本数量统计
        n_samples = len(X)
        
        # 意图分布统计
        intent_counts = {}
        for intent in y:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # 查询长度统计
        query_lengths = [len(query) for query in X]
        avg_length = np.mean(query_lengths) if query_lengths else 0
        min_length = min(query_lengths) if query_lengths else 0
        max_length = max(query_lengths) if query_lengths else 0
        
        return {
            'n_samples': n_samples,
            'intent_distribution': intent_counts,
            'n_intents': len(intent_counts),
            'avg_query_length': avg_length,
            'min_query_length': min_length,
            'max_query_length': max_length
        }
    
    def _calculate_slot_metadata(self, sentences: List[List[str]], labels: List[List[str]]) -> Dict:
        """
        计算槽位填充数据元数据
        
        Args:
            sentences: 句子列表
            labels: 标签列表
            
        Returns:
            元数据字典
        """
        # 句子数量统计
        n_sentences = len(sentences)
        
        # 词数统计
        total_tokens = sum(len(sentence) for sentence in sentences)
        avg_tokens_per_sentence = total_tokens / n_sentences if n_sentences > 0 else 0
        
        # 标签分布统计
        label_counts = {}
        for label_seq in labels:
            for label in label_seq:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # 句子长度统计
        sentence_lengths = [len(sentence) for sentence in sentences]
        avg_length = np.mean(sentence_lengths) if sentence_lengths else 0
        min_length = min(sentence_lengths) if sentence_lengths else 0
        max_length = max(sentence_lengths) if sentence_lengths else 0
        
        return {
            'n_sentences': n_sentences,
            'total_tokens': total_tokens,
            'avg_tokens_per_sentence': avg_tokens_per_sentence,
            'label_distribution': label_counts,
            'n_labels': len(label_counts),
            'avg_sentence_length': avg_length,
            'min_sentence_length': min_length,
            'max_sentence_length': max_length
        } 

    def get_train_data(self):
        """
        获取训练数据。

        Returns:
            Tuple[List[str], List[str], List[List[str]]]: 训练集的 (文本, 意图, 槽位)
        """
        if not hasattr(self, 'train_data') or not self.train_data:
            return [], [], []
        return self.train_data

    def get_test_data(self):
        """
        获取测试数据。

        Returns:
            Tuple[List[str], List[str], List[List[str]]]: 测试集的 (文本, 意图, 槽位)
        """
        if not hasattr(self, 'test_data') or not self.test_data:
            return [], [], []
        return self.test_data 