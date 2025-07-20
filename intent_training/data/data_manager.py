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
        CoNLL格式预处理：自动修正各种格式错误
        
        Args:
            file_path: 原始CoNLL文件路径
            
        Returns:
            预处理后的文件内容字符串
        """
        logger.info(f"开始CoNLL格式预处理: {file_path}")
        
        processed_lines = []
        current_sentence_lines = []
        
        def process_sentence_lines(sentence_lines):
            """处理单个句子的行"""
            if not sentence_lines:
                return []
            
            processed_sentence = []
            for i, line in enumerate(sentence_lines):
                line = line.strip()
                if line == '':
                    continue
                
                # 1. 处理只有label没有token的行
                # 检查是否只有标签没有token（如：B-UNIT_TYPE）
                parts_by_space = line.split()
                if len(parts_by_space) == 1:
                    # 只有一部分，需要判断是token还是label
                    word = parts_by_space[0]
                    if word.startswith('B-') or word.startswith('I-') or word.startswith('O'):
                        # 这是标签，但没有对应的token，删除这一行
                        logger.warning(f"删除只有label没有token的行: {line}")
                        continue
                    # 如果是以特定关键词开头的描述性文本，也删除
                    elif word in ['只有label没有token', 'B-UNIT_TYPE']:
                        logger.warning(f"删除描述性文本行: {line}")
                        continue
                    # 其他情况：可能是token，需要添加O标签
                    else:
                        # 这是token，但没有label，添加O标签
                        processed_sentence.append(f"{word}\tO")
                        logger.info(f"为只有token的行添加O标签: {word}")
                        continue
                
                # 2. 统一分隔符：将多个空格转换为制表符
                if '\t' in line:
                    # 已经有制表符，直接使用
                    parts = line.split('\t')
                else:
                    # 使用空格分割
                    parts = line.split()
                
                if len(parts) == 0:
                    # 空行
                    continue
                elif len(parts) == 1:
                    # 只有token没有label
                    token = parts[0]
                    processed_sentence.append(f"{token}\tO")
                    logger.info(f"为只有token的行添加O标签: {token}")
                elif len(parts) >= 2:
                    # 有token和label
                    token = parts[0]
                    label = parts[1]
                    
                    # 3. 将0标签转换为O
                    if label == '0':
                        label = 'O'
                        logger.info(f"将0标签转换为O: {token}")
                    
                    # 4. 处理缺少前缀的标签
                    if not label.startswith('B-') and not label.startswith('I-') and not label.startswith('O'):
                        label = f"B-{label}"
                        logger.info(f"为缺少前缀的标签添加B-: {token} -> {label}")
                    
                    # 5. 处理不匹配的I标签（前面没有B或I标签时，改为B标签）
                    if label.startswith('I-'):
                        entity_type = label[2:]
                        should_convert = False
                        
                        if len(processed_sentence) == 0:
                            # 第一个标签就是I标签，需要转换
                            should_convert = True
                        else:
                            prev_line = processed_sentence[-1].strip()
                            if prev_line:
                                prev_parts = prev_line.split('\t')
                                if len(prev_parts) >= 2:
                                    prev_label = prev_parts[1]
                                    if prev_label.startswith('B-') or prev_label.startswith('I-'):
                                        # 前一行是B或I标签，检查实体类型是否一致
                                        prev_entity_type = prev_label[2:]
                                        if prev_entity_type != entity_type:
                                            # 实体类型不一致，需要转换
                                            should_convert = True
                                    else:
                                        # 前一行不是B或I标签，需要转换
                                        should_convert = True
                        
                        if should_convert:
                            label = f"B-{entity_type}"
                            logger.info(f"将不匹配的I标签改为B标签: {token} -> {label}")
                    
                    processed_sentence.append(f"{token}\t{label}")
                else:
                    # 其他情况，保持原样
                    processed_sentence.append(line)
            
            return processed_sentence
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    # 空行表示句子结束，处理当前句子
                    if current_sentence_lines:
                        processed_sentence = process_sentence_lines(current_sentence_lines)
                        processed_lines.extend(processed_sentence)
                        processed_lines.append('')  # 添加空行分隔
                        current_sentence_lines = []
                else:
                    current_sentence_lines.append(line)
            
            # 处理最后一个句子
            if current_sentence_lines:
                processed_sentence = process_sentence_lines(current_sentence_lines)
                processed_lines.extend(processed_sentence)
        
        return '\n'.join(processed_lines)

    def load_slot_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        加载槽位填充数据
        
        Args:
            file_path: CoNLL文件路径
            
        Returns:
            (sentences, labels) 句子列表和标签列表的元组
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误或标签格式错误
        """
        logger.info(f"加载槽位填充数据: {file_path}")
        
        # 文件存在性检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # CoNLL格式预处理
        processed_content = self._preprocess_conll_format(file_path)
        
        # CoNLL文件解析
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []
        
        try:
            for line in processed_content.split('\n'):
                line = line.strip()
                if line == '':
                    # 空行表示句子结束
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    # 解析token和label
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[1]
                        current_sentence.append(token)
                        current_labels.append(label)
                    else:
                        raise ValueError(f"CoNLL格式错误: {line}")
            
            # 处理最后一个句子
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
        
        except Exception as e:
            raise ValueError(f"CoNLL文件解析失败: {e}")
        
        # 数据验证
        self._validate_slot_data(sentences, labels)
        
        # 数据清洗
        cleaned_sentences, cleaned_labels = self._clean_slot_data(sentences, labels)
        
        # 保存到实例变量
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
        """
        cleaned_sentences = []
        cleaned_labels = []
        
        for sentence, label_seq in zip(sentences, labels):
            # 空句子过滤
            if not sentence or len(sentence) == 0:
                continue
            
            # 长度过滤（去除过短的句子）
            if len(sentence) < 2:
                continue
            
            # 长度一致性检查
            if len(sentence) != len(label_seq):
                continue
            
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
    
    def split_data(self, data_type: str = 'both') -> Dict:
        """
        数据切分
        
        Args:
            data_type: 数据类型，'intent', 'slot', 'both'
            
        Returns:
            包含训练/验证/测试集的字典
        """
        logger.info(f"开始数据切分: {data_type}")
        
        result = {}
        
        # 意图数据切分
        if data_type in ['intent', 'both']:
            if self.intent_data is not None:
                result['intent'] = self._split_intent_data()
            else:
                logger.warning("意图数据未加载，跳过意图数据切分")
        
        # 槽位数据切分
        if data_type in ['slot', 'both']:
            if self.slot_data is not None:
                result['slot'] = self._split_slot_data()
            else:
                logger.warning("槽位数据未加载，跳过槽位数据切分")
        
        return result
    
    def _split_intent_data(self) -> Dict:
        """
        切分意图识别数据
        
        Returns:
            包含训练/验证/测试集的字典
        """
        X = self.intent_data['query'].tolist()
        y = self.intent_data['intent'].tolist()
        
        # 获取配置参数
        test_size = self.test_size
        val_size = self.val_size
        random_state = self.random_state
        stratify = self.stratify
        
        # 首先划分出测试集
        if stratify:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # 从剩余数据中划分验证集
        val_size_adjusted = val_size / (1 - test_size)
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
            )
        
        # 计算元数据
        train_metadata = self._calculate_intent_metadata(X_train, y_train)
        val_metadata = self._calculate_intent_metadata(X_val, y_val)
        test_metadata = self._calculate_intent_metadata(X_test, y_test)
        total_metadata = self._calculate_intent_metadata(X, y)
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'metadata': {
                'n_samples': total_metadata['n_samples'],
                'train': train_metadata,
                'val': val_metadata,
                'test': test_metadata
            }
        }
    
    def _split_slot_data(self) -> Dict:
        """
        切分槽位填充数据
        
        Returns:
            包含训练/验证/测试集的字典
        """
        sentences, labels = self.slot_data
        
        # 获取配置参数
        test_size = self.test_size
        val_size = self.val_size
        random_state = self.random_state
        
        # 计算切分索引
        n_samples = len(sentences)
        test_size_abs = int(n_samples * test_size)
        val_size_abs = int(n_samples * val_size)
        
        # 随机打乱索引
        indices = list(range(n_samples))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # 划分索引
        test_indices = indices[:test_size_abs]
        val_indices = indices[test_size_abs:test_size_abs + val_size_abs]
        train_indices = indices[test_size_abs + val_size_abs:]
        
        # 根据索引切分数据
        train_sentences = [sentences[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_sentences = [sentences[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        test_sentences = [sentences[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        # 计算元数据
        train_metadata = self._calculate_slot_metadata(train_sentences, train_labels)
        val_metadata = self._calculate_slot_metadata(val_sentences, val_labels)
        test_metadata = self._calculate_slot_metadata(test_sentences, test_labels)
        total_metadata = self._calculate_slot_metadata(sentences, labels)
        
        return {
            'train': (train_sentences, train_labels),
            'val': (val_sentences, val_labels),
            'test': (test_sentences, test_labels),
            'metadata': {
                'n_sentences': total_metadata['n_sentences'],
                'train': train_metadata,
                'val': val_metadata,
                'test': test_metadata
            }
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