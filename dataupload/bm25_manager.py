import os
import jieba
import json
import pickle
import glob
import logging
from rank_bm25 import BM25Okapi
from typing import Dict, List, Any, Optional
from datetime import datetime
from config import PINECONE_MAX_SPARSE_VALUES

logger = logging.getLogger(__name__)

class BM25Manager:
    def __init__(self, k1=1.5, b=0.75, min_freq=5, max_vocab_size=10000, custom_dict_path=None, user_dict_path=None):
        self.k1 = k1
        self.b = b
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.bm25_model = None
        self.vocabulary = {}
        self.corpus_tokens = []
        self.raw_texts = []  # 原始文本列表
        self.is_fitted = False
        self.whitelist_phrases = set()  # 新增：存储白名单短语
        if custom_dict_path and os.path.exists(custom_dict_path):
            jieba.load_userdict(custom_dict_path)
        if user_dict_path and os.path.exists(user_dict_path):
            print(f"[DEBUG] 加载保护性userdict: {user_dict_path}")
            with open(user_dict_path, 'r', encoding='utf-8') as f:
                print("[DEBUG] userdict内容预览:")
                for i, line in enumerate(f):
                    print(line.strip())
                    # 解析白名单短语，支持格式：短语 频率 词性
                    phrase = line.strip().split()[0] if line.strip() else None
                    if phrase:
                        self.whitelist_phrases.add(phrase)
                    if i >= 19:
                        print("... (仅显示前20行)")
                        break
            jieba.load_userdict(user_dict_path)

    def tokenize_chinese(self, text):
        return list(jieba.cut(text))

    def _build_vocabulary(self, corpus_texts):
        from collections import Counter
        all_tokens = []
        for text in corpus_texts:
            all_tokens.extend(self.tokenize_chinese(text))
        counter = Counter(all_tokens)
        # 剪枝：只保留高频词
        most_common = [w for w, c in counter.items() if c >= self.min_freq]
        most_common = most_common[:self.max_vocab_size]
        # ====== 白名单强制保护逻辑 ======
        # 业务背景：白名单（保护性词典）中的短语，属于业务兜底、必须整体召回的高优先级token。
        # 如果仅靠频率剪枝，低频/未出现的白名单短语会被遗漏，导致检索漏召回，违背保护性设计初衷。
        # 因此：无论白名单短语在语料中出现频率如何，都必须强制纳入vocabulary。
        for phrase in getattr(self, 'whitelist_phrases', set()):
            if phrase not in most_common:
                most_common.append(phrase)
        self.vocabulary = {w: i for i, w in enumerate(most_common)}

    def fit(self, corpus_texts):
        self.raw_texts = list(corpus_texts)
        self._build_vocabulary(self.raw_texts)
        # ====== 白名单短语虚拟文本兜底 ======
        # 业务背景：BM25Okapi 只会为 fit 语料中出现过的 token 计算 idf。
        # 如果白名单短语仅被强制加入 vocabulary，但 fit 语料未出现，则 idf 不会分配，导致稀疏向量链路断裂。
        # 工程兜底：如有白名单短语未出现在语料中，则追加一条“虚拟文本”，内容为所有未出现的白名单短语，用空格拼接。
        # 这样 BM25Okapi 的 idf 统计能覆盖所有白名单短语，且对正常权重分布影响极小。
        missing_phrases = [p for p in getattr(self, 'whitelist_phrases', set()) if all(p not in text for text in self.raw_texts)]
        if missing_phrases:
            self.raw_texts.append(' '.join(missing_phrases))
        self.corpus_tokens = [self.tokenize_chinese(text) for text in self.raw_texts]
        self.bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        self.is_fitted = True

    def incremental_update(self, new_corpus_texts, storage=None):
        """
        增量训练：持久化+全量重建
        storage: RawChunkStorage实例，若为None则不做本地持久化
        """
        if storage is not None:
            # 1. 读取历史切片
            history_chunks = storage.load()
            # 2. 合并新切片并去重
            all_chunks = list(dict.fromkeys(history_chunks + new_corpus_texts))
            # 3. 保存全量切片
            storage.storage(new_corpus_texts)
        else:
            # 兼容老逻辑
            all_chunks = self.raw_texts + new_corpus_texts
        # 4. 用全量切片重建BM25模型
        self.raw_texts = all_chunks
        self.corpus_tokens = [self.tokenize_chinese(text) for text in self.raw_texts]
        self._build_vocabulary(self.raw_texts)
        self.bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        self.is_fitted = True

    def get_sparse_vector(self, text):
        if not self.is_fitted:
            raise ValueError("BM25Manager未训练，请先fit语料库")
        tokens = self.tokenize_chinese(text)
        logger.info("拆分后的query是:" +str(tokens))
        indices, values = [], []
        avgdl = self.bm25_model.avgdl
        k1, b = self.bm25_model.k1, self.bm25_model.b
        dl = len(tokens)
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        for token, freq in token_freq.items():
            if token in self.vocabulary:
                idf = self.bm25_model.idf.get(token, 0)
                tf = freq
                denom = tf + k1 * (1 - b + b * dl / avgdl)
                score = idf * tf * (k1 + 1) / denom
                if score > 0:
                    indices.append(self.vocabulary[token])
                    values.append(score)
        logger.info(f"[DEBUG] sparse indices: {indices}, values: {values}")
        # 追加写入本地 query_sparse.log
        with open("query_sparse.log", "a", encoding="utf-8") as f:
            f.write(f"[DEBUG] sparse indices: {indices}, values: {values}\n")
        # 防御性检查：indices和values必须长度一致，这是稀疏向量的基本要求。
        # indices代表词典中token的id，values代表对应的权重，二者必须一一对应。
        # 注意：tokens（分词结果）和indices长度可能不一致，因为只有词典内的token才会分配index。
        if len(indices) != len(values):
            raise Exception(f"indices和values长度不一致: indices={len(indices)}, values={len(values)}")
        
        # 防御性检查：indices长度不能超过PINECONE_MAX_SPARSE_VALUES，否则Pinecone会拒绝。
        # 一旦超长说明chunk过大或词典过密，需要优化分块策略或词典配置。
        if len(indices) > PINECONE_MAX_SPARSE_VALUES:
            raise Exception(f"生成的稀疏向量长度超过PINECONE_MAX_SPARSE_VALUES ({PINECONE_MAX_SPARSE_VALUES}): {len(indices)}")
        
        # 防御性检查：indices不能为负数，这违反了稀疏向量的基本语义。
        # 一旦出现负数说明词典数据污染或上游bug，直接raise异常，防止脏数据流入下游。
        if any(i < 0 for i in indices):
            raise Exception(f"检测到非法负数index: {indices}")
        
        return {"indices": indices, "values": values}

    def save_model(self, model_path, vocab_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.bm25_model, f)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, ensure_ascii=False)

    def load_model(self, model_path, vocab_path):
        with open(model_path, 'rb') as f:
            self.bm25_model = pickle.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)
        self.is_fitted = True

    # ===== 重构目标：新增方法 =====
    
    def get_latest_dict_file(self, dict_dir: str = "dict") -> str:
        """查找dict目录下最新的bm25_vocab_*.json文件（排除带freq的文件）"""
        if not os.path.exists(dict_dir):
            return None
        
        # 查找所有bm25_vocab_*.json文件，但排除带freq的文件
        vocab_pattern = os.path.join(dict_dir, "bm25_vocab_*.json")
        vocab_files = glob.glob(vocab_pattern)
        
        # 过滤掉包含freq的文件
        vocab_files = [f for f in vocab_files if 'freq' not in os.path.basename(f)]
        
        if not vocab_files:
            return None
        
        # 按文件名中的时间戳排序，取最新的
        vocab_files.sort(reverse=True)
        return vocab_files[0]
    
    def get_latest_freq_dict_file(self, dict_dir: str = "dict") -> str:
        """查找dict目录下最新的bm25_vocab_freq_*.json文件"""
        if not os.path.exists(dict_dir):
            return None
        
        # 查找所有bm25_vocab_freq_*.json文件
        freq_pattern = os.path.join(dict_dir, "bm25_vocab_freq_*.json")
        freq_files = glob.glob(freq_pattern)
        
        if not freq_files:
            return None
        
        # 按文件名中的时间戳排序，取最新的
        freq_files.sort(reverse=True)
        return freq_files[0]
    
    def load_dictionary(self, dict_file_path: str) -> dict:
        """加载词典文件 - 根据文件名判断格式"""
        try:
            with open(dict_file_path, 'r', encoding='utf-8') as f:
                dict_data = json.load(f)
            
            # 根据文件名判断格式
            if 'freq' in os.path.basename(dict_file_path):
                # 带词频格式：提取weight值
                self.vocabulary = {word: data['weight'] for word, data in dict_data.items()}
                logger.info("检测到带词频格式词典，提取weight值")
            else:
                # 传统格式：直接使用
                if 'vocabulary' in dict_data:
                    self.vocabulary = dict_data['vocabulary']
                else:
                    self.vocabulary = dict_data
                logger.info("检测到传统格式词典")
            
            logger.info(f"成功加载词典，词汇量: {len(self.vocabulary)}")
            return dict_data
        except Exception as e:
            raise Exception(f"加载词典失败: {e}")
    
    def generate_sparse_vectors(self, texts: list, config=None) -> list:
        """为文本列表生成sparse向量 - 支持参数配置"""
        if not self.vocabulary:
            raise Exception("词典未加载，请先加载词典")
        
        # 如果提供了配置，创建新实例以支持参数调优（evaltool的方式）
        if config:
            logger.info("🔄 使用配置参数创建新的BM25Manager实例")
            # 创建新的BM25Manager实例，使用配置参数
            bm25 = BM25Manager(
                min_freq=config.min_freq,
                max_vocab_size=config.max_vocab_size
            )
            bm25.vocabulary = self.vocabulary
            # 重新训练以应用新参数
            bm25.fit(texts)
            logger.info("✅ 新BM25Manager实例训练完成")
        else:
            # 使用当前实例（upsert.py的方式）
            logger.info("📝 使用当前BM25Manager实例")
            bm25 = self
            # 如果模型未训练，进行训练
            if not bm25.is_fitted:
                logger.info("🔄 开始训练BM25模型...")
                bm25.fit(texts)
                logger.info("✅ BM25模型训练完成")
        
        # 生成稀疏向量
        sparse_vectors = []
        for text in texts:
            try:
                sparse_vector = bm25.get_sparse_vector(text)
                sparse_vectors.append(sparse_vector)
            except Exception as e:
                logger.warning(f"生成sparse向量失败: {e}")
                sparse_vectors.append({'indices': [], 'values': []})
        return sparse_vectors
    
    def analyze_vocabulary_coverage(self, texts: list) -> dict:
        """分析词汇表覆盖率 - 参考evaltool的实现"""
        if not texts:
            return {
                'total_tokens': 0,
                'covered_tokens': 0,
                'coverage_rate': 0,
                'missing_tokens': []
            }
        
        import jieba
        all_tokens = []
        for text in texts:
            tokens = list(jieba.cut(text))
            all_tokens.extend(tokens)
        
        total_tokens = len(all_tokens)
        unique_tokens = set(all_tokens)
        covered_tokens = [token for token in unique_tokens if token in self.vocabulary]
        missing_tokens = [token for token in unique_tokens if token not in self.vocabulary]
        coverage_rate = len(covered_tokens) / len(unique_tokens) if unique_tokens else 0
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens': len(unique_tokens),
            'covered_tokens': len(covered_tokens),
            'coverage_rate': coverage_rate,
            'missing_tokens': missing_tokens[:100]
        }
    
    def get_vocabulary_stats(self) -> dict:
        """获取词汇统计信息 - 与evaltool保持一致"""
        if not self.vocabulary:
            return {
                'vocabulary_size': 0,
                'avg_frequency': 0,
                'max_frequency': 0,
                'min_frequency': 0
            }
        
        # 注意：这里假设vocabulary的值是频率，如果不是，需要调整逻辑
        # 我不知道您的词典文件中vocabulary的值具体是什么含义，请确认
        frequencies = list(self.vocabulary.values())
        
        return {
            'vocabulary_size': len(self.vocabulary),
            'avg_frequency': sum(frequencies) / len(frequencies),
            'max_frequency': max(frequencies),
            'min_frequency': min(frequencies)
        }
    
    def generate_dictionary(self, min_freq=None, max_vocab_size=None, input_dir=None) -> str:
        """生成词典文件 - 调用gen_userdict.py"""
        import subprocess
        import sys
        
        # 构建命令参数
        cmd = [sys.executable, '-m', 'dataupload.gen_userdict']
        
        if min_freq is not None:
            cmd.extend(['--min-freq', str(min_freq)])
        if max_vocab_size is not None:
            cmd.extend(['--max-vocab-size', str(max_vocab_size)])
        if input_dir is not None:
            cmd.extend(['--input-dir', input_dir])
        
        try:
            logger.info(f"执行词典生成命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("词典生成成功")
            
            # 返回生成的词典文件路径
            dict_file = self.get_latest_dict_file()
            if dict_file:
                logger.info(f"生成的词典文件: {dict_file}")
                return dict_file
            else:
                raise Exception("无法找到生成的词典文件")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"词典生成失败: {e}")
            logger.error(f"错误输出: {e.stderr}")
            raise Exception(f"词典生成失败: {e.stderr}")
        except Exception as e:
            logger.error(f"词典生成异常: {e}")
            raise Exception(f"词典生成异常: {e}") 