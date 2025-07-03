import os
import jieba
import json
import pickle
from rank_bm25 import BM25Okapi

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
        if custom_dict_path and os.path.exists(custom_dict_path):
            jieba.load_userdict(custom_dict_path)
        if user_dict_path and os.path.exists(user_dict_path):
            print(f"[DEBUG] 加载保护性userdict: {user_dict_path}")
            with open(user_dict_path, 'r', encoding='utf-8') as f:
                print("[DEBUG] userdict内容预览:")
                for i, line in enumerate(f):
                    print(line.strip())
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
        self.vocabulary = {w: i for i, w in enumerate(most_common)}

    def fit(self, corpus_texts):
        self.raw_texts = list(corpus_texts)
        self.corpus_tokens = [self.tokenize_chinese(text) for text in self.raw_texts]
        self._build_vocabulary(self.raw_texts)
        self.bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        self.is_fitted = True

    def incremental_update(self, new_corpus_texts):
        self.raw_texts.extend(new_corpus_texts)
        self.corpus_tokens = [self.tokenize_chinese(text) for text in self.raw_texts]
        self._build_vocabulary(self.raw_texts)
        self.bm25_model = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        self.is_fitted = True

    def get_sparse_vector(self, text):
        if not self.is_fitted:
            raise ValueError("BM25Manager未训练，请先fit语料库")
        tokens = self.tokenize_chinese(text)
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
                weight = idf * (tf * (k1 + 1)) / denom if denom > 0 else 0
                if weight > 0:
                    indices.append(self.vocabulary[token])
                    values.append(float(weight))
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

    def get_vocabulary_stats(self):
        return {"vocab_size": len(self.vocabulary), "example": list(self.vocabulary.items())[:10]} 