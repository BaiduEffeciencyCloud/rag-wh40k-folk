import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from datetime import datetime
import logging
from DATAUPLOD.bm25_manager import BM25Manager
import json
import dotenv
from collections import Counter, defaultdict
import re
import jieba
import math
import string

# 加载.env配置
dotenv.load_dotenv()

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 常量优先从.env读取
DEFAULT_MIN_FREQ = int(os.getenv('BM25_MIN_FREQ', '5'))
DEFAULT_MAX_VOCAB_SIZE = int(os.getenv('BM25_MAX_VOCAB_SIZE', '10000'))
DEFAULT_MIN_PMI = float(os.getenv('BM25_MIN_PMI', '3.0'))
DEFAULT_USERDICT_SUFFIX = os.getenv('USERDICT_SUFFIX', '')
DEFAULT_BM25_DICT_SUFFIX = os.getenv('BM25_DICT_SUFFIX', '')
DEFAULT_OUTPUT_DIR = os.getenv('DICT_OUTPUT_DIR', 'document')
DEFAULT_INPUT_EXTS = os.getenv('DICT_INPUT_EXTS', '.md,.txt').split(',')

STOPWORDS = set([
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "的", "是", "不是", "了", "在", "和", "与", "或", "也", "但", "而", "被", "有", "无", "为", "于", "其", "这", "那", "一个", "一些", "自己"
])

# 过滤引擎
class FilterEngine:
    def __init__(self):
        self.filters = []
    def append(self, func):
        self.filters.append(func)
    def __call__(self, token):
        for f in self.filters:
            if not f(token):
                return False
        return True

# 标点符号过滤
def filter_punct(token):
    if not token:
        return False
    if all(c in string.punctuation for c in token):
        return False
    if re.match(r'^[\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F]+$', token):
        return False
    return True

# markdown转义符过滤
MARKDOWN_SYMBOLS = set(['#', '*', '-', '`', '>', '[', ']', '(', ')', '!', '_', '~', '|', ':'])
def filter_markdown(token):
    if not token:
        return False
    if all(c in MARKDOWN_SYMBOLS for c in token):
        return False
    return True

# 人称代词过滤
PRONOUNS = set([
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们"
])
def filter_pronoun(token):
    if token in PRONOUNS:
        return False
    return True

# html标签和特殊转义符过滤
def filter_html_escape(token):
    if not token:
        return False
    # 过滤html标签，如<h1>、h2、h3等
    if re.match(r'^<\/?[a-zA-Z0-9]+.*?>$', token):
        return False
    # 过滤常见转义符，如\n、\r、\t
    if token in ['\n', '\r', '\t']:
        return False
    # 过滤h1、h2、h3等单独出现的标签名
    if token.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'br', 'hr', 'div', 'span', 'p', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot', 'a', 'img', 'code', 'pre', 'blockquote']:
        return False
    return True

# 结构助词过滤（仅过滤单独出现的"的" "地" "得"）
STRUCT_PARTICLES = {"的", "地", "得","是","在","和","或","也","但","而","被","有","无","为","于","其","这","那","一个","一些","自己"}
def filter_struct_particle(token):
    if token in STRUCT_PARTICLES:
        return False
    return True

# 构建过滤引擎
filter_engine = FilterEngine()
filter_engine.append(filter_punct)
filter_engine.append(filter_markdown)
filter_engine.append(filter_pronoun)
filter_engine.append(filter_html_escape)
filter_engine.append(filter_struct_particle)

def get_datetime_suffix():
    return datetime.now().strftime('%y%m%d%H%M')

def collect_texts_from_folder(folder, exts=None):
    """递归遍历文件夹，收集所有文本内容"""
    if exts is None:
        exts = ['.md', '.txt']
    texts = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if any(fname.lower().endswith(e) for e in exts):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception as e:
                    logging.warning(f"读取文件失败: {fpath}, 错误: {e}")
    return texts

def is_valid_token(token, do_filter=True):
    if not do_filter:
        return True
    return filter_engine(token)

def is_bm25_valid_token(token, do_filter=True):
    if not do_filter:
        return True
    return filter_engine(token)

def extract_phrases(texts, min_freq=5, min_pmi=3.0, max_len=6, whitelist=None, do_filter=True):
    # 统计unigram/bigram/trigram
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()
    total_unigrams = 0
    total_bigrams = 0
    total_trigrams = 0
    for text in texts:
        tokens = list(jieba.cut(text))
        unigram_counter.update(tokens)
        total_unigrams += len(tokens)
        for i in range(len(tokens)-1):
            bigram = (tokens[i], tokens[i+1])
            bigram_counter[bigram] += 1
            total_bigrams += 1
        for i in range(len(tokens)-2):
            trigram = (tokens[i], tokens[i+1], tokens[i+2])
            trigram_counter[trigram] += 1
            total_trigrams += 1
    # bigram PMI
    phrase_set = set()
    for (w1, w2), freq in bigram_counter.items():
        if freq < min_freq:
            continue
        if not (is_valid_token(w1, do_filter) and is_valid_token(w2, do_filter)):
            continue
        p_w1 = unigram_counter[w1] / total_unigrams
        p_w2 = unigram_counter[w2] / total_unigrams
        p_w1w2 = freq / total_bigrams
        pmi = math.log(p_w1w2 / (p_w1 * p_w2) + 1e-10)
        if pmi >= min_pmi and 2 <= len(w1+w2) <= max_len:
            phrase = w1 + w2
            if is_valid_token(phrase, do_filter):
                phrase_set.add(phrase)
    # trigram PMI
    for (w1, w2, w3), freq in trigram_counter.items():
        if freq < min_freq:
            continue
        if not (is_valid_token(w1, do_filter) and is_valid_token(w2, do_filter) and is_valid_token(w3, do_filter)):
            continue
        p_w1 = unigram_counter[w1] / total_unigrams
        p_w2 = unigram_counter[w2] / total_unigrams
        p_w3 = unigram_counter[w3] / total_unigrams
        p_w1w2w3 = freq / total_trigrams
        pmi = math.log(p_w1w2w3 / (p_w1 * p_w2 * p_w3) + 1e-10)
        if pmi >= min_pmi and 3 <= len(w1+w2+w3) <= max_len:
            phrase = w1 + w2 + w3
            if is_valid_token(phrase, do_filter):
                phrase_set.add(phrase)
    # 合并白名单
    if whitelist:
        phrase_set.update([p for p in whitelist if is_valid_token(p, do_filter)])
    # 返回排序后的短语
    print("[DEBUG] unigram_counter:", unigram_counter.most_common(10))
    print("[DEBUG] phrases count:", len(phrase_set))
    return sorted(phrase_set)

def load_whitelist(path):
    phrases = set()
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    phrases.add(line.split()[0])
    return phrases

def save_userdict(phrases, output_path, unigram_counter=None, do_filter=True):
    print("[DEBUG] save_userdict, phrases count:", len(phrases))
    max_freq = max(unigram_counter.values()) if unigram_counter else 1000
    freq = max_freq + 1
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in phrases:
            if is_valid_token(p, do_filter):
                f.write(f"{p} {freq} n\n")
    logging.info(f"✅ 保护性词典已保存: {output_path}")

def save_bm25_vocab(bm25_manager, output_path, do_filter=True):
    print("[DEBUG] save_bm25_vocab, vocab size:", len(bm25_manager.vocabulary))
    vocab_counter = Counter()
    for token in bm25_manager.vocabulary:
        if is_bm25_valid_token(token, do_filter):
            vocab_counter[token] = bm25_manager.vocabulary[token]
    if hasattr(bm25_manager, 'term_freqs'):
        freq_dict = bm25_manager.term_freqs
        sorted_vocab = sorted(freq_dict.items(), key=lambda x: -x[1])
        vocab_dict = {token: bm25_manager.vocabulary[token] for token, freq in sorted_vocab if is_bm25_valid_token(token, do_filter)}
    else:
        vocab_dict = {token: bm25_manager.vocabulary[token] for token in bm25_manager.vocabulary if is_bm25_valid_token(token, do_filter)}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ BM25词典已保存: {output_path}")

def read_all_texts(input_folder):
    texts = []
    for root, _, files in os.walk(input_folder):
        for fname in files:
            if fname.endswith('.txt') or fname.endswith('.json') or fname.endswith('.md'):
                fpath = os.path.join(root, fname)
                try:
                    if fname.endswith('.txt') or fname.endswith('.md'):
                        with open(fpath, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                    elif fname.endswith('.json'):
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                texts.extend([str(v) for v in data.values()])
                            elif isinstance(data, list):
                                texts.extend([str(x) for x in data])
                except Exception as e:
                    print(f"[WARN] 读取文件失败: {fpath}, {e}")
    return texts

def main():
    parser = argparse.ArgumentParser(description='自动生成BM25保护性userdict和vocab')
    parser.add_argument('--input-folder', type=str, required=True, help='输入文件夹，递归遍历md/txt/json')
    parser.add_argument('--output-dir', type=str, default='dict', help='产物输出目录')
    parser.add_argument('--filter', action='store_true', default=True, help='是否过滤短语，默认True')
    parser.add_argument('--no-filter', dest='filter', action='store_false', help='不过滤短语')
    parser.add_argument('--min-freq', type=int, default=DEFAULT_MIN_FREQ, help='短语/词频最小阈值')
    parser.add_argument('--min-pmi', type=float, default=DEFAULT_MIN_PMI, help='短语PMI最小阈值')
    parser.add_argument('--max-len', type=int, default=6, help='短语最大长度')
    parser.add_argument('--whitelist', type=str, default=None, help='白名单短语文件路径，可选')
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"[ERROR] 输入文件夹不存在: {args.input_folder}")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    exts = ['.md', '.txt', '.json']
    texts = collect_texts_from_folder(args.input_folder, exts=exts)
    if not texts:
        print("[WARN] 未读取到任何文本，产物为空。")
    whitelist = load_whitelist(args.whitelist)
    phrases = extract_phrases(
        texts,
        min_freq=args.min_freq,
        min_pmi=args.min_pmi,
        max_len=args.max_len,
        whitelist=whitelist,
        do_filter=args.filter
    ) if args.filter else []
    timestamp = datetime.now().strftime('%y%m%d%H%M')
    dict_path = os.path.join(args.output_dir, f'all_docs_dict_{timestamp}.txt')
    vocab_path = os.path.join(args.output_dir, f'bm25_vocab_{timestamp}.json')
    save_userdict(phrases, dict_path)
    # BM25Manager生成vocab
    bm25 = BM25Manager(user_dict_path=dict_path)
    bm25.fit(texts)
    save_bm25_vocab(bm25, vocab_path)
    print(f'[INFO] userdict已保存: {dict_path}')
    print(f'[INFO] vocab已保存: {vocab_path}')

if __name__ == '__main__':
    main() 