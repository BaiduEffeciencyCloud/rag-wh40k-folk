import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from datetime import datetime
import logging
from dataupload.bm25_manager import BM25Manager
import json
import dotenv
from collections import Counter, defaultdict
import re
import jieba
import math
import string
import ahocorasick
import dataupload.config as global_config
from dataupload.phrase_weight import PhraseWeightScorer, AutoThresholdPhraseWeightScorer
from dataupload.wh40k_whitelist_generator import WH40KWhitelistGenerator
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
STRUCT_PARTICLES = {"的", "地", "得","是","在","和","或","也","但","而","被","有","无","为","于","其","这","那",}
def filter_struct_particle(token):
    if token in STRUCT_PARTICLES:
        return False
    return True

# 数字和空格过滤
def filter_number_and_space(token):
    if not token: return False
    if token.strip().isdigit(): return False  # 只过滤纯数字
    if token.strip() == '': return False
    if token != token.strip(): return False
    return True

# 特殊符号过滤
def filter_special_symbols(token):
    if not token:
        return False
    # 过滤包含特殊符号的词汇（如@、#、$、%等）
    special_chars = set('@#$%^&*()_+-=[]{}|\\:;"\'<>?,./')
    if any(c in special_chars for c in token):
        return False
    return True

# 构建过滤引擎
filter_engine = FilterEngine()
filter_engine.append(filter_punct)
filter_engine.append(filter_markdown)
filter_engine.append(filter_pronoun)
filter_engine.append(filter_html_escape)
filter_engine.append(filter_struct_particle)
filter_engine.append(filter_number_and_space)

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

def extract_phrases(texts, min_freq=5, min_pmi=3.0, max_len=6, do_filter=True):
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
    # 直接返回短语集（不再合并whitelist）
    print("[DEBUG] unigram_counter:", unigram_counter.most_common(10))
    print("[DEBUG] phrases count:", len(phrase_set))
    return sorted(phrase_set)

def generate_whitelist(phrases, output_path, unigram_counter=None, do_filter=True, phrase_df=None, phrase_pmi=None, scorer=None, config=None):
    """
    支持从config注入分档scorer/DF/PMI，优先级：config > 显式参数 > 兜底
    """
    if config:
        if 'phrase_weight_scorer' in config:
            scorer = config['phrase_weight_scorer']
        if 'phrase_df' in config:
            phrase_df = config['phrase_df']
        if 'phrase_pmi' in config:
            phrase_pmi = config['phrase_pmi']
    print("[DEBUG] generate_whitelist, phrases count:", len(phrases))
    max_freq = max(unigram_counter.values()) if unigram_counter else 1000
    idf0 = max_freq + 1
    meta = {}
    
    # 1. 生成完整格式词典（包含频率和词性）
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in phrases:
            if is_valid_token(p, do_filter):
                if scorer and phrase_df and phrase_pmi:
                    df = phrase_df.get(p, 1)
                    pmi = phrase_pmi.get(p, None)
                    weight = int(scorer.get_weight(idf0, df, pmi))
                else:
                    df = phrase_df.get(p, 1) if phrase_df else 1
                    pmi = phrase_pmi.get(p, None) if phrase_pmi else None
                    weight = idf0
                f.write(f"{p} {weight} n\n")
                meta[p] = {"df": df, "pmi": pmi, "weight": weight}
    
    # 2. 生成纯词汇文件（用于OpenSearch synonyms_path）
    warhammer_dict_path = output_path.replace('all_docs_dict_', 'warhammer_dict_').replace('.txt', '.txt')
    with open(warhammer_dict_path, 'w', encoding='utf-8') as f:
        for p in phrases:
            # 只做基本的有效性检查，不过度过滤
            if p and p.strip() and not p.strip().isdigit():
                f.write(f"{p.strip()}\n")
    
    # 保存meta信息
    meta_path = output_path + ".meta.json"
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)
    
    logging.info(f"✅ 保护性词典已保存: {output_path}, meta: {meta_path}")
    logging.info(f"✅ OpenSearch词典已保存: {warhammer_dict_path}")

def save_bm25_vocab(bm25_manager, output_path, do_filter=True, scorer=None, phrase_df=None, phrase_pmi=None):
    print("[DEBUG] save_bm25_vocab, vocab size:", len(bm25_manager.vocabulary))
    
    # 1. 生成传统格式词典（保持向后兼容）
    vocab_dict = {}
    for token in bm25_manager.vocabulary:
        if is_bm25_valid_token(token, do_filter):
            idf = bm25_manager.bm25_model.idf.get(token, 1.0)
            df = phrase_df.get(token, 1) if phrase_df else 1
            pmi = phrase_pmi.get(token, None) if phrase_pmi else None
            if scorer:
                weight = scorer.get_weight(idf, df, pmi)
            else:
                weight = idf
            vocab_dict[token] = weight
    
    # 保存传统格式词典
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ BM25词典已保存: {output_path}")
    
    # 2. 生成带词频格式词典（新增功能）
    freq_vocab_dict = {}
    for token in bm25_manager.vocabulary:
        if is_bm25_valid_token(token, do_filter):
            idf = bm25_manager.bm25_model.idf.get(token, 1.0)
            df = phrase_df.get(token, 1) if phrase_df else 1
            pmi = phrase_pmi.get(token, None) if phrase_pmi else None
            if scorer:
                weight = scorer.get_weight(idf, df, pmi)
            else:
                weight = idf
            
            # 计算词频（在训练文本中的总出现次数）
            freq = 0
            if hasattr(bm25_manager, 'bm25_model') and hasattr(bm25_manager.bm25_model, 'tf'):
                # 从BM25模型的tf中获取词频
                freq = sum(bm25_manager.bm25_model.tf.get(token, {}).values())
            else:
                # 如果没有tf信息，使用df作为freq的估计值
                freq = df
            
            freq_vocab_dict[token] = {
                "weight": weight,
                "freq": freq,
                "df": df
            }
    
    # 生成带词频词典的文件路径
    freq_output_path = output_path.replace('bm25_vocab_', 'bm25_vocab_freq_')
    
    # 保存带词频格式词典
    with open(freq_output_path, 'w', encoding='utf-8') as f:
        json.dump(freq_vocab_dict, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ 带词频BM25词典已保存: {freq_output_path}")

def generate_bm25_manager_from_texts(texts, min_freq=5, min_pmi=3.0, max_len=6, 
                                   do_filter=True, ac_mask=False, auto_threshold=False):
    """
    从文本列表生成BM25Manager实例
    
    Args:
        texts: 文本列表
        min_freq: 短语/词频最小阈值
        min_pmi: 短语PMI最小阈值
        max_len: 短语最大长度
        do_filter: 是否过滤短语
        ac_mask: 是否启用AC自动机短语保护
        auto_threshold: 是否启用自动阈值推荐分档
    
    Returns:
        BM25Manager: 训练好的BM25Manager实例
        dict: 词汇表字典
    """
    if not texts:
        raise ValueError("文本列表不能为空")
    
    # 1. 提取短语
    phrases = extract_phrases(
        texts,
        min_freq=min_freq,
        min_pmi=min_pmi,
        max_len=max_len,
        do_filter=do_filter
    )
    
    # 2. 统计DF
    phrase_df = Counter()
    for text in texts:
        for p in phrases:
            if p in text:
                phrase_df[p] += 1
    
    # 3. PMI统计
    phrase_pmi = {p: None for p in phrases}
    
    # 4. 选择scorer
    if auto_threshold:
        scorer = AutoThresholdPhraseWeightScorer.from_df_values(phrase_df)
    else:
        scorer = global_config.phrase_weight_scorer
    
    # 5. 创建临时词典文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        temp_dict_path = f.name
    
    try:
        # 6. 生成保护性词典
        config_dict = {
            "phrase_weight_scorer": scorer,
            "phrase_df": phrase_df,
            "phrase_pmi": phrase_pmi
        }
        generate_whitelist(phrases, temp_dict_path, unigram_counter=None, do_filter=True, config=config_dict)
        
        # 7. 创建并训练BM25Manager
        bm25 = BM25Manager(user_dict_path=temp_dict_path)
        
        # 8. AC自动机短语保护流程
        if ac_mask and phrases:
            masker = PhraseMasker(phrases)
            masked_texts, intervals_list, phrase2placeholder = masker.mask_texts(texts)
            bm25.fit(masked_texts)
        else:
            bm25.fit(texts)
        
        # 9. 构建词汇表字典 - 按照原始逻辑，固定使用do_filter=True
        vocab_dict = {}
        for token in bm25.vocabulary:
            # 按照原始逻辑，固定使用do_filter=True进行过滤
            if is_bm25_valid_token(token, do_filter=True):
                idf = bm25.bm25_model.idf.get(token, 1.0)
                df = phrase_df.get(token, 1) if phrase_df else 1
                pmi = phrase_pmi.get(token, None) if phrase_pmi else None
                if scorer:
                    weight = scorer.get_weight(idf, df, pmi)
                else:
                    weight = idf
                vocab_dict[token] = weight
        
        return bm25, vocab_dict
        
    finally:
        # 清理临时文件
        try:
            os.unlink(temp_dict_path)
        except:
            pass



class PhraseMasker:
    """
    基于AC自动机的短语屏蔽与还原工具，支持占位符嵌入原始短语。
    """
    def __init__(self, phrases):
        self.automaton = ahocorasick.Automaton()
        self.phrase_map = {}
        self.placeholder_map = {}  # 占位符到短语
        for idx, phrase in enumerate(phrases):
            # 生成唯一占位符，嵌入原始短语
            ph = f"__PHRASE_{idx}_{phrase}__"
            self.automaton.add_word(phrase, (idx, phrase, ph))
            self.phrase_map[phrase] = ph
            self.placeholder_map[ph] = phrase
        self.automaton.make_automaton()

    def mask_phrases(self, text, embed_phrase=True):
        """
        将text中的所有短语替换为占位符，返回masked_text和intervals
        intervals: [(start, end, phrase, placeholder)]
        """
        if not self.phrase_map or len(self.automaton) == 0:
            return text, []
        intervals = []
        for end_pos, (idx, phrase, ph) in self.automaton.iter(text):
            start_pos = end_pos - len(phrase) + 1
            intervals.append((start_pos, end_pos, phrase, ph))
        # 合并重叠区间，优先长短语
        intervals.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        merged = []
        last_end = -1
        for s, e, p, ph in intervals:
            if s > last_end:
                merged.append((s, e, p, ph))
                last_end = e
        masked = []
        last = 0
        for s, e, p, ph in merged:
            masked.append(text[last:s])
            masked.append(ph)
            last = e + 1
        masked.append(text[last:])
        return "".join(masked), merged

    def mask_texts(self, texts, embed_phrase=True):
        """
        批量掩码，返回(masked_texts, intervals_list, phrase2placeholder)
        """
        masked_texts = []
        intervals_list = []
        for text in texts:
            masked, intervals = self.mask_phrases(text, embed_phrase=embed_phrase)
            masked_texts.append(masked)
            intervals_list.append(intervals)
        return masked_texts, intervals_list, self.phrase_map.copy()

    def restore_text(self, masked_text, intervals):
        """
        将占位符还原为原始短语
        """
        restored = masked_text
        for _, _, _, ph in intervals:
            restored = restored.replace(ph, self.placeholder_map[ph])
        return restored

def generate_userdict_and_vocab(input_dir, output_dir, min_freq=5, min_pmi=3.0, max_len=6, 
                               do_filter=True, ac_mask=False, auto_threshold=False, 
                               exts=None, timestamp=None):
    """
    生成用户词典和词汇表的核心函数
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出目录路径
        min_freq: 短语/词频最小阈值
        min_pmi: 短语PMI最小阈值
        max_len: 短语最大长度
        do_filter: 是否过滤短语
        ac_mask: 是否启用AC自动机短语保护
        auto_threshold: 是否启用自动阈值推荐分档
        exts: 文件扩展名列表
        timestamp: 时间戳，如果为None则自动生成
    
    Returns:
        tuple: (dict_path, vocab_path) 生成的词典文件路径和词汇表文件路径
    """
    if exts is None:
        exts = ['.md', '.txt', '.json']
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%y%m%d%H%M')
    
    # 收集文本
    texts = collect_texts_from_folder(input_dir, exts=exts)
    if not texts:
        print("[WARN] 未读取到任何文本，产物为空。")
        return None, None
    
    # 提取短语
    phrases = extract_phrases(
        texts,
        min_freq=min_freq,
        min_pmi=min_pmi,
        max_len=max_len,
        do_filter=do_filter
    )
    
    # 生成文件路径
    dict_path = os.path.join(output_dir, f'all_docs_dict_{timestamp}.txt')
    vocab_path = os.path.join(output_dir, f'bm25_vocab_{timestamp}.json')
    
    # 统计DF
    phrase_df = Counter()
    for text in texts:
        for p in phrases:
            if p in text:
                phrase_df[p] += 1
    
    # PMI统计
    phrase_pmi = {p: None for p in phrases}
    
    # 选择scorer
    if auto_threshold:
        scorer = AutoThresholdPhraseWeightScorer.from_df_values(phrase_df)
    else:
        scorer = global_config.phrase_weight_scorer
    
    config_dict = {
        "phrase_weight_scorer": scorer,
        "phrase_df": phrase_df,
        "phrase_pmi": phrase_pmi
    }
    
    # 生成保护性词典
    if global_config.ENABLE_WH40K_WHITELIST:
        # 使用混合白名单生成（传递配置参数）
        generate_hybrid_whitelist(texts, dict_path, global_config.WH40K_VOCAB_PATH, config=config_dict)
    else:
        # 使用原来的逻辑
        generate_whitelist(phrases, dict_path, unigram_counter=None, do_filter=True, config=config_dict)
    
    # 创建并训练BM25Manager
    bm25 = BM25Manager(user_dict_path=dict_path)
    
    # AC自动机短语保护流程
    if ac_mask and phrases:
        masker = PhraseMasker(phrases)
        masked_texts, intervals_list, phrase2placeholder = masker.mask_texts(texts)
        bm25.fit(masked_texts)
    else:
        bm25.fit(texts)
    
    # 保存词汇表
    save_bm25_vocab(bm25, vocab_path, scorer=scorer, phrase_df=phrase_df, phrase_pmi=phrase_pmi)
    
    return dict_path, vocab_path

def main():
    parser = argparse.ArgumentParser(description='自动生成BM25保护性userdict和vocab')
    parser.add_argument('--input-dir', type=str, required=True, help='输入文件夹，递归遍历md/txt/json')
    parser.add_argument('--output-dir', type=str, default='dict', help='产物输出目录')
    parser.add_argument('--filter', action='store_true', default=False, help='是否过滤短语，默认False')
    parser.add_argument('--min-freq', type=int, default=DEFAULT_MIN_FREQ, help='短语/词频最小阈值')
    parser.add_argument('--min-pmi', type=float, default=DEFAULT_MIN_PMI, help='短语PMI最小阈值')
    parser.add_argument('--max-len', type=int, default=6, help='短语最大长度')
    parser.add_argument('--ac-mask', action='store_true', default=False, help='是否启用AC自动机短语保护')
    parser.add_argument('--auto-threshold', action='store_true', default=False, help='是否启用自动阈值推荐分档')
    args = parser.parse_args()

    # 参数验证
    if not os.path.exists(args.input_dir):
        print(f"[ERROR] 输入文件夹不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 调用核心函数生成词典和词汇表
    dict_path, vocab_path = generate_userdict_and_vocab(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_freq=args.min_freq,
        min_pmi=args.min_pmi,
        max_len=args.max_len,
        do_filter=args.filter,
        ac_mask=args.ac_mask,
        auto_threshold=args.auto_threshold
    )
    
    # 输出结果
    if dict_path and vocab_path:
        print(f'[INFO] userdict已保存: {dict_path}')
        print(f'[INFO] vocab已保存: {vocab_path}')
    else:
        print('[WARN] 未生成任何文件')

def generate_hybrid_whitelist(texts, output_path, wh40k_vocab_path=None, config=None):
    """生成混合白名单：自动发现 + 专业术语"""
    
    # 1. 自动发现短语
    auto_phrases = extract_phrases(texts)
    
    # 2. 加载专业术语（如果提供）
    wh40k_phrases = []
    if wh40k_vocab_path:
        generator = WH40KWhitelistGenerator(wh40k_vocab_path)
        wh40k_phrases = generator.extract_wh40k_phrases()
    
    # 3. 合并去重
    all_phrases = list(set(auto_phrases + wh40k_phrases))
    
    # 4. 生成白名单（传递配置参数）
    generate_whitelist(all_phrases, output_path, config=config)
    
    return output_path


if __name__ == '__main__':
    main() 