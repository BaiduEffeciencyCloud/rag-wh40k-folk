import unittest
import sys, os, shutil, json
import tempfile
import subprocess
from datetime import datetime
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataupload.gen_userdict import extract_phrases, collect_texts_from_folder, PhraseMasker

class TestGenUserdictFunctions(unittest.TestCase):
    """测试gen_userdict的核心功能函数"""
    
    def setUp(self):
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test', 'testdata')
        self.empty_dir = os.path.join(self.base_dir, 'empty')
        os.makedirs(self.empty_dir, exist_ok=True)
        self.sub_dir = os.path.join(self.base_dir, 'subdir')
        os.makedirs(self.sub_dir, exist_ok=True)
        # 创建极端小文件
        with open(os.path.join(self.base_dir, 'tiny.txt'), 'w', encoding='utf-8') as f:
            f.write('a')
        # 创建极端大文件
        with open(os.path.join(self.base_dir, 'huge.txt'), 'w', encoding='utf-8') as f:
            f.write('b' * 10000)
        # 创建特殊字符文件名
        with open(os.path.join(self.base_dir, '特殊@#￥.md'), 'w', encoding='utf-8') as f:
            f.write('特殊内容')
        # 创建子目录文件
        with open(os.path.join(self.sub_dir, 'sub.txt'), 'w', encoding='utf-8') as f:
            f.write('子目录内容')

    def test_extract_phrases_normal(self):
        """测试正常短语提取"""
        texts = ["人工智能正在改变世界。", "深度学习是人工智能的重要分支。"]
        phrases = extract_phrases(texts, min_freq=1, min_pmi=0, max_len=10, do_filter=True)
        self.assertIsInstance(phrases, list)
        self.assertTrue(any("人工智能" in p for p in phrases))

    def test_extract_phrases_empty(self):
        """测试空文本短语提取"""
        phrases = extract_phrases([], min_freq=1, min_pmi=0, max_len=10, do_filter=True)
        self.assertEqual(phrases, [])

    def test_extract_phrases_extreme(self):
        """测试极端情况短语提取"""
        # 极端长文本
        long_text = ["深度学习" * 1000]
        phrases = extract_phrases(long_text, min_freq=1, min_pmi=0, max_len=20, do_filter=True)
        self.assertIsInstance(phrases, list)
        # 极端短文本
        short_text = ["的"]
        phrases2 = extract_phrases(short_text, min_freq=1, min_pmi=0, max_len=10, do_filter=True)
        self.assertIsInstance(phrases2, list)

    def test_extract_phrases_invalid(self):
        """测试非法输入短语提取"""
        # 非法输入类型
        with self.assertRaises(Exception):
            extract_phrases(None)
        with self.assertRaises(Exception):
            extract_phrases(123)

    def test_collect_texts_normal(self):
        """测试正常文本收集"""
        texts = collect_texts_from_folder(self.base_dir, exts=['.txt', '.md'])
        self.assertIsInstance(texts, list)
        self.assertTrue(any('a' == t for t in texts))
        self.assertTrue(any('b' * 10000 == t for t in texts))
        self.assertTrue(any('特殊内容' == t for t in texts))
        self.assertTrue(any('子目录内容' == t for t in texts))
        self.assertGreaterEqual(len(texts), 4)

    def test_collect_texts_empty_dir(self):
        """测试空目录文本收集"""
        texts = collect_texts_from_folder(self.empty_dir, exts=['.txt'])
        self.assertEqual(texts, [])

    def test_collect_texts_nonexistent_dir(self):
        """测试不存在目录文本收集"""
        texts = collect_texts_from_folder('/not/exist/path', exts=['.txt'])
        self.assertEqual(texts, [])

    def test_collect_texts_no_permission(self):
        """测试无权限目录文本收集"""
        # 仅模拟，不实际更改权限，保证不会抛异常
        try:
            texts = collect_texts_from_folder('/root', exts=['.txt'])
        except Exception as e:
            self.fail(f"collect_texts_from_folder raised Exception: {e}")

    def test_collect_texts_mixed_file_types(self):
        """测试混合文件类型文本收集"""
        texts = collect_texts_from_folder(self.base_dir, exts=['.txt', '.md', '.json'])
        self.assertIsInstance(texts, list)
        self.assertGreaterEqual(len(texts), 4)

    def test_filter_number_and_space(self):
        """测试数字和空格过滤功能"""
        from dataupload.gen_userdict import filter_number_and_space
        
        # 应该被过滤的情况
        self.assertFalse(filter_number_and_space(""))  # 空字符串
        self.assertFalse(filter_number_and_space("   "))  # 全空格
        self.assertFalse(filter_number_and_space("123"))  # 纯数字
        self.assertFalse(filter_number_and_space("456789"))  # 纯数字
        self.assertFalse(filter_number_and_space(" 123 "))  # 带空格的纯数字
        
        # 应该保留的情况
        self.assertTrue(filter_number_and_space("D3"))  # 骰子词汇
        self.assertTrue(filter_number_and_space("D6"))  # 骰子词汇
        self.assertTrue(filter_number_and_space("2D6"))  # 骰子词汇
        self.assertTrue(filter_number_and_space("3D4"))  # 骰子词汇
        self.assertTrue(filter_number_and_space("D20"))  # 骰子词汇
        self.assertTrue(filter_number_and_space("1D100"))  # 骰子词汇
        self.assertTrue(filter_number_and_space("test"))  # 普通词汇
        self.assertTrue(filter_number_and_space("test123"))  # 包含数字的词汇
        self.assertTrue(filter_number_and_space("123test"))  # 包含数字的词汇
        self.assertTrue(filter_number_and_space("test123test"))  # 包含数字的词汇
        self.assertTrue(filter_number_and_space("D"))  # 单个字母
        self.assertFalse(filter_number_and_space("2"))  # 单个数字应该被过滤
        self.assertTrue(filter_number_and_space("A"))  # 单个字母
        # 特殊骰子表达（D[数字]+数字）应该保留
        self.assertTrue(filter_number_and_space("D6+2"))
        self.assertTrue(filter_number_and_space("D6+3"))
        self.assertTrue(filter_number_and_space("2D6+1"))
        self.assertTrue(filter_number_and_space("D3+5"))
        self.assertTrue(filter_number_and_space("D20+10"))
        self.assertTrue(filter_number_and_space("3D4+2"))

    def test_filter_number_and_space_with_extract_phrases(self):
        """测试在短语提取中数字过滤的效果"""
        texts = [
            "玩家掷骰子D6获得6点，然后掷2D6获得12点。",
            "这个技能需要D20检定，基础难度是15。",
            "武器伤害是3D4，最大伤害是12。",
            "生命值是100，护甲值是5。"
        ]
        
        # 使用当前过滤逻辑
        phrases = extract_phrases(texts, min_freq=1, min_pmi=0, max_len=10, do_filter=True)
        
        # 验证骰子相关词汇被保留
        dice_phrases = [p for p in phrases if 'D' in p and any(c.isdigit() for c in p)]
        self.assertGreater(len(dice_phrases), 0, "应该保留骰子相关词汇")
        
        # 验证纯数字被过滤
        pure_numbers = [p for p in phrases if p.isdigit()]
        self.assertEqual(len(pure_numbers), 0, "应该过滤掉纯数字")
        
        # 打印结果用于调试
        print(f"提取的短语: {phrases}")
        print(f"骰子相关词汇: {dice_phrases}")

    def tearDown(self):
        """清理测试文件"""
        # 清理极端小/大/特殊文件
        for fname in ['tiny.txt', 'huge.txt', '特殊@#￥.md']:
            fpath = os.path.join(self.base_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        # 清理子目录文件
        subfile = os.path.join(self.sub_dir, 'sub.txt')
        if os.path.exists(subfile):
            os.remove(subfile)
        # 不删除目录，避免影响其他测试


class TestGenUserdictAutoThreshold(unittest.TestCase):
    """测试自动阈值功能"""
    
    def setUp(self):
        # 直接用真实测试数据目录
        self.input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
        self.output_dir = tempfile.mkdtemp()

    def test_auto_threshold_vs_fixed(self):
        """测试自动阈值与固定阈值的差异"""
        # 先跑固定阈值
        cmd1 = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd1, shell=True, check=True)
        files1 = os.listdir(self.output_dir)
        meta1 = [f for f in files1 if f.endswith('.meta.json')][0]
        with open(os.path.join(self.output_dir, meta1), 'r', encoding='utf-8') as f:
            meta_fixed = json.load(f)
        weights_fixed = set(v['weight'] for v in meta_fixed.values())
        # 清空产物
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))
        # 再跑自动阈值
        cmd2 = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter --auto-threshold"
        subprocess.run(cmd2, shell=True, check=True)
        files2 = os.listdir(self.output_dir)
        meta2 = [f for f in files2 if f.endswith('.meta.json')][0]
        with open(os.path.join(self.output_dir, meta2), 'r', encoding='utf-8') as f:
            meta_auto = json.load(f)
        weights_auto = set(v['weight'] for v in meta_auto.values())
        self.assertNotEqual(weights_fixed, weights_auto)
        self.assertTrue(len(meta_auto) > 0)
        self.assertIn('weight', list(meta_auto.values())[0])
        self.assertIn('df', list(meta_auto.values())[0])

    def test_no_filter_phrase_extraction(self):
        """测试不使用 --filter 参数时短语提取是否正常工作"""
        # 不使用 --filter 参数
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10"
        subprocess.run(cmd, shell=True, check=True)
        
        files = os.listdir(self.output_dir)
        meta_file = [f for f in files if f.endswith('.meta.json')][0]
        dict_file = [f for f in files if f.endswith('.txt') and not f.endswith('.meta.json')][0]
        
        # 检查 meta.json 是否包含内容
        with open(os.path.join(self.output_dir, meta_file), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 检查 dict.txt 是否包含内容
        with open(os.path.join(self.output_dir, dict_file), 'r', encoding='utf-8') as f:
            dict_content = f.read()
        
        # 断言：不使用 --filter 时也应该提取到短语
        self.assertTrue(len(meta) > 0, "不使用 --filter 时 meta.json 应该包含短语")
        self.assertTrue(len(dict_content.strip()) > 0, "不使用 --filter 时 dict.txt 应该包含内容")
        
        # 检查 meta 结构是否正确
        for phrase, info in meta.items():
            self.assertIn('weight', info)
            self.assertIn('df', info)
            self.assertIn('pmi', info)

    def test_filter_vs_no_filter_difference(self):
        """测试使用 --filter 和不使用 --filter 的差异"""
        # 清空产物
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))
        
        # 不使用 --filter
        cmd1 = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10"
        subprocess.run(cmd1, shell=True, check=True)
        
        files1 = os.listdir(self.output_dir)
        meta1 = [f for f in files1 if f.endswith('.meta.json')][0]
        with open(os.path.join(self.output_dir, meta1), 'r', encoding='utf-8') as f:
            meta_no_filter = json.load(f)
        
        # 清空产物
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))
        
        # 使用 --filter
        cmd2 = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd2, shell=True, check=True)
        
        files2 = os.listdir(self.output_dir)
        meta2 = [f for f in files2 if f.endswith('.meta.json')][0]
        with open(os.path.join(self.output_dir, meta2), 'r', encoding='utf-8') as f:
            meta_filter = json.load(f)
        
        # 断言：两种方式都应该提取到短语，但可能有不同的结果
        self.assertTrue(len(meta_no_filter) > 0, "不使用 --filter 时应该提取到短语")
        self.assertTrue(len(meta_filter) > 0, "使用 --filter 时应该提取到短语")
        
        # 由于过滤逻辑不同，结果可能不同，但不应该都是空的
        print(f"不使用 --filter 提取到 {len(meta_no_filter)} 个短语")
        print(f"使用 --filter 提取到 {len(meta_filter)} 个短语")

    def tearDown(self):
        shutil.rmtree(self.output_dir)


class TestGenUserdictACMask(unittest.TestCase):
    """测试AC掩码功能"""
    
    def setUp(self):
        self.input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
        self.output_dir = tempfile.mkdtemp()

    def test_ac_mask_consistency(self):
        """测试AC掩码功能的一致性"""
        # 先跑ac_mask=False
        cmd1 = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd1, shell=True, check=True)
        
        # 再跑ac_mask=True
        cmd2 = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter --ac-mask"
        subprocess.run(cmd2, shell=True, check=True)
        
        # 检查生成的文件
        files = os.listdir(self.output_dir)
        dict_files = [f for f in files if f.endswith('.txt') and 'dict' in f]
        vocab_files = [f for f in files if f.endswith('.json') and 'vocab' in f]
        
        self.assertGreater(len(dict_files), 0, "应该生成词典文件")
        self.assertGreater(len(vocab_files), 0, "应该生成词汇文件")

    def test_protective_dict_loading(self):
        """测试保护性词典是否被正确加载到jieba中"""
        # 创建包含特殊词汇的测试文本
        test_text = "阿苏焉尼是一个特殊的词汇，不应该被拆分"
        
        # 生成保护性词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter --ac-mask"
        subprocess.run(cmd, shell=True, check=True)
        
        # 找到生成的词典文件
        files = os.listdir(self.output_dir)
        dict_files = [f for f in files if f.endswith('.txt') and 'dict' in f]
        self.assertGreater(len(dict_files), 0, "应该生成词典文件")
        dict_path = os.path.join(self.output_dir, dict_files[0])
        
        # 检查词典文件是否包含目标词汇
        with open(dict_path, 'r', encoding='utf-8') as f:
            dict_content = f.read()
        self.assertIn('阿苏焉尼', dict_content, "词典文件应该包含'阿苏焉尼'")
        
        # 测试jieba分词是否使用了保护性词典
        import jieba
        # 重置jieba，确保测试环境干净
        jieba.initialize()
        
        # 加载保护性词典
        jieba.load_userdict(dict_path)
        
        # 测试分词结果
        tokens = list(jieba.cut(test_text))
        print(f"[DEBUG] 分词结果: {tokens}")
        
        # 验证"阿苏焉尼"是否作为一个整体被识别
        # 方法1：检查是否在分词结果中
        self.assertIn('阿苏焉尼', tokens, "jieba应该将'阿苏焉尼'识别为一个整体")
        
        # 方法2：检查是否没有被拆分
        for token in tokens:
            if token in ['阿苏', '焉尼']:
                self.fail(f"保护性词典失效：'阿苏焉尼'被错误拆分为 {token}")
        
        # 方法3：检查分词结果中不包含部分词汇
        self.assertNotIn('阿苏', tokens, "保护性词典应该防止'阿苏焉尼'被拆分为'阿苏'")
        self.assertNotIn('焉尼', tokens, "保护性词典应该防止'阿苏焉尼'被拆分为'焉尼'")

    def test_bm25_manager_dict_loading(self):
        """测试BM25Manager是否正确加载了保护性词典"""
        # 生成保护性词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter --ac-mask"
        subprocess.run(cmd, shell=True, check=True)
        
        # 找到生成的词典文件
        files = os.listdir(self.output_dir)
        dict_files = [f for f in files if f.endswith('.txt') and 'dict' in f]
        self.assertGreater(len(dict_files), 0, "应该生成词典文件")
        dict_path = os.path.join(self.output_dir, dict_files[0])
        
        # 测试BM25Manager的分词
        from dataupload.bm25_manager import BM25Manager
        
        # 创建包含特殊词汇的测试文本（增加频率）
        test_texts = [
            "阿苏焉尼是一个特殊的词汇",
            "阿苏焉尼不应该被拆分", 
            "这个阿苏焉尼词汇很重要",
            "阿苏焉尼在文本中出现了多次",
            "阿苏焉尼是我们要保护的词汇"
        ]
        
        # 初始化BM25Manager（设置min_freq=1，确保低频词汇也能进入词汇表）
        bm25 = BM25Manager(user_dict_path=dict_path, min_freq=1)
        
        # 测试分词
        tokens = bm25.tokenize_chinese(test_texts[0])
        print(f"[DEBUG] BM25Manager分词结果: {tokens}")
        
        # 验证保护性词典是否生效
        self.assertIn('阿苏焉尼', tokens, "BM25Manager应该将'阿苏焉尼'识别为一个整体")
        self.assertNotIn('阿苏', tokens, "BM25Manager应该防止'阿苏焉尼'被拆分为'阿苏'")
        self.assertNotIn('焉尼', tokens, "BM25Manager应该防止'阿苏焉尼'被拆分为'焉尼'")
        
        # 测试fit后的词汇表
        bm25.fit(test_texts)
        vocab = bm25.vocabulary
        print(f"[DEBUG] BM25Manager词汇表大小: {len(vocab)}")
        print(f"[DEBUG] BM25Manager词汇表前10个: {list(vocab.keys())[:10]}")
        
        # 验证词汇表中包含完整词汇
        self.assertIn('阿苏焉尼', vocab, "BM25Manager的词汇表应该包含完整的'阿苏焉尼'")
        self.assertNotIn('阿苏', vocab, "BM25Manager的词汇表不应该包含被拆分的'阿苏'")
        self.assertNotIn('焉尼', vocab, "BM25Manager的词汇表不应该包含被拆分的'焉尼'")
        
        # 额外验证：检查词汇表中的"阿苏焉尼"的索引
        asu_yan_ni_index = vocab.get('阿苏焉尼')
        print(f"[DEBUG] '阿苏焉尼'在词汇表中的索引: {asu_yan_ni_index}")
        self.assertIsNotNone(asu_yan_ni_index, "'阿苏焉尼'应该在词汇表中有索引")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)


class TestGenUserdictWithFrequency(unittest.TestCase):
    """测试生成带词频信息的词典功能"""
    
    def setUp(self):
        self.input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
        self.output_dir = tempfile.mkdtemp()

    def test_frequency_vocab_generation(self):
        """测试生成带词频信息的词典"""
        # 生成词典（应该同时生成传统格式和带词频格式）
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd, shell=True, check=True)
        
        # 检查生成的文件
        files = os.listdir(self.output_dir)
        freq_vocab_files = [f for f in files if f.startswith('bm25_vocab_freq_') and f.endswith('.json')]
        normal_vocab_files = [f for f in files if f.startswith('bm25_vocab_') and f.endswith('.json') and not f.startswith('bm25_vocab_freq_')]
        
        # 应该同时生成两种格式的词典
        self.assertGreater(len(freq_vocab_files), 0, "应该生成带词频的词典文件")
        self.assertGreater(len(normal_vocab_files), 0, "应该生成传统格式的词典文件")
        
        # 检查带词频词典文件格式
        vocab_file = os.path.join(self.output_dir, freq_vocab_files[0])
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 验证格式：应该是 {word: {"weight": float, "freq": int, "df": int}}
        for word, info in vocab_data.items():
            self.assertIsInstance(word, str, "词汇应该是字符串")
            self.assertIsInstance(info, dict, "词汇信息应该是字典")
            self.assertIn('weight', info, "应该包含weight字段")
            self.assertIn('freq', info, "应该包含freq字段")
            self.assertIn('df', info, "应该包含df字段")
            self.assertIsInstance(info['weight'], (int, float), "weight应该是数值")
            self.assertIsInstance(info['freq'], int, "freq应该是整数")
            self.assertIsInstance(info['df'], int, "df应该是整数")
            self.assertGreaterEqual(info['freq'], 0, "freq应该非负")
            self.assertGreaterEqual(info['df'], 0, "df应该非负")

    def test_frequency_vocab_backward_compatibility(self):
        """测试带词频词典的向后兼容性"""
        # 生成词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd, shell=True, check=True)
        
        # 找到生成的词典文件
        files = os.listdir(self.output_dir)
        freq_vocab_files = [f for f in files if f.startswith('bm25_vocab_freq_') and f.endswith('.json')]
        vocab_file = os.path.join(self.output_dir, freq_vocab_files[0])
        
        # 测试BM25Manager是否能正确加载新格式
        from dataupload.bm25_manager import BM25Manager
        
        # 应该能正常加载，不会抛出异常
        try:
            bm25 = BM25Manager(user_dict_path=vocab_file, min_freq=1)
            # 测试分词功能
            test_text = "人工智能技术"
            tokens = bm25.tokenize_chinese(test_text)
            self.assertIsInstance(tokens, list, "分词应该返回列表")
        except Exception as e:
            self.fail(f"BM25Manager应该能兼容新格式，但抛出了异常: {e}")

    def test_frequency_vocab_content_validation(self):
        """测试带词频词典内容的正确性"""
        # 生成词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd, shell=True, check=True)
        
        # 找到生成的词典文件
        files = os.listdir(self.output_dir)
        freq_vocab_files = [f for f in files if f.startswith('bm25_vocab_freq_') and f.endswith('.json')]
        vocab_file = os.path.join(self.output_dir, freq_vocab_files[0])
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # 验证词频统计的正确性
        # 检查词汇的freq和df是否合理
        for word, info in vocab_data.items():
            # freq应该大于等于df（总词频 >= 文档频率）
            self.assertGreaterEqual(info['freq'], info['df'], f"词汇'{word}'的总词频应该大于等于文档频率")
            # df应该大于0（至少在一个文档中出现）
            self.assertGreater(info['df'], 0, f"词汇'{word}'应该在至少一个文档中出现")

    def test_frequency_vocab_naming_convention(self):
        """测试带词频词典的命名规范"""
        # 生成词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd, shell=True, check=True)
        
        # 检查文件命名
        files = os.listdir(self.output_dir)
        freq_vocab_files = [f for f in files if f.startswith('bm25_vocab_freq_') and f.endswith('.json')]
        
        self.assertGreater(len(freq_vocab_files), 0, "应该生成带词频的词典文件")
        
        # 验证命名格式：bm25_vocab_freq_YYMMDDHHMM.json
        vocab_file = freq_vocab_files[0]
        self.assertTrue(vocab_file.startswith('bm25_vocab_freq_'), "文件名应该以'bm25_vocab_freq_'开头")
        self.assertTrue(vocab_file.endswith('.json'), "文件名应该以'.json'结尾")
        
        # 验证时间戳格式（YYMMDDHHMM）
        timestamp_part = vocab_file.replace('bm25_vocab_freq_', '').replace('.json', '')
        self.assertEqual(len(timestamp_part), 10, "时间戳应该是10位数字")
        self.assertTrue(timestamp_part.isdigit(), "时间戳应该全是数字")

    def test_both_vocab_formats_generated(self):
        """测试同时生成两种格式的词典"""
        # 生成词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd, shell=True, check=True)
        
        # 检查生成的文件
        files = os.listdir(self.output_dir)
        freq_vocab_files = [f for f in files if f.startswith('bm25_vocab_freq_') and f.endswith('.json')]
        normal_vocab_files = [f for f in files if f.startswith('bm25_vocab_') and f.endswith('.json') and not f.startswith('bm25_vocab_freq_')]
        
        # 应该同时生成两种格式
        self.assertGreater(len(freq_vocab_files), 0, "应该生成带词频的词典文件")
        self.assertGreater(len(normal_vocab_files), 0, "应该生成传统格式的词典文件")
        
        # 检查两种格式的词汇数量应该一致
        freq_vocab_file = os.path.join(self.output_dir, freq_vocab_files[0])
        normal_vocab_file = os.path.join(self.output_dir, normal_vocab_files[0])
        
        with open(freq_vocab_file, 'r', encoding='utf-8') as f:
            freq_data = json.load(f)
        with open(normal_vocab_file, 'r', encoding='utf-8') as f:
            normal_data = json.load(f)
        
        self.assertEqual(len(freq_data), len(normal_data), "两种格式的词典应该包含相同数量的词汇")
        
        # 检查词汇列表应该一致
        freq_words = set(freq_data.keys())
        normal_words = set(normal_data.keys())
        self.assertEqual(freq_words, normal_words, "两种格式的词典应该包含相同的词汇")

    def test_frequency_vocab_weight_consistency(self):
        """测试带词频词典中权重值的一致性"""
        # 生成词典
        cmd = f"python3 dataupload/gen_userdict.py --input-dir {self.input_dir} --output-dir {self.output_dir} --min-freq 1 --min-pmi 0 --max-len 10 --filter"
        subprocess.run(cmd, shell=True, check=True)
        
        # 找到生成的词典文件
        files = os.listdir(self.output_dir)
        freq_vocab_files = [f for f in files if f.startswith('bm25_vocab_freq_') and f.endswith('.json')]
        normal_vocab_files = [f for f in files if f.startswith('bm25_vocab_') and f.endswith('.json') and not f.startswith('bm25_vocab_freq_')]
        
        freq_vocab_file = os.path.join(self.output_dir, freq_vocab_files[0])
        normal_vocab_file = os.path.join(self.output_dir, normal_vocab_files[0])
        
        with open(freq_vocab_file, 'r', encoding='utf-8') as f:
            freq_data = json.load(f)
        with open(normal_vocab_file, 'r', encoding='utf-8') as f:
            normal_data = json.load(f)
        
        # 检查权重值应该一致
        for word in freq_data:
            freq_weight = freq_data[word]['weight']
            normal_weight = normal_data[word]
            self.assertEqual(freq_weight, normal_weight, f"词汇'{word}'的权重值应该一致")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main() 