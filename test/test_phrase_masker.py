import unittest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataupload.gen_userdict import PhraseMasker

class TestPhraseMasker(unittest.TestCase):
    def test_normal(self):
        phrases = ["人工智能", "深度学习"]
        masker = PhraseMasker(phrases)
        text = "人工智能和深度学习是热门领域。"
        masked, intervals = masker.mask_phrases(text)
        self.assertIn("__PHRASE_0__", masked)
        self.assertIn("__PHRASE_1__", masked)
        restored = masker.restore_phrases(masked, intervals)
        self.assertEqual(restored, text)

    def test_overlap(self):
        phrases = ["人工", "人工智能"]
        masker = PhraseMasker(phrases)
        text = "人工智能是人工的产物。"
        masked, intervals = masker.mask_phrases(text)
        # 应优先匹配长短语
        self.assertIn("__PHRASE_1__", masked)
        restored = masker.restore_phrases(masked, intervals)
        self.assertEqual(restored, text)

    def test_empty(self):
        masker = PhraseMasker([])
        text = ""
        masked, intervals = masker.mask_phrases(text)
        self.assertEqual(masked, "")
        self.assertEqual(intervals, [])
        restored = masker.restore_phrases(masked, intervals)
        self.assertEqual(restored, "")

    def test_special_chars(self):
        phrases = ["A*B", "C#D"]
        masker = PhraseMasker(phrases)
        text = "A*B和C#D是特殊短语。"
        masked, intervals = masker.mask_phrases(text)
        self.assertIn("__PHRASE_0__", masked)
        self.assertIn("__PHRASE_1__", masked)
        restored = masker.restore_phrases(masked, intervals)
        self.assertEqual(restored, text)

    def test_long_text(self):
        phrases = ["深度学习"]
        masker = PhraseMasker(phrases)
        text = "深度学习" * 1000
        masked, intervals = masker.mask_phrases(text)
        self.assertTrue(masked.count("__PHRASE_0__") >= 1000)
        restored = masker.restore_phrases(masked, intervals)
        self.assertEqual(restored, text)

    def test_no_match(self):
        phrases = ["不存在"]
        masker = PhraseMasker(phrases)
        text = "人工智能和深度学习。"
        masked, intervals = masker.mask_phrases(text)
        self.assertEqual(masked, text)
        self.assertEqual(intervals, [])
        restored = masker.restore_phrases(masked, intervals)
        self.assertEqual(restored, text)

if __name__ == '__main__':
    unittest.main() 