import unittest
from ..base_test import TestBase
from aAggregation.dedup_agg import DedupAggregation

class TestDedupAggregation(TestBase):
    def setUp(self):
        self.aggregator = DedupAggregation()

    def test_empty_input(self):
        search_results = []
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, [])

    def test_all_empty_text(self):
        search_results = [[{'text': ''}], [{'text': ''}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, [])

    def test_all_none_text(self):
        search_results = [[{'text': None}], [{'text': None}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, [])

    def test_all_unique(self):
        search_results = [[{'text': 'A'}], [{'text': 'B'}], [{'text': 'C'}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, ['A', 'B', 'C'])

    def test_all_duplicate(self):
        search_results = [[{'text': 'A'}], [{'text': 'A'}], [{'text': 'A'}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, ['A'])

    def test_mixed_empty_and_valid(self):
        search_results = [[{'text': ''}], [{'text': 'A'}], [{'text': ''}], [{'text': 'B'}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, ['A', 'B'])

    def test_missing_text_field(self):
        search_results = [[{'doc_id': 1}], [{'text': 'A'}], [{}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, ['A'])

    def test_order_preserved(self):
        search_results = [[{'text': 'A'}], [{'text': 'B'}], [{'text': 'A'}], [{'text': 'C'}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, ['A', 'B', 'C'])

    def test_large_input(self):
        search_results = [[{'text': str(i)}] for i in range(1000)] + [[{'text': 'A'}]]*10
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers[:3], ['0', '1', '2'])
        self.assertEqual(answers[-1], 'A')
        self.assertEqual(len(set(answers)), 1001)

    def test_special_characters(self):
        search_results = [[{'text': 'A!@#'}], [{'text': 'B\nC'}], [{'text': 'A!@#'}], [{'text': 'D  E'}]]
        answers = self.aggregator.extract_answer(search_results)
        self.assertEqual(answers, ['A!@#', 'B\nC', 'D  E'])

    def test_aggregate_only_answers(self):
        # 聚合只返回去重后的answers，不测LLM
        search_results = [[{'text': 'A'}], [{'text': 'B'}], [{'text': 'A'}]]
        result = self.aggregator.aggregate(search_results, query='Q', params=None)
        self.assertIn('answers', result)
        self.assertEqual(result['answers'], ['A', 'B'])
        self.assertEqual(result['count'], 2)

if __name__ == '__main__':
    unittest.main() 