from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AggregationInterface(ABC):
    @abstractmethod
    def aggregate(self, results: List[Dict[str, Any]], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        聚合检索结果，输出最终答案。
        :param results: 检索结果列表，每个元素为 dict。
        :param params: 可选参数字典。
        :return: 聚合后的答案 dict。
        """
        pass 
    @abstractmethod
    def extract_answer(self, search_results) -> list:
        """
        从原始search_results中抽取并去重答案，返回答案列表。
        """
        pass