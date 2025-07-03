import os
import sys
from typing import List, Dict, Any
from pinecone import Pinecone
from openai import OpenAI
# 动态添加项目根目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, RERANK_MODEL, EMBADDING_MODEL
import logging

logger = logging.getLogger(__name__)

class SubQueryRetriever:
    """
    子查询检索器：负责实体识别+filter+向量召回
    """
    def __init__(self, pinecone_api_key: str = None, index_name: str = None, openai_api_key: str = None):
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.index_name = index_name or os.getenv('PINECONE_INDEX')
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.client = OpenAI(api_key=self.openai_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key, environment=os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter'))
        self.index = self.pc.Index(self.index_name)

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=EMBADDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def retrieve_for_queries(self, queries: List[str], entities: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """
        对一组子查询，分别用实体filter+向量召回相关切片
        Args:
            queries: 子查询列表
            entities: 每个子查询对应的实体（如单位、技能等）
            top_k: 每个子查询召回数量
        Returns:
            List[List[Dict]]: 每个子查询召回的切片列表
        """
        results = []
        for query, entity in zip(queries, entities):
            try:
                query_embedding = self.get_embedding(query)
                filter_dict = {"$or": [
                    {"h1": entity},
                    {"h2": entity},
                    {"h3": entity},
                    {"h4": entity},
                    {"h5": entity},
                    {"h6": entity}
                ]}
                res = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict,
                    hybrid_search=True,
                    alpha=0.3,
                    rerank_config={
                        "model": RERANK_MODEL,
                        "top_k": top_k
                    }
                )
                chunk_list = []
                for match in res.matches:
                    text = match.metadata.get('text', '')
                    if not text or len(text.strip()) < 10:
                        continue
                    chunk_list.append({
                        'text': text,
                        'score': match.score,
                        'metadata': match.metadata
                    })
                results.append(chunk_list)
            except Exception as e:
                logger.error(f"子查询检索出错: {str(e)}")
                results.append([])
        return results

# DEMO用法
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="子查询检索器demo")
    parser.add_argument('--queries', nargs='+', required=True, help='子查询列表')
    parser.add_argument('--entities', nargs='+', required=True, help='实体列表（与子查询一一对应）')
    parser.add_argument('--topk', type=int, default=10, help='每个子查询返回数量')
    args = parser.parse_args()
    retriever = SubQueryRetriever()
    results = retriever.retrieve_for_queries(args.queries, args.entities, top_k=args.topk)
    for i, chunk_list in enumerate(results):
        print(f"\n=== 子查询{i+1}: {args.queries[i]} (实体: {args.entities[i]}) ===")
        for j, chunk in enumerate(chunk_list):
            print(f"--- 结果{j+1} ---\n分数: {chunk['score']}\n内容: {chunk['text'][:100]}...\n元数据: {chunk['metadata']}\n") 