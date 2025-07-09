import os
from typing import Dict, List, Any
from pinecone import Pinecone

class PineconeManager:
    """
    Pinecone 操作管理类
    支持：
    1. 删除指定Index下指定filter（如faction）的所有chunk
    2. 删除指定Index下所有数据
    3. 搜索指定Index下指定filter的所有数据
    4. 统计索引总记录数
    """
    def __init__(self, index_name: str = None, api_key: str = None, environment: str = None):
        self.index_name = index_name or os.getenv('PINECONE_INDEX')
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        self.environment = environment or os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
        if not self.index_name or not self.api_key:
            raise ValueError("PINECONE_INDEX 和 PINECONE_API_KEY 必须配置在环境变量中")
        # 新版SDK初始化
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)

    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> dict:
        """
        删除指定filter下的所有chunk
        :param filter_dict: 例如 {"faction": "aeldari"}
        :return: 删除操作返回信息
        """
        return self.index.delete(filter=filter_dict)

    def delete_all(self) -> dict:
        """
        删除该index下所有数据
        :return: 删除操作返回信息
        """
        return self.index.delete(delete_all=True)

    def get_index_stats(self) -> dict:
        """
        获取索引统计信息
        :return: 索引统计信息
        """
        return self.index.describe_index_stats()

    def count_all(self) -> int:
        """
        统计索引总记录数
        :return: 记录总数
        """
        stats = self.get_index_stats()
        total_count = stats.get('total_vector_count', 0)
        return total_count

    def query_by_filter(self, filter_dict: Dict[str, Any], limit: int = 1000) -> List[dict]:
        """
        使用Pinecone的query接口，通过metadata filter批量检索相关记录
        :param filter_dict: metadata过滤条件
        :param limit: 最大返回数量
        :return: 结果列表
        """
        results = []
        try:
            # 这里的vector内容无关紧要，只要用filter即可，长度需与index一致
            # 这里假设1536维，如有不同请根据实际模型调整
            query_result = self.index.query(
                vector=[0.0]*1536,
                filter=filter_dict,
                top_k=limit,
                include_metadata=True
            )
            for match in getattr(query_result, 'matches', []):
                results.append({
                    'id': match.id,
                    'metadata': match.metadata
                })
        except Exception as e:
            print(f"查询时出错: {e}")
        return results

    def query_by_section_heading_and_faction(self, section_heading: str, faction: str, limit: int = 10) -> list:
        filter_dict = {
            "section_heading": section_heading,
            "faction": faction
        }
        return self.query_by_filter(filter_dict, limit=limit)

    def query_by_h3(self, h3_value: str, limit: int = 10) -> list:
        filter_dict = {
            "h3": h3_value
        }
        return self.query_by_filter(filter_dict, limit=limit)

if __name__ == "__main__":
    # 简单命令行测试
    import argparse
    parser = argparse.ArgumentParser(description="Pinecone管理工具")
    parser.add_argument('--delete-all', action='store_true', help='删除所有数据')
    parser.add_argument('--delete-faction', type=str, help='删除指定faction的所有数据')
    parser.add_argument('--query-faction', type=str, help='查询指定faction的所有数据')
    parser.add_argument('--limit', type=int, default=1000, help='查询最大数量')
    parser.add_argument('--count-all', action='store_true', help='统计索引总记录数')
    parser.add_argument('--query-section-faction', nargs=2, metavar=('SECTION_HEADING', 'FACTION'), help='通过section_heading和faction联合过滤查询')
    parser.add_argument('--query-h3', type=str, help='通过h3字段过滤查询')
    args = parser.parse_args()
    mgr = PineconeManager()
    
    if args.count_all:
        count = mgr.count_all()
        print(f"索引 {mgr.index_name} 总记录数: {count}")
    
    if args.delete_all:
        print(mgr.delete_all())
    
    if args.delete_faction:
        print(mgr.delete_by_filter({"faction": args.delete_faction}))
    
    if args.query_faction:
        results = mgr.query_by_filter({"faction": args.query_faction}, limit=args.limit)
        print(f"共{len(results)}条：")
        for r in results:
            print(r)
    
    if args.query_section_faction:
        section_heading, faction = args.query_section_faction
        results = mgr.query_by_section_heading_and_faction(section_heading, faction, limit=args.limit)
        print(f"共{len(results)}条：")
        for r in results:
            print(r)
    
    if args.query_h3:
        results = mgr.query_by_h3(args.query_h3, limit=args.limit)
        print(f"共{len(results)}条：")
        for r in results:
            print(r) 