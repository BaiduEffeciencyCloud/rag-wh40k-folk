import os
from typing import Dict, List, Any
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

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
        self.index_name = index_name or PINECONE_INDEX
        self.api_key = api_key or PINECONE_API_KEY
        self.environment = environment or 'gcp-starter'
        if not self.index_name or not self.api_key:
            raise ValueError("PINECONE_INDEX 和 PINECONE_API_KEY 必须配置在config.py中")
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
            # 这里假设1024维，如有不同请根据实际模型调整
            query_result = self.index.query(
                vector=[0.0]*1024,
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

    def check_sparse_vectors(self, limit: int = 10) -> dict:
        """
        检查索引中的记录是否包含sparse向量
        :param limit: 检查的记录数量
        :return: 包含统计信息的字典
        """
        try:
            # 使用一个简单的查询来获取记录
            query_result = self.index.query(
                vector=[0.0]*1024,  # 假设1024维
                top_k=limit,
                include_metadata=True,
                include_values=True  # 包含向量值
            )
            
            total_records = len(query_result.matches)
            sparse_count = 0
            dense_only_count = 0
            sample_records = []
            
            for match in query_result.matches:
                record_info = {
                    'id': match.id,
                    'has_sparse': hasattr(match, 'sparse_values') and match.sparse_values is not None,
                    'metadata': match.metadata
                }
                
                if record_info['has_sparse']:
                    sparse_count += 1
                    # 记录sparse向量的详细信息
                    sparse_values = match.sparse_values
                    record_info['sparse_indices_count'] = len(sparse_values.get('indices', []))
                    record_info['sparse_values_count'] = len(sparse_values.get('values', []))
                    if record_info['sparse_indices_count'] > 0:
                        record_info['sparse_sample'] = {
                            'indices': sparse_values['indices'][:5],
                            'values': sparse_values['values'][:5]
                        }
                else:
                    dense_only_count += 1
                
                sample_records.append(record_info)
            
            stats = {
                'total_checked': total_records,
                'sparse_count': sparse_count,
                'dense_only_count': dense_only_count,
                'sparse_percentage': (sparse_count / total_records * 100) if total_records > 0 else 0,
                'sample_records': sample_records
            }
            
            return stats
            
        except Exception as e:
            print(f"检查sparse向量时出错: {e}")
            return {
                'error': str(e),
                'total_checked': 0,
                'sparse_count': 0,
                'dense_only_count': 0,
                'sparse_percentage': 0,
                'sample_records': []
            }

    def fetch_record_details(self, record_id: str) -> dict:
        """
        获取指定记录的详细信息
        :param record_id: 记录ID
        :return: 记录详细信息
        """
        try:
            # 使用fetch方法获取特定记录
            fetch_result = self.index.fetch(ids=[record_id])
            
            if record_id in fetch_result.vectors:
                vector = fetch_result.vectors[record_id]
                return {
                    'id': record_id,
                    'has_sparse': hasattr(vector, 'sparse_values') and vector.sparse_values is not None,
                    'metadata': vector.metadata,
                    'dense_vector_length': len(vector.values) if hasattr(vector, 'values') else 0,
                    'sparse_info': {
                        'indices_count': len(vector.sparse_values['indices']) if hasattr(vector, 'sparse_values') and vector.sparse_values else 0,
                        'values_count': len(vector.sparse_values['values']) if hasattr(vector, 'sparse_values') and vector.sparse_values else 0,
                        'sample_indices': vector.sparse_values['indices'][:5] if hasattr(vector, 'sparse_values') and vector.sparse_values and len(vector.sparse_values['indices']) > 0 else [],
                        'sample_values': vector.sparse_values['values'][:5] if hasattr(vector, 'sparse_values') and vector.sparse_values and len(vector.sparse_values['values']) > 0 else []
                    }
                }
            else:
                return {'error': f'Record {record_id} not found'}
                
        except Exception as e:
            return {'error': f'Fetch record failed: {e}'}

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