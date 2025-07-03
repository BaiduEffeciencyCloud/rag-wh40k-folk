import logging
import asyncio
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

from query_cot import QueryCoTExpander
from vector_search import VectorSearch
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    LOG_FORMAT,
    LOG_LEVEL,
    LLM_MODEL,
    QUERY_TIMEOUT,
    ANSWER_GENERATION_TEMPERATURE,
    ANSWER_MAX_TOKENS,
    DEDUPLICATION_THRESHOLD,
    DEFAULT_MAX_WORKERS,
    MAX_CONTEXT_RESULTS
)
from searchengine.subquery_retriever import SubQueryRetriever

# 配置日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """查询结果数据类"""
    query: str
    query_type: str
    results: List[Dict[str, Any]]
    score: float
    timestamp: float

@dataclass
class IntegratedResult:
    """整合后的结果数据类"""
    text: str
    score: float
    source_queries: List[str]
    metadata: Dict[str, Any]

class CoTQueryProcessor:
    """CoT查询处理器 - 处理CoT生成的多个查询变体"""
    
    def __init__(self, cot_expander: QueryCoTExpander):
        self.cot_expander = cot_expander
        logger.info("CoTQueryProcessor初始化完成")
    
    def classify_query(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        # 澄清查询：包含"是什么"、"是什么？"等
        if re.search(r'是什么\??$', query_lower):
            return "clarification"
        
        # 效果查询：包含"如何"、"影响"、"效果"等
        if re.search(r'(如何|影响|效果|降低|提升)', query_lower):
            return "effect"
        
        # 应用查询：包含"使用"、"应用"、"战术"等
        if re.search(r'(使用|应用|战术|策略)', query_lower):
            return "application"
        
        # 默认分类
        return "general"
    
    def process_query(self, user_query: str) -> Dict[str, List[str]]:
        """
        处理用户查询，生成结构化的查询变体
        
        Args:
            user_query: 原始用户查询
            
        Returns:
            按类型分组的查询变体字典
        """
        try:
            # 使用CoT扩展器生成查询变体
            expanded_queries = self.cot_expander.expand_query(user_query)
            
            # 按类型分类查询
            query_groups = {
                "clarification": [],
                "effect": [],
                "application": [],
                "general": []
            }
            
            for query in expanded_queries:
                query_type = self.classify_query(query)
                query_groups[query_type].append(query)
            
            # 记录分类结果
            logger.info(f"查询分类结果: {query_groups}")
            
            return query_groups
            
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            return {"general": [user_query]}

class MultiQueryRetriever:
    """多查询检索器 - 并行处理多个查询变体"""
    
    def __init__(self, vector_search: VectorSearch, max_workers: int = DEFAULT_MAX_WORKERS):
        self.vector_search = vector_search
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"MultiQueryRetriever初始化完成，最大并发数: {max_workers}")
    
    def retrieve_single_query(self, query: str, query_type: str, top_k: int = DEFAULT_TOP_K) -> QueryResult:
        """检索单个查询"""
        try:
            start_time = time.time()
            # 使用新的相邻切片检索方法
            results = self.vector_search.search_with_adjacent_chunks(query, top_k, adjacent_range=2)
            end_time = time.time()
            
            # 计算平均分数
            avg_score = sum(r.get('score', 0) for r in results) / len(results) if results else 0
            
            return QueryResult(
                query=query,
                query_type=query_type,
                results=results,
                score=avg_score,
                timestamp=end_time - start_time
            )
            
        except Exception as e:
            logger.error(f"检索查询 '{query}' 时出错: {e}")
            return QueryResult(
                query=query,
                query_type=query_type,
                results=[],
                score=0.0,
                timestamp=0.0
            )
    
    def retrieve_for_queries(self, query_groups: Dict[str, List[str]], top_k: int = DEFAULT_TOP_K) -> Dict[str, List[QueryResult]]:
        """
        为每组查询执行检索
        
        Args:
            query_groups: 按类型分组的查询列表
            top_k: 每个查询返回的结果数量
            
        Returns:
            每组查询的检索结果
        """
        try:
            all_results = {}
            
            # 为每个查询类型并行执行检索
            for query_type, queries in query_groups.items():
                if not queries:
                    all_results[query_type] = []
                    continue
                
                logger.info(f"开始检索 {query_type} 类型的 {len(queries)} 个查询")
                
                # 并行执行检索
                futures = []
                for query in queries:
                    future = self.executor.submit(self.retrieve_single_query, query, query_type, top_k)
                    futures.append(future)
                
                # 收集结果
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=QUERY_TIMEOUT)  # 使用配置的超时时间
                        results.append(result)
                    except Exception as e:
                        logger.error(f"获取检索结果时出错: {e}")
                
                all_results[query_type] = results
                logger.info(f"{query_type} 类型检索完成，获得 {len(results)} 个结果")
            
            return all_results
            
        except Exception as e:
            logger.error(f"多查询检索时出错: {e}")
            return {}

class ResultIntegrator:
    """结果整合器 - 智能整合多个查询的检索结果"""
    
    def __init__(self, deduplication_threshold: float = DEDUPLICATION_THRESHOLD):
        self.deduplication_threshold = deduplication_threshold
        logger.info(f"ResultIntegrator初始化完成，去重阈值: {deduplication_threshold}")
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版本）"""
        # 这里使用简单的Jaccard相似度，实际项目中可以使用更复杂的算法
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重处理"""
        if not results:
            return results
        
        deduplicated = []
        seen_texts = set()
        
        for result in results:
            text = result.get('text', '').strip()
            if not text:
                continue
            
            # 检查是否与已有结果重复
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = self.calculate_text_similarity(text, seen_text)
                if similarity > self.deduplication_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_texts.add(text)
        
        logger.info(f"去重处理: {len(results)} -> {len(deduplicated)}")
        return deduplicated
    
    def rank_results(self, results: List[Dict]) -> List[Dict]:
        """按相关性排序"""
        # 按分数降序排序
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_results
    
    def integrate_results(self, query_results: Dict[str, List[QueryResult]]) -> List[IntegratedResult]:
        """
        整合多个查询的检索结果
        
        Args:
            query_results: 每个查询的检索结果
            
        Returns:
            整合后的结果列表
        """
        try:
            all_results = []
            
            # 收集所有检索结果
            for query_type, query_result_list in query_results.items():
                for query_result in query_result_list:
                    for result in query_result.results:
                        # 添加查询信息到结果中
                        result['source_query'] = query_result.query
                        result['query_type'] = query_result.query_type
                        result['query_score'] = query_result.score
                        all_results.append(result)
            
            if not all_results:
                return []
            
            # 去重处理
            deduplicated_results = self.deduplicate_results(all_results)
            
            # 排序处理
            ranked_results = self.rank_results(deduplicated_results)
            
            # 转换为IntegratedResult格式
            integrated_results = []
            for result in ranked_results:
                integrated_result = IntegratedResult(
                    text=result.get('text', ''),
                    score=result.get('score', 0.0),
                    source_queries=[result.get('source_query', '')],
                    metadata={
                        'query_type': result.get('query_type', ''),
                        'query_score': result.get('query_score', 0.0),
                        'original_metadata': result.get('metadata', {})
                    }
                )
                integrated_results.append(integrated_result)
            
            logger.info(f"结果整合完成: {len(integrated_results)} 个整合结果")
            return integrated_results
            
        except Exception as e:
            logger.error(f"整合结果时出错: {e}")
            return []

class HierarchicalAnswerGenerator:
    """分层答案生成器 - 基于不同层次的查询生成结构化答案"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        logger.info("HierarchicalAnswerGenerator初始化完成")
    
    def generate_answer(self, user_query: str, integrated_results: List[IntegratedResult]) -> str:
        """
        生成分层的最终答案
        
        Args:
            user_query: 原始用户查询
            integrated_results: 整合后的检索结果
            
        Returns:
            结构化的最终答案
        """
        try:
            if not integrated_results:
                return "抱歉，没有找到相关信息。"
            
            # 提取检索到的文本内容
            retrieved_texts = [result.text for result in integrated_results[:MAX_CONTEXT_RESULTS]]  # 限制上下文结果数
            
            # 构建分层答案生成的提示
            prompt = f"""你是一位专业的战锤40K规则专家。请基于提供的上下文信息，为用户提供准确、全面且结构化的答案。

## 用户问题
{user_query}

## 上下文信息
{chr(10).join([f"【{i+1}】{text}" for i, text in enumerate(retrieved_texts)])}

## 回答要求
1. **结构化组织**：按照"定义→效果→应用"的结构组织答案
2. **信息整合**：智能整合来自不同查询的信息，避免重复
3. **准确性**：严格基于提供的上下文信息，不添加推测内容
4. **完整性**：确保答案完整回答用户问题
5. **清晰性**：使用清晰、易懂的语言表达

## 回答结构
请按照以下结构组织答案：

### 核心定义
- 解释相关单位、技能或规则的基本概念

### 具体效果
- 详细说明相关机制和效果
- 包括触发条件、影响范围等

### 实际应用
- 说明如何在实际游戏中使用
- 提供战术建议和注意事项

请直接输出结构化的答案，无需添加"问题："、"回答："等标签。"""

            # 调用LLM生成答案
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=ANSWER_GENERATION_TEMPERATURE,
                max_tokens=ANSWER_MAX_TOKENS
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("分层答案生成完成")
            return answer
            
        except Exception as e:
            logger.error(f"生成答案时出错: {e}")
            return "抱歉，生成答案时出现错误。"

class CoTSearch:
    """CoT搜索主类 - 整合所有组件"""
    
    def __init__(self, 
                 pinecone_api_key: str = PINECONE_API_KEY,
                 index_name: str = PINECONE_INDEX,
                 openai_api_key: str = OPENAI_API_KEY,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_workers: int = DEFAULT_MAX_WORKERS):
        """
        初始化CoT搜索系统
        
        Args:
            pinecone_api_key: Pinecone API密钥
            index_name: Pinecone索引名称
            openai_api_key: OpenAI API密钥
            temperature: 生成温度
            max_workers: 最大并发工作线程数
        """
        # 初始化组件
        self.cot_expander = QueryCoTExpander(openai_api_key, temperature)
        self.vector_search = VectorSearch(pinecone_api_key, index_name, openai_api_key)
        
        # 初始化处理器
        self.query_processor = CoTQueryProcessor(self.cot_expander)
        self.retriever = MultiQueryRetriever(self.vector_search, max_workers)
        self.integrator = ResultIntegrator()
        self.answer_generator = HierarchicalAnswerGenerator(self.vector_search.client)
        self.subquery_retriever = SubQueryRetriever(pinecone_api_key, index_name, openai_api_key)
        
        logger.info("CoTSearch系统初始化完成")
    
    def extract_entity(self, query: str) -> str:
        """
        简单规则：取第一个连续的中文/英文词组作为实体（可替换为更复杂的NLP方法）
        """
        import re
        m = re.search(r'([\u4e00-\u9fa5A-Za-z0-9_]+)', query)
        return m.group(1) if m else query[:4]

    def search(self, user_query: str, top_k: int = DEFAULT_TOP_K, use_new_retriever: bool = False) -> Dict[str, Any]:
        """
        执行CoT搜索
        
        Args:
            user_query: 用户查询
            top_k: 每个查询返回的结果数量
            use_new_retriever: 是否使用新的检索逻辑
            
        Returns:
            包含搜索结果的字典
        """
        try:
            start_time = time.time()
            logger.info(f"开始CoT搜索: {user_query}")
            
            # 第一步：处理查询，生成查询变体
            query_groups = self.query_processor.process_query(user_query)
            all_queries = [q for group in query_groups.values() for q in group]
            # 新检索逻辑：实体识别+filter召回
            if use_new_retriever:
                logger.info("使用SubQueryRetriever进行检索...")
                entities = [self.extract_entity(q) for q in all_queries]
                chunk_results = self.subquery_retriever.retrieve_for_queries(all_queries, entities, top_k)
                # 转为原有格式（每个query一个QueryResult）
                query_results = {}
                idx = 0
                for group, queries in query_groups.items():
                    group_results = []
                    for q in queries:
                        results = chunk_results[idx]
                        avg_score = sum(r.get('score', 0) for r in results) / len(results) if results else 0
                        group_results.append(QueryResult(
                            query=q,
                            query_type=group,
                            results=results,
                            score=avg_score,
                            timestamp=0.0
                        ))
                        idx += 1
                    query_results[group] = group_results
            else:
                # 原有检索逻辑
                query_results = self.retriever.retrieve_for_queries(query_groups, top_k)
            
            # 第三步：整合结果
            integrated_results = self.integrator.integrate_results(query_results)
            
            # 第四步：生成最终答案
            final_answer = self.answer_generator.generate_answer(user_query, integrated_results)
            
            end_time = time.time()
            
            # 构建返回结果
            result = {
                'user_query': user_query,
                'final_answer': final_answer,
                'query_groups': query_groups,
                'query_results': query_results,
                'integrated_results': integrated_results,
                'processing_time': end_time - start_time,
                'total_queries': sum(len(queries) for queries in query_groups.values()),
                'total_results': len(integrated_results)
            }
            
            logger.info(f"CoT搜索完成，耗时: {result['processing_time']:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"CoT搜索时出错: {e}")
            return {
                'user_query': user_query,
                'final_answer': f"抱歉，搜索时出现错误: {str(e)}",
                'error': str(e)
            }
    
    def close(self):
        """关闭资源"""
        try:
            self.retriever.executor.shutdown(wait=True)
            logger.info("CoTSearch资源已关闭")
        except Exception as e:
            logger.error(f"关闭资源时出错: {e}")


def main():
    """主函数：用于测试CoTSearch"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CoT搜索系统测试')
    parser.add_argument('query', type=str, help='要搜索的查询')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K, help='每个查询返回的结果数量')
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS, help='最大并发工作线程数')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('--use-new', action='store_true', help='使用新检索逻辑（SubQueryRetriever）')
    args = parser.parse_args()
    
    print("🤖 CoT搜索系统测试")
    print("=" * 60)
    print(f"📝 查询: {args.query}")
    print(f"📊 每个查询结果数: {args.top_k}")
    print(f"🔧 最大并发数: {args.max_workers}")
    print(f"🆕 使用新检索逻辑: {args.use_new}")
    print("-" * 60)
    
    try:
        # 创建CoT搜索实例
        cot_search = CoTSearch(max_workers=args.max_workers)
        
        # 执行搜索
        result = cot_search.search(args.query, args.top_k, use_new_retriever=args.use_new)
        
        # 显示结果
        print("\n📋 搜索结果:")
        print("=" * 60)
        print(f"⏱️  处理时间: {result.get('processing_time', 0):.2f}秒")
        print(f"🔍 总查询数: {result.get('total_queries', 0)}")
        print(f"📄 总结果数: {result.get('total_results', 0)}")
        print("-" * 60)
        
        print("\n💡 最终答案:")
        print(result.get('final_answer', '无答案'))
        
        # 显示详细信息
        if args.verbose:
            print("\n🔍 详细信息:")
            print(f"查询分组: {result.get('query_groups', {})}")
            print(f"整合结果数量: {len(result.get('integrated_results', []))}")
        
        print("=" * 60)
        print("✅ 搜索完成！")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    finally:
        # 清理资源
        try:
            cot_search.close()
        except:
            pass


if __name__ == "__main__":
    main() 