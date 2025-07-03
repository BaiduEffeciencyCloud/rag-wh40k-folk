import os
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Dict, Any
import logging
from config import OPENAI_API_KEY, RERANK_MODEL, EMBADDING_MODEL, get_embedding_model, BM25_K1, BM25_B, BM25_MIN_FREQ, BM25_MAX_VOCAB_SIZE, BM25_CUSTOM_DICT_PATH, BM25_MODEL_PATH, BM25_VOCAB_PATH
import pinecone
from DATAUPLOD.bm25_manager import BM25Manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorSearch:
    def __init__(self, pinecone_api_key: str, index_name: str, openai_api_key=OPENAI_API_KEY):
        """
        初始化向量搜索类
        
        Args:
            pinecone_api_key: Pinecone API密钥
            index_name: Pinecone索引名称
            openai_api_key: OpenAI API密钥
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = get_embedding_model()
        self.bm25 = BM25Manager(
            k1=BM25_K1,
            b=BM25_B,
            min_freq=BM25_MIN_FREQ,
            max_vocab_size=BM25_MAX_VOCAB_SIZE,
            custom_dict_path=BM25_CUSTOM_DICT_PATH
        )
        # 尝试加载已训练模型
        try:
            self.bm25.load_model(BM25_MODEL_PATH, BM25_VOCAB_PATH)
        except Exception as e:
            logger.warning(f"BM25模型加载失败: {e}，请先训练BM25模型")
        logger.info("VectorSearch初始化完成")
        
    def generate_query_variants(self, query: str) -> List[str]:
        """
        生成查询变体
        
        Args:
            query: 原始查询
            
        Returns:
            List[str]: 查询变体列表
        """
        try:
            prompt = f"""请基于以下问题生成3个相关的查询变体，每个变体都应该能够获取部分答案：

原始问题：{query}

要求：
1. 变体应该包含原始问题的关键信息
2. 变体之间应该有逻辑关联
3. 变体应该从不同角度探索原始问题
4. 变体应该具体且明确

请生成3个查询变体："""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            variants = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
            logger.info(f"生成的查询变体：{variants}")
            return variants
            
        except Exception as e:
            logger.error(f"生成查询变体时出错：{str(e)}")
            return []
            
    def semantic_parse(self, query: str, context: List[str]) -> str:
        """
        使用OpenAI进行语义解析
        
        Args:
            query: 用户问题
            context: 上下文信息
            
        Returns:
            str: 解析后的回答
        """
        try:
            prompt = f"""# 角色：战锤40K规则专家
你是一名专业的战锤40K比赛规则专家，具备精准的信息提取与分析能力。你的任务是基于提供的上下文资料，为用户提供准确、全面且有条理的回答。

## 核心要求
1. **严格基于上下文**：只能使用提供的上下文信息回答，绝对不允许添加任何推测、虚构或不在上下文中的内容
2. **完整性优先**：必须整合所有相关的规则信息，包括主要规则、例外情况、限制条件和特殊说明
3. **结构化回答**：按照逻辑顺序组织信息，先说明核心规则，再补充例外和细节
4. **专业术语**：使用战锤40K的标准术语和表达方式

## 严格限制
- **禁止推测**：如果上下文中没有明确提到某个武器、单位或规则，绝对不能在答案中提及
- **禁止泛化**：不要使用"通常"、"一般"、"可能"等模糊表述，只陈述上下文中明确存在的信息
- **禁止添加**：不要添加任何不在上下文中的具体武器名称、单位名称或规则细节
- **明确标注**：如果上下文信息不足以回答完整问题，明确说明"根据提供的资料，无法确定..."

## 回答结构
请按照以下结构组织回答：

### 核心规则
- 直接回答用户问题的核心内容
- 说明基本机制和效果
- **只基于上下文中的明确信息**

### 详细说明
- 补充相关的规则细节
- 说明使用条件和限制
- **严格限制在上下文范围内**

### 例外情况
- **必须仔细检查所有上下文，寻找任何例外规则**
- 特别关注包含以下关键词的内容：
  * "但是"、"然而"、"不过"
  * "例外"、"特殊"、"允许"
  * "如果...那么"、"尽管...仍然"
  * "突击武器"、"特殊能力"、"技能"
- 列出所有相关的例外规则和特殊情况
- 说明例外规则的具体条件和效果
- **只包含上下文中明确提到的例外**

### 注意事项
- 重要的提醒和警告
- 容易忽略的关键点
- **基于上下文中的具体信息**

## 信息整合原则
1. **严格上下文限制**：只使用提供的上下文信息，不添加任何外部知识
2. **全面扫描例外**：仔细检查每个上下文片段，寻找任何形式的例外情况
3. **关键词识别**：特别关注包含"突击"、"特殊"、"例外"、"允许"等关键词的内容
4. **逻辑完整性**：确保答案包含完整的规则逻辑，包括所有例外情况
5. **避免遗漏**：如果上下文中提到任何例外或特殊情况，必须包含在答案中
6. **保持逻辑性**：按照规则逻辑顺序组织信息，避免跳跃性表述
7. **突出关键信息**：用清晰的语言强调重要的规则要点

## 输出格式
直接输出结构化的答案，无需添加"问题："、"回答："等标签。确保答案完整、准确、易于理解，且严格基于提供的上下文信息。

上下文信息：
{chr(10).join([f"【{i+1}】{text}" for i, text in enumerate(context)])}

用户问题：{query}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,  # 进一步降低温度以获得更稳定的输出
                max_tokens=3000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"语义解析时出错：{str(e)}")
            return "抱歉，解析问题时出错。"
            
    def get_embedding(self, text):
        # 统一通过config.get_embedding_model()获取embedding
        embedding = self.embedding_model.embed_query(text)
        return embedding

    def get_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """
        通过BM25Manager生成稀疏向量
        """
        return self.bm25.get_sparse_vector(text)

    def hybrid_scale(self, dense: List[float], sparse: Dict[str, Any], alpha: float):
        """
        按alpha加权dense和sparse embedding
        """
        hdense = [v * alpha for v in dense]
        hsparse = {"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]}
        return hdense, hsparse

    def search(self, query: str, top_k: int = 10, alpha: float = 0.3) -> List[Dict[str, Any]]:
        """
        执行hybrid向量搜索，返回原始检索结果
        Args:
            query: 查询文本
            top_k: 返回结果数量
            alpha: dense/sparse加权参数
        Returns:
            List[Dict[str, Any]]: 原始搜索结果列表，包含metadata和score
        """
        try:
            # 获取dense embedding
            query_embedding = self.get_embedding(query)
            # 获取sparse embedding
            query_sparse = self.get_sparse_embedding(query)
            # 按alpha加权
            hdense, hsparse = self.hybrid_scale(query_embedding, query_sparse, alpha)
            
            # Pinecone官方hybrid search接口
            results = self.index.query(
                vector=hdense,
                sparse_vector=hsparse,
                top_k=top_k,
                include_metadata=True
            )
            
            # 返回原始检索结果
            search_results = []
            for match in results.matches:
                text = match.metadata.get('text', '')
                # 过滤空内容或过短的内容
                if not text or len(text.strip()) < 10:
                    continue
                search_results.append({
                    'text': text,
                    'score': match.score,
                    'metadata': match.metadata
                })
            return search_results
        except Exception as e:
            logger.error(f"搜索时出错：{str(e)}")
            return []

    def search_with_adjacent_chunks(self, query: str, top_k: int = 10, adjacent_range: int = 2) -> List[Dict[str, Any]]:
        """
        执行向量搜索，并召回相邻切片
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            adjacent_range: 相邻切片范围（前后各几个切片）
            
        Returns:
            List[Dict[str, Any]]: 包含相邻切片的搜索结果列表
        """
        try:
            # 首先执行常规搜索
            initial_results = self.search(query, top_k)
            
            if not initial_results:
                return []
            
            # 收集所有需要获取的切片ID
            chunk_ids_to_fetch = set()
            
            for result in initial_results:
                metadata = result.get('metadata', {})
                chunk_id = metadata.get('chunk_id')
                if chunk_id:
                    # 添加当前切片ID
                    chunk_ids_to_fetch.add(chunk_id)
                    
                    # 添加相邻切片ID
                    try:
                        chunk_num = int(chunk_id)
                        for i in range(1, adjacent_range + 1):
                            chunk_ids_to_fetch.add(str(chunk_num - i))  # 前几个切片
                            chunk_ids_to_fetch.add(str(chunk_num + i))  # 后几个切片
                    except ValueError:
                        # 如果chunk_id不是数字，跳过相邻切片逻辑
                        pass
            
            # 获取所有相关切片的内容
            all_results = initial_results.copy()
            
            # 通过ID获取相邻切片
            for chunk_id in chunk_ids_to_fetch:
                # 检查是否已经在初始结果中
                if any(r.get('metadata', {}).get('chunk_id') == chunk_id for r in initial_results):
                    continue
                
                # 通过ID查询切片
                try:
                    fetch_results = self.index.query(
                        id=chunk_id,
                        include_metadata=True,
                        top_k=1
                    )
                    
                    if fetch_results.matches:
                        match = fetch_results.matches[0]
                        text = match.metadata.get('text', '')
                        if text and len(text.strip()) >= 10:
                            all_results.append({
                                'text': text,
                                'score': match.score,
                                'metadata': match.metadata,
                                'source': 'adjacent_chunk'  # 标记为相邻切片
                            })
                except Exception as e:
                    logger.warning(f"获取相邻切片 {chunk_id} 时出错: {e}")
                    continue
            
            # 按分数排序并去重
            seen_texts = set()
            final_results = []
            
            for result in sorted(all_results, key=lambda x: x.get('score', 0), reverse=True):
                text = result.get('text', '').strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    final_results.append(result)
            
            logger.info(f"原始检索: {len(initial_results)} 个结果，添加相邻切片后: {len(final_results)} 个结果")
            
            return final_results
            
        except Exception as e:
            logger.error(f"搜索相邻切片时出错：{str(e)}")
            return self.search(query, top_k)  # 回退到普通搜索
    
    def search_and_generate_answer(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        执行向量搜索并生成答案
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 包含生成答案的结果列表
        """
        try:
            # 执行纯检索
            search_results = self.search(query, top_k)
            
            if not search_results:
                return []
            
            # 提取检索到的文本内容
            retrieved_texts = [result['text'] for result in search_results]
            
            # 使用semantic_parse生成答案
            integrated_answer = self.semantic_parse(query, retrieved_texts)
            
            # 返回整合后的答案
            return [{
                'text': integrated_answer, 
                'score': 1.0,
                'retrieved_results': search_results
            }]
            
        except Exception as e:
            logger.error(f"搜索并生成答案时出错：{str(e)}")
            return []
    
    def search_and_integrate(self, query: str, top_k: int = 5) -> str:
        """
        搜索并整合结果（向后兼容方法）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            str: 整合后的答案
        """
        results = self.search_and_generate_answer(query, top_k)
        if not results:
            return "抱歉，没有找到相关信息。"
        return results[0]['text']

    def close(self):
        # 关闭资源
        pass  # 这里可以替换为实际的关闭逻辑

    def search_with_filter(self, query: str, entity: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        实体->filter+向量->召回切片demo
        Args:
            query: 查询文本
            entity: 识别到的实体（如单位、技能等）
            top_k: 返回结果数量
        Returns:
            List[Dict[str, Any]]: 检索结果列表，包含metadata和score
        """
        try:
            query_embedding = self.get_embedding(query)
            # Pinecone filter: hierarchy包含entity
            filter_dict = {"hierarchy": {"$contains": entity}}
            results = self.index.query(
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
            search_results = []
            for match in results.matches:
                text = match.metadata.get('text', '')
                if not text or len(text.strip()) < 10:
                    continue
                search_results.append({
                    'text': text,
                    'score': match.score,
                    'metadata': match.metadata
                })
            return search_results
        except Exception as e:
            logger.error(f"search_with_filter出错: {str(e)}")
            return []

# 确保类可以被导入
__all__ = ['VectorSearch']

def main():
    """
    主函数：处理命令行参数并执行向量搜索
    """
    import argparse
    import sys
    from config import PINECONE_API_KEY, PINECONE_INDEX
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='战锤40K向量搜索系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python3 vector_search.py "幽冥骑士的阔步巨兽技能如何使用？"
  python3 vector_search.py --top_k 10 "地形模型对移动有什么影响？"
  python3 vector_search.py --query "冲锋技能会造成多少伤害？"
        """
    )
    
    parser.add_argument(
        'query',
        type=str,
        nargs='?',
        help='要搜索的查询文本'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        dest='query_arg',
        help='要搜索的查询文本（可选参数形式）'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='返回结果数量（默认：5）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确定查询文本
    query = args.query or args.query_arg
    
    if not query:
        print("❌ 错误：请提供查询文本")
        print("使用方法：")
        print("  python3 vector_search.py '你的查询'")
        print("  python3 vector_search.py --query '你的查询'")
        sys.exit(1)
    
    try:
        # 创建向量搜索实例
        vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX
        )
        
        print("=" * 60)
        print("🔍 战锤40K向量搜索系统")
        print("=" * 60)
        print(f"📝 查询文本: {query}")
        print(f"📊 预期返回结果数: {args.top_k}")
        print("-" * 60)
        
        # 执行搜索（只观察召回效果）
        results = vector_search.search(query, args.top_k)
        
        if not results:
            print("❌ 没有找到相关结果")
            sys.exit(1)
        
        # 显示结果
        print(f"✅ 找到 {len(results)} 个召回结果:")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"📋 召回结果 {i}:")
            print(f"📊 相似度分数: {result.get('score', 'N/A'):.4f}")
            print(f"📄 内容:")
            print(result.get('text', '无内容'))
            print("\n" + "="*40 + "\n")
        
        print("=" * 60)
        print("✅ 搜索完成！")
        
        # 显示详细信息（如果启用）
        if args.verbose:
            print("\n🔍 详细信息:")
            print(f"   查询变体: {vector_search.generate_query_variants(query)}")
            print(f"   搜索结果数量: {len(results)}")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理资源
        try:
            vector_search.close()
        except:
            pass

if __name__ == "__main__":
    main() 