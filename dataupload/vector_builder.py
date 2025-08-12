"""
vector_builder.py
================

本模块为upsert.py等主流程提供可复用的向量构造与批量上传工具函数。

【用途说明】
- 统一封装"构造pinecone向量（含稠密+稀疏embedding+元数据）"的标准流程，避免主流程重复造轮子。
- 提供分批上传pinecone向量的通用工具，支持异常处理与日志，便于大规模数据入库。

【主要函数】
1. build_vector_data(chunk, embedding, bm25_manager, chunk_id, base_metadata)
   - 输入：单个chunk、其embedding、BM25Manager实例、chunk_id、基础元数据
   - 输出：包含id、values（稠密向量）、sparse_values（稀疏向量）、metadata的dict
   - 自动处理sparse embedding生成异常，保证主流程健壮

2. batch_upsert_vectors(pinecone_index, vectors, batch_size=100)
   - 输入：pinecone_index对象、待上传向量list、批次大小
   - 功能：分批上传，自动日志与异常处理，适配大规模数据

【典型用法】
>>> from dataupload.vector_builder import build_vector_data, batch_upsert_vectors
>>> vector = build_vector_data(chunk, embedding, bm25_manager, chunk_id, base_metadata)
>>> batch_upsert_vectors(pinecone_index, [vector1, vector2, ...], batch_size=100)

【设计原则】
- 只做"构造+上传"两件事，所有业务逻辑、参数组装、异常兜底均在主流程实现
- 便于单元测试和mock，接口参数显式、无副作用
- 代码可读性、可维护性优先

"""
import logging
from typing import List, Dict, Any
from config import OPENSEARCH_INDEX

def build_vector_data(chunk: dict, embedding: list, bm25_manager, chunk_id: str, base_metadata: dict) -> dict:
    """
    构造包含sparse_values的pinecone向量dict
    """
    # 记录sparse向量生成过程
    logging.info(f"[SPARSE_LOG] 开始处理chunk: {chunk_id}")
    logging.info(f"[SPARSE_LOG] chunk文本长度: {len(chunk['text'])}")
    
    try:
        if bm25_manager:
            sparse_embedding = bm25_manager.get_sparse_vector(chunk['text'])
            # 详细记录sparse向量信息
            if sparse_embedding:
                indices_count = len(sparse_embedding.get('indices', []))
                values_count = len(sparse_embedding.get('values', []))
                logging.info(f"[SPARSE_LOG] {chunk_id} - sparse向量生成成功:")
                logging.info(f"[SPARSE_LOG]   - indices数量: {indices_count}")
                logging.info(f"[SPARSE_LOG]   - values数量: {values_count}")
                if indices_count > 0:
                    logging.info(f"[SPARSE_LOG]   - 前5个indices: {sparse_embedding['indices'][:5]}")
                    logging.info(f"[SPARSE_LOG]   - 前5个values: {sparse_embedding['values'][:5]}")
                else:
                    logging.warning(f"[SPARSE_LOG] {chunk_id} - sparse向量为空!")
            else:
                logging.warning(f"[SPARSE_LOG] {chunk_id} - sparse_embedding为None")
        else:
            logging.warning(f"[SPARSE_LOG] {chunk_id} - bm25_manager为None，跳过sparse向量生成")
            sparse_embedding = None
    except Exception as e:
        logging.error(f"[SPARSE_LOG] {chunk_id} - sparse embedding生成失败: {e}")
        sparse_embedding = None
    
    metadata = {
        'text': chunk['text'],
        'chunk_type': chunk.get('chunk_type', ''),
        'content_type': chunk.get('content_type', ''),
        'faction': chunk.get('faction', base_metadata.get('faction', 'unknown')),
        'section_heading': chunk.get('section_heading', ''),
    }
    for j in range(1, 7):
        metadata[f'h{j}'] = chunk.get(f'h{j}', '')
    metadata['chunk_id'] = chunk_id
    metadata['source_file'] = chunk.get('source_file', '')
    
    # 记录最终构造的向量信息
    vector_data = {
        'id': chunk_id,
        'values': embedding,
        'metadata': metadata
    }
    
    # 只有当sparse_embedding存在且不为空时才添加sparse_values字段
    if sparse_embedding and sparse_embedding.get('indices') and len(sparse_embedding['indices']) > 0:
        vector_data['sparse_values'] = sparse_embedding
        logging.info(f"[SPARSE_LOG] {chunk_id} - 向量构造完成:")
        logging.info(f"[SPARSE_LOG]   - dense向量长度: {len(embedding)}")
        logging.info(f"[SPARSE_LOG]   - sparse向量: 有 (indices: {len(sparse_embedding['indices'])})")
        logging.info(f"[SPARSE_LOG]   - metadata字段数: {len(metadata)}")
    else:
        logging.info(f"[SPARSE_LOG] {chunk_id} - 向量构造完成:")
        logging.info(f"[SPARSE_LOG]   - dense向量长度: {len(embedding)}")
        logging.info(f"[SPARSE_LOG]   - sparse向量: 无 (跳过sparse_values字段)")
        logging.info(f"[SPARSE_LOG]   - metadata字段数: {len(metadata)}")
    
    return vector_data

def batch_upsert_vectors(db_conn, vectors: List[dict], batch_size: int = 100):
    """
    分批上传vectors到pinecone，异常处理与日志可选
    返回每个批次的成功状态列表
    """
    total = len(vectors)
    logging.info(f"[UPLOAD_LOG] 开始批量上传，总向量数: {total}")
    
    # 统计sparse向量情况
    sparse_count = sum(1 for v in vectors if v.get('sparse_values'))
    empty_sparse_count = sum(1 for v in vectors if not v.get('sparse_values'))
    logging.info(f"[UPLOAD_LOG] sparse向量统计: 有sparse={sparse_count}, 无sparse={empty_sparse_count}")
    
    batch_results = []  # 记录每个批次的成功状态
    
    for i in range(0, total, batch_size):
        batch = vectors[i:i+batch_size]
        batch_num = i//batch_size + 1
        
        # 检查当前批次的sparse向量情况
        batch_sparse_count = sum(1 for v in batch if v.get('sparse_values'))
        batch_empty_sparse_count = sum(1 for v in batch if not v.get('sparse_values'))
        logging.info(f"[UPLOAD_LOG] 批次 {batch_num}: {len(batch)} 个向量 (有sparse: {batch_sparse_count}, 无sparse: {batch_empty_sparse_count})")
        
        try:
            # 生成doc_ids: faction + "_" + (i + j)
            doc_ids = []
            for j, vector in enumerate(batch):
                faction = vector.get('metadata', {}).get('faction', 'unknown')
                doc_ids.append(f"{faction}_{i + j}")
            
            response = db_conn.bulk_upload_data(index_name=OPENSEARCH_INDEX, documents=batch, batch_size=batch_size, doc_ids=doc_ids)
            # 检查bulk upload的返回值，计算成功上传的文档数量
            upserted_count = sum(1 for item in response.get('items', []) 
                               if item.get('index', {}).get('result') == 'created')
            logging.info(f"[DEBUG] 批次 {batch_num} - Pinecone响应: {response}")
            logging.info(f"[DEBUG] 批次 {batch_num} - upserted_count: {upserted_count}, 批次大小: {len(batch)}")
            
            if upserted_count == len(batch):
                logging.info(f"✅ 上传批次 {batch_num}: {len(batch)} 个切片成功")
                batch_results.append(True)
                logging.info(f"[DEBUG] 批次 {batch_num} 返回: True")
            else:
                # 部分成功或失败
                logging.error(f"❌ 批次 {batch_num} 部分失败: 成功{upserted_count}/{len(batch)}")
                # 记录失败的批次详细信息
                logging.error(f"[UPLOAD_LOG] 部分失败批次详情:")
                for j, vector in enumerate(batch):
                    chunk_id = vector.get('id', f'unknown-{j}')
                    has_sparse = bool(vector.get('sparse_values'))
                    dense_len = len(vector.get('values', []))
                    logging.error(f"[UPLOAD_LOG]   - {chunk_id}: dense={dense_len}, sparse={'有' if has_sparse else '无'}")
                batch_results.append(False)
                logging.info(f"[DEBUG] 批次 {batch_num} 返回: False (部分失败)")
        except Exception as e:
            logging.error(f"❌ 批次 {batch_num} 上传失败: {e}")
            logging.error(f"[DEBUG] 批次 {batch_num} 异常类型: {type(e).__name__}")
            logging.error(f"[DEBUG] 批次 {batch_num} 异常详情: {str(e)}")
            # 记录失败的批次详细信息
            logging.error(f"[UPLOAD_LOG] 失败批次详情:")
            for j, vector in enumerate(batch):
                chunk_id = vector.get('id', f'unknown-{j}')
                has_sparse = bool(vector.get('sparse_values'))
                dense_len = len(vector.get('values', []))
                logging.error(f"[UPLOAD_LOG]   - {chunk_id}: dense={dense_len}, sparse={'有' if has_sparse else '无'}")
            # 不要重新抛出异常，而是返回False表示批次失败
            batch_results.append(False)
            logging.info(f"[DEBUG] 批次 {batch_num} 返回: False (异常)")
    
    return batch_results 