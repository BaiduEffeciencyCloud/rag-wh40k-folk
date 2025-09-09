from fastapi import APIRouter, HTTPException
from ..models.query import QueryRequest
from ..models.response import QueryResponse
from ..services.rag_service import RAGService
from datetime import datetime
import uuid
from typing import Any, Iterable, AsyncIterable
import asyncio

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        rag_service = RAGService()
        result = await rag_service.process_query(request.query, request.advance)
        
        # 将流式结果在 /query 中汇总为完整文本，便于脚本消费
        if isinstance(result, dict) and 'results' in result:
            results_obj = result['results']
            if isinstance(results_obj, dict):
                # 汇总 final_answer_stream → final_answer
                stream_iter = results_obj.get('final_answer_stream')
                if stream_iter is not None:
                    text_parts = []
                    if hasattr(stream_iter, '__aiter__'):
                        async for chunk in stream_iter:  # type: ignore
                            if chunk is not None:
                                text_parts.append(str(chunk))
                    else:
                        for chunk in stream_iter:  # type: ignore
                            if chunk is not None:
                                text_parts.append(str(chunk))
                    results_obj['final_answer'] = ''.join(text_parts)
                    del results_obj['final_answer_stream']
                # 在接口处移除answers字段以减少网络开销
                if 'answers' in results_obj:
                    del results_obj['answers']
        
        return QueryResponse(
            status="success",
            data=result,
            error=None,
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        )
    except Exception as e:
        return QueryResponse(
            status="error",
            data=None,
            error={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            },
            timestamp=datetime.now().isoformat(),
            request_id=str(uuid.uuid4())
        )
