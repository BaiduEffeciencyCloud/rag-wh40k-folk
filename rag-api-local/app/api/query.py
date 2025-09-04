from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest
from app.models.response import QueryResponse
from app.services.rag_service import RAGService
from datetime import datetime
import uuid

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        rag_service = RAGService()
        result = await rag_service.process_query(request.query, request.advance)
        
        # 在接口处移除answers字段以减少网络开销
        if isinstance(result, dict) and 'results' in result:
            if isinstance(result['results'], dict) and 'answers' in result['results']:
                del result['results']['answers']
        
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
