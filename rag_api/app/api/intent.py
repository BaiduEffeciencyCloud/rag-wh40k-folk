from fastapi import APIRouter, HTTPException
from ..models.query import IntentRequest
from ..models.response import IntentResponse
from ..services.rag_service import RAGService
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=IntentResponse)
async def identify_intent(request: IntentRequest):
    try:
        rag_service = RAGService()
        result = await rag_service.identify_intent(request.query)
        
        return IntentResponse(
            status="success",
            data=result,
            error=None
        )
    except Exception as e:
        return IntentResponse(
            status="error",
            data=None,
            error={
                "code": "INTENT_RECOGNITION_FAILED",
                "message": str(e)
            }
        )
