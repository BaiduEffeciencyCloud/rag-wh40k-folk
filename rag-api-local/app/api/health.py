from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "intent_classifier": "healthy",
            "search_engine": "healthy",
            "aggregator": "healthy",
            "database": "healthy"
        }
    }
