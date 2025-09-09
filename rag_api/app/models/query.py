from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询内容，不能为空")
    advance: bool = False
    options: Optional[Dict[str, Any]] = None

class IntentRequest(BaseModel):
    query: str = Field(..., min_length=1, description="查询内容，不能为空")
