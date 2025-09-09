import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .api import query, intent, health, streamquery
from .services.model_service import ModelService

logger = logging.getLogger(__name__)

# 创建模型服务实例
model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时预加载模型
    await model_service.preload_models()
    
    yield
    
    # 关闭时清理资源
    await model_service.cleanup_models()

app = FastAPI(
    title="RAG检索服务API", 
    version="1.0.0",
    lifespan=lifespan
)

# 将模型服务注入到应用上下文
app.state.model_service = model_service

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(streamquery.router, prefix="/streamquery", tags=["streamquery"])
app.include_router(intent.router, prefix="/intent", tags=["intent"])

@app.get("/")
async def root():
    return {"message": "RAG检索服务API", "version": "1.0.0"}

@app.get("/models/status")
async def get_models_status():
    """获取模型加载状态"""
    model_manager = model_service.get_model_manager()
    return {
        "loaded_models": model_manager.get_loaded_models(),
        "sentence_transformer_loaded": model_manager.is_model_loaded('shibing624/text2vec-base-chinese')
    }
