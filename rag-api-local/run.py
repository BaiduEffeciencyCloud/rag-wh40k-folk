#!/usr/bin/env python3
"""
RAG API Local 启动脚本
用于本地测试和开发
"""

import uvicorn
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 启动RAG检索服务API...")
    print("📍 服务地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("🔍 健康检查: http://localhost:8000/health/")
    print("⏹️  按 Ctrl+C 停止服务")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
