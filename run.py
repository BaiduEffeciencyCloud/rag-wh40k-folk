#!/usr/bin/env python3
"""
RAG API 启动脚本（根目录）
"""

import logging
import uvicorn

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logging.info("启动RAG检索服务API...")
    logging.info("服务地址: http://localhost:8000")
    logging.info("API文档: http://localhost:8000/docs")
    logging.info("健康检查: http://localhost:8000/health/")
    logging.info("按 Ctrl+C 停止服务")
    logging.info("-" * 50)
    uvicorn.run(
        "rag_api.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()


