#!/bin/bash
# RAG API 服务停止脚本

echo "⏹️  正在停止 RAG API 服务..."

# 1. 优雅停止
pkill -f "rag-api-local.run" 2>/dev/null || true

# 2. 等待进程结束
sleep 2

# 3. 检查端口是否释放
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  端口仍被占用，强制清理..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# 4. 确认端口已释放
if lsof -i :8000 >/dev/null 2>&1; then
    echo "❌ 端口仍被占用"
    lsof -i :8000
    exit 1
else
    echo "✅ 服务已停止，端口已释放"
fi
