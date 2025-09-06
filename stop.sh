#!/bin/bash
# RAG API 服务停止脚本（根目录）

echo "停止 RAG API 服务..."

# 1) 优雅停止：按端口定位PID并发送 SIGTERM
PIDS=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  echo "发送 SIGTERM 到进程: $PIDS"
  echo "$PIDS" | xargs -r kill 2>/dev/null || true
else
  # 兜底：按常见启动方式匹配（python3 run.py / uvicorn）
  pkill -f "python3 run.py" 2>/dev/null || true
  pkill -f "uvicorn" 2>/dev/null || true
fi

# 2) 等待进程退出
sleep 2

# 3) 如仍占用端口，发送 SIGKILL 强制清理
if lsof -i :8000 >/dev/null 2>&1; then
  echo "端口仍被占用，发送 SIGKILL 清理..."
  lsof -ti :8000 | xargs -r kill -9 2>/dev/null || true
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


