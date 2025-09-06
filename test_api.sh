#!/bin/bash

# RAG API 测试脚本（根目录）
# 用法: ./test_api.sh "查询文本" [-i|-s]
# -i: 使用 intent 接口
# -s: 使用 query 接口

# 默认配置
API_BASE_URL="http://localhost:8000"
INTENT_ENDPOINT="/intent/"
QUERY_ENDPOINT="/query/"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}RAG API 测试脚本${NC}"
    echo ""
    echo "用法:"
    echo "  $0 \"查询文本\" [-i|-s]"
    echo ""
    echo "参数:"
    echo "  查询文本    要测试的查询内容"
    echo "  -i         使用 intent 接口 (/intent)"
    echo "  -s         使用 query 接口 (/query)"
    echo ""
    echo "示例:"
    echo "  $0 \"什么是星际战士\" -i"
    echo "  $0 \"星际战士的装备\" -s"
    echo ""
}

# 检查参数
if [ $# -lt 2 ]; then
    echo -e "${RED}错误: 参数不足${NC}"
    show_help
    exit 1
fi

QUERY="$1"
INTERFACE="$2"

# 验证接口参数
if [ "$INTERFACE" != "-i" ] && [ "$INTERFACE" != "-s" ]; then
    echo -e "${RED}错误: 接口参数必须是 -i 或 -s${NC}"
    show_help
    exit 1
fi

# 确定使用的接口
if [ "$INTERFACE" = "-i" ]; then
    ENDPOINT="$INTENT_ENDPOINT"
    INTERFACE_NAME="Intent"
else
    ENDPOINT="$QUERY_ENDPOINT"
    INTERFACE_NAME="Query"
fi

echo -e "${BLUE}=== RAG API 测试 ===${NC}"
echo -e "${YELLOW}接口:${NC} $INTERFACE_NAME"
echo -e "${YELLOW}查询:${NC} $QUERY"
echo -e "${YELLOW}URL:${NC} $API_BASE_URL$ENDPOINT"
echo ""

# 构建请求数据（转义引号）
SAFE_QUERY=$(printf '%s' "$QUERY" | sed 's/"/\\"/g')
REQUEST_DATA=$(cat <<EOF
{
    "query": "$SAFE_QUERY"
}
EOF
)

echo -e "${BLUE}发送请求...${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s.%N)

RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$REQUEST_DATA" \
    "$API_BASE_URL$ENDPOINT")

# 记录结束时间并计算耗时（两位小数）
END_TIME=$(date +%s.%N)
DURATION=$(echo "scale=2; $END_TIME - $START_TIME" | bc -l)

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 请求失败，请检查服务是否启动${NC}"
    echo -e "${YELLOW}提示: 确保服务运行在 $API_BASE_URL${NC}"
    exit 1
fi

if [ -z "$RESPONSE" ]; then
    echo -e "${RED}错误: 服务器返回空响应${NC}"
    exit 1
fi

echo -e "${GREEN}=== 响应结果 ===${NC}"
echo -e "${YELLOW}请求耗时:${NC} ${DURATION} 秒"
echo ""

if command -v jq >/dev/null 2>&1; then
    echo "$RESPONSE" | jq .
else
    echo "$RESPONSE"
    echo ""
    echo -e "${YELLOW}提示: 安装 jq 可获得更好的 JSON 格式化显示${NC}"
fi

echo ""
echo -e "${GREEN}测试完成${NC}"


