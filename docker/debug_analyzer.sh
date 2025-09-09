#!/bin/bash

# OpenSearch分析器调试脚本
# 用于诊断分词问题

# 配置
OPENSEARCH_URL="https://localhost:9200"
USERNAME="admin"
PASSWORD="Francis198!!!"
INDEX_NAME="40ktest"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_query() {
    echo -e "${BLUE}[QUERY]${NC} $1"
}

# 测试不同的分析器
test_different_analyzers() {
    local query_text="$1"
    
    print_info "测试不同分析器的分词效果..."
    echo ""
    
    # 测试标准分析器
    print_query "1. 标准分析器 (standard):"
    test_analyzer "$query_text" "standard"
    echo ""
    
    # 测试IK分析器
    print_query "2. IK分析器 (ik_smart):"
    test_analyzer "$query_text" "ik_smart"
    echo ""
    
    # 测试warhammer_search_analyzer
    print_query "3. Warhammer搜索分析器:"
    test_analyzer "$query_text" "warhammer_search_analyzer"
    echo ""
    
    # 测试warhammer_index_analyzer
    print_query "4. Warhammer索引分析器:"
    test_analyzer "$query_text" "warhammer_index_analyzer"
    echo ""
}

# 测试分析器
test_analyzer() {
    local query_text="$1"
    local analyzer_name="$2"
    
    local url="$OPENSEARCH_URL/$INDEX_NAME/_analyze"
    local payload="{\"analyzer\": \"$analyzer_name\", \"text\": \"$query_text\"}"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        -k \
        -u "$USERNAME:$PASSWORD" \
        --connect-timeout 10 \
        --max-time 10)
    
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        # 解析分词结果
        tokens=$(echo "$response_body" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'tokens' in data and isinstance(data['tokens'], list):
        tokens = []
        for token in data['tokens']:
            if isinstance(token, dict):
                token_text = token.get('token', '')
                token_type = token.get('type', 'unknown')
                token_position = token.get('position', 0)
                tokens.append(f'{token_text} (类型: {token_type}, 位置: {token_position})')
        print('  ' + ', '.join(tokens))
    else:
        print('  响应格式异常')
except Exception as e:
    print('  解析失败: ' + str(e))
" 2>/dev/null)
        
        if [ -n "$tokens" ]; then
            echo "$tokens"
        else
            echo "  无法解析分词结果"
        fi
    else
        print_error "分析器测试失败，HTTP状态码: $http_code"
    fi
}

# 检查词典文件
check_dict_files() {
    print_info "检查词典文件..."
    echo ""
    
    # 检查warhammer_dict.txt
    dict_file="opensearch-config/analysis/warhammer_dict.txt"
    if [ -f "$dict_file" ]; then
        print_info "词典文件存在: $dict_file"
        
        # 检查文件大小
        file_size=$(wc -c < "$dict_file")
        print_info "文件大小: ${file_size} 字节"
        
        # 检查是否包含"支派武士"
        if grep -q "支派武士" "$dict_file"; then
            print_info "✅ 词典文件包含 '支派武士'"
            grep -n "支派武士" "$dict_file" | head -3
        else
            print_error "❌ 词典文件不包含 '支派武士'"
        fi
        
        # 检查文件编码
        file_encoding=$(file -bi "$dict_file" | cut -d';' -f2 | cut -d'=' -f2)
        print_info "文件编码: $file_encoding"
        
    else
        print_error "词典文件不存在: $dict_file"
    fi
    
    echo ""
    
    # 检查warhammer_synonyms.txt
    synonyms_file="opensearch-config/analysis/warhammer_synonyms.txt"
    if [ -f "$synonyms_file" ]; then
        print_info "同义词文件存在: $synonyms_file"
        file_size=$(wc -c < "$synonyms_file")
        print_info "文件大小: ${file_size} 字节"
    else
        print_error "同义词文件不存在: $synonyms_file"
    fi
}

# 检查索引设置
check_index_settings() {
    print_info "检查索引设置..."
    echo ""
    
    response=$(curl -s -k -u "$USERNAME:$PASSWORD" "$OPENSEARCH_URL/$INDEX_NAME/_settings")
    
    if [ $? -eq 0 ]; then
        print_info "索引设置获取成功"
        echo "分析器配置:"
        echo "$response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if '$INDEX_NAME' in data and 'settings' in data['$INDEX_NAME']:
        settings = data['$INDEX_NAME']['settings']
        if 'index' in settings and 'analysis' in settings['index']:
            analysis = settings['index']['analysis']
            print('  Analyzers:')
            if 'analyzer' in analysis:
                for name, config in analysis['analyzer'].items():
                    print(f'    {name}: {config}')
            print('  Filters:')
            if 'filter' in analysis:
                for name, config in analysis['filter'].items():
                    print(f'    {name}: {config}')
        else:
            print('  未找到分析器配置')
    else:
        print('  索引不存在或配置异常')
except Exception as e:
    print('  解析失败: ' + str(e))
" 2>/dev/null || echo "  无法解析索引设置"
    else
        print_error "获取索引设置失败"
    fi
}

# 主函数
main() {
    print_info "OpenSearch分析器调试脚本启动"
    echo ""
    
    # 检查词典文件
    check_dict_files
    
    # 检查索引设置
    check_index_settings
    
    echo ""
    print_info "开始测试分词效果..."
    echo ""
    
    # 测试查询
    query_text="艾达灵族有哪些支派武士"
    print_query "测试查询: \"$query_text\""
    echo ""
    
    # 测试不同分析器
    test_different_analyzers "$query_text"
    
    echo ""
    print_info "调试完成！"
}

# 执行主函数
main
