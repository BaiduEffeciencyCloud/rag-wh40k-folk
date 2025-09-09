#!/bin/bash

# OpenSearch分析器重新加载脚本
# 用于在不重建索引的情况下更新词典文件

# 配置
OPENSEARCH_URL="https://localhost:9200"
USERNAME="admin"
PASSWORD="Francis198!!!"
INDEX_NAME=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --index INDEX_NAME    指定要重新加载的索引名称"
    echo "  -a, --all                  重新加载所有索引的分析器"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -i my_index            # 重新加载指定索引的分析器"
    echo "  $0 -a                     # 重新加载所有索引的分析器"
    echo "  $0                        # 重新加载所有索引的分析器（默认）"
}

# 检查OpenSearch是否运行
check_opensearch() {
    print_info "检查OpenSearch连接..."
    
    if curl -s -k -u "$USERNAME:$PASSWORD" "$OPENSEARCH_URL/_cluster/health" > /dev/null 2>&1; then
        print_info "OpenSearch连接正常"
        return 0
    else
        print_error "无法连接到OpenSearch，请检查服务是否运行"
        return 1
    fi
}

# 重新加载分析器
reload_analyzers() {
    print_info "在OpenSearch中重新加载分析器..."
    
    if [ -n "$INDEX_NAME" ]; then
        print_info "重新加载索引 '$INDEX_NAME' 的分析器..."
        
        # 方法1: 尝试使用Refresh Search Analyzer API (推荐)
        print_info "尝试使用Refresh Search Analyzer API..."
        refresh_response=$(curl -s -w "\n%{http_code}" -X POST "$OPENSEARCH_URL/_plugins/_refresh_search_analyzers/$INDEX_NAME" \
            -k \
            -u "$USERNAME:$PASSWORD" \
            --connect-timeout 10 \
            --max-time 30)
        
        refresh_http_code=$(echo "$refresh_response" | tail -n1)
        if [ "$refresh_http_code" = "200" ]; then
            print_info "Refresh Search Analyzer API 成功！"
            print_info "分析器已重新加载，新的词典配置已生效"
            return 0
        else
            print_warning "Refresh Search Analyzer API 失败，HTTP状态码: $refresh_http_code"
            print_info "尝试备用方法：关闭并重新打开索引..."
        fi
        
        # 方法2: 备用方法 - 关闭并重新打开索引
        print_info "关闭索引..."
        close_response=$(curl -s -w "\n%{http_code}" -X POST "$OPENSEARCH_URL/$INDEX_NAME/_close" \
            -k \
            -u "$USERNAME:$PASSWORD" \
            --connect-timeout 10 \
            --max-time 30)
        
        close_http_code=$(echo "$close_response" | tail -n1)
        if [ "$close_http_code" != "200" ]; then
            print_warning "关闭索引失败，HTTP状态码: $close_http_code"
        else
            print_info "索引关闭成功"
        fi
        
        # 等待一下
        sleep 2
        
        # 重新打开索引
        print_info "重新打开索引..."
        open_response=$(curl -s -w "\n%{http_code}" -X POST "$OPENSEARCH_URL/$INDEX_NAME/_open" \
            -k \
            -u "$USERNAME:$PASSWORD" \
            --connect-timeout 10 \
            --max-time 30)
        
        open_http_code=$(echo "$open_response" | tail -n1)
        if [ "$open_http_code" = "200" ]; then
            print_info "索引重新打开成功！"
            print_info "分析器已重新加载，新的词典配置已生效"
        else
            print_error "重新打开索引失败，HTTP状态码: $open_http_code"
            return 1
        fi
        
    else
        print_info "重新加载所有索引的分析器..."
        
        # 获取所有索引
        indices_response=$(curl -s -k -u "$USERNAME:$PASSWORD" "$OPENSEARCH_URL/_cat/indices?format=json")
        if [ $? -eq 0 ]; then
            indices=$(echo "$indices_response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    indices = [item['index'] for item in data if not item['index'].startswith('.')]
    print(' '.join(indices))
except:
    print('')
" 2>/dev/null)
            
            if [ -n "$indices" ]; then
                print_info "找到索引: $indices"
                for index in $indices; do
                    print_info "处理索引: $index"
                    
                    # 首先尝试Refresh Search Analyzer API
                    refresh_response=$(curl -s -w "\n%{http_code}" -X POST "$OPENSEARCH_URL/_plugins/_refresh_search_analyzers/$index" \
                        -k \
                        -u "$USERNAME:$PASSWORD" \
                        --connect-timeout 10 \
                        --max-time 30)
                    
                    refresh_http_code=$(echo "$refresh_response" | tail -n1)
                    if [ "$refresh_http_code" = "200" ]; then
                        print_info "索引 $index 使用Refresh API成功"
                        continue
                    fi
                    
                    # 如果Refresh API失败，使用关闭/打开方法
                    print_info "索引 $index 使用备用方法..."
                    
                    # 关闭索引
                    curl -s -X POST "$OPENSEARCH_URL/$index/_close" \
                        -k \
                        -u "$USERNAME:$PASSWORD" > /dev/null 2>&1
                    
                    # 等待一下
                    sleep 1
                    
                    # 重新打开索引
                    open_response=$(curl -s -w "\n%{http_code}" -X POST "$OPENSEARCH_URL/$index/_open" \
                        -k \
                        -u "$USERNAME:$PASSWORD" \
                        --connect-timeout 10 \
                        --max-time 30)
                    
                    open_http_code=$(echo "$open_response" | tail -n1)
                    if [ "$open_http_code" = "200" ]; then
                        print_info "索引 $index 重新打开成功"
                    else
                        print_warning "索引 $index 重新打开失败"
                    fi
                done
                print_info "所有索引的分析器已重新加载"
            else
                print_warning "未找到可用的索引"
            fi
        else
            print_error "获取索引列表失败"
            return 1
        fi
    fi
}

# 测试分析器
test_analyzer() {
    if [ -z "$INDEX_NAME" ]; then
        print_warning "未指定索引名称，跳过分析器测试"
        return 0
    fi
    
    print_info "测试分析器是否正常工作..."
    
    test_text="艾达灵族"
    url="$OPENSEARCH_URL/$INDEX_NAME/_analyze"
    
    response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "{\"analyzer\": \"warhammer_search_analyzer\", \"text\": \"$test_text\"}" \
        -k \
        -u "$USERNAME:$PASSWORD" \
        --connect-timeout 10 \
        --max-time 10)
    
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        print_info "分析器测试成功！"
        echo "测试文本: '$test_text'"
        echo "分词结果:"
        echo "$response_body" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'tokens' in data and isinstance(data['tokens'], list):
        tokens = [token.get('token', '') for token in data['tokens'] if isinstance(token, dict)]
        if tokens:
            print('  ' + ', '.join(tokens))
        else:
            print('  无分词结果')
    else:
        print('  响应格式异常')
        print('  原始响应: ' + str(data))
except Exception as e:
    print('  解析失败: ' + str(e))
    print('  原始响应: ' + str(sys.stdin.read()))
" 2>/dev/null || echo "  无法解析响应"
    else
        print_warning "分析器测试失败，HTTP状态码: $http_code"
        echo "错误详情:"
        echo "$response_body"
    fi
}

# 主函数
main() {
    print_info "OpenSearch分析器重新加载脚本启动"
    
    # 检查OpenSearch连接
    if ! check_opensearch; then
        exit 1
    fi
    
    # 重新加载分析器
    if ! reload_analyzers; then
        exit 1
    fi
    
    # 测试分析器
    test_analyzer
    
    print_info "脚本执行完成！"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--index)
            INDEX_NAME="$2"
            shift 2
            ;;
        -a|--all)
            INDEX_NAME=""
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 执行主函数
main
