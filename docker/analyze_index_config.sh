#!/bin/bash

# OpenSearch索引分析器配置分析脚本
# 用于分析索引的分析器配置和测试分词效果

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
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_section() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_subsection() {
    echo -e "${CYAN}--- $1 ---${NC}"
}

print_detail() {
    echo -e "${PURPLE}$1${NC}"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项] [索引名称]"
    echo ""
    echo "选项:"
    echo "  -i, --index INDEX_NAME    指定要分析的索引名称 (默认: 40ktest)"
    echo "  -t, --test                测试分析器分词效果"
    echo "  -c, --config              只显示配置信息"
    echo "  -h, --help                显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                        # 分析默认索引"
    echo "  $0 -i my_index           # 分析指定索引"
    echo "  $0 -t                    # 分析配置并测试分词"
    echo "  $0 -c                    # 只显示配置信息"
}

# 检查OpenSearch连接
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

# 检查索引是否存在
check_index_exists() {
    print_info "检查索引是否存在..."
    
    response=$(curl -s -w "\n%{http_code}" -k -u "$USERNAME:$PASSWORD" \
        --connect-timeout 10 \
        --max-time 15 \
        "$OPENSEARCH_URL/$INDEX_NAME")
    
    if [ $? -ne 0 ]; then
        print_error "连接索引失败，可能是网络问题或索引不存在"
        return 1
    fi
    
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "404" ]; then
        print_error "索引 '$INDEX_NAME' 不存在"
        return 1
    elif [ "$http_code" = "200" ]; then
        print_info "索引 '$INDEX_NAME' 存在"
        return 0
    else
        print_warning "索引状态不明确，HTTP状态码: $http_code"
        print_info "尝试继续分析..."
        return 0
    fi
}

# 获取索引设置
get_index_settings() {
    print_section "索引设置分析"
    
    response=$(curl -s -k -u "$USERNAME:$PASSWORD" "$OPENSEARCH_URL/$INDEX_NAME/_settings")
    
    if [ $? -eq 0 ]; then
        print_info "成功获取索引设置"
        
        # 解析并显示分析器配置
        echo "$response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if '$INDEX_NAME' in data and 'settings' in data['$INDEX_NAME']:
        settings = data['$INDEX_NAME']['settings']
        
        print('索引名称: $INDEX_NAME')
        print('分片数:', settings.get('index', {}).get('number_of_shards', 'N/A'))
        print('副本数:', settings.get('index', {}).get('number_of_replicas', 'N/A'))
        print('KNN支持:', settings.get('index', {}).get('knn', 'N/A'))
        
        if 'index' in settings and 'analysis' in settings['index']:
            analysis = settings['index']['analysis']
            print('\\n分析器配置:')
            print('==========')
            
            if 'analyzer' in analysis:
                print('\\nAnalyzers:')
                for name, config in analysis['analyzer'].items():
                    print(f'  {name}:')
                    print(f'    类型: {config.get(\"type\", \"N/A\")}')
                    print(f'    Tokenizer: {config.get(\"tokenizer\", \"N/A\")}')
                    if 'filter' in config:
                        print(f'    Filters: {config[\"filter\"]}')
                    print()
            
            if 'filter' in analysis:
                print('Filters:')
                for name, config in analysis['filter'].items():
                    print(f'  {name}:')
                    print(f'    类型: {config.get(\"type\", \"N/A\")}')
                    if 'keep_words_path' in config:
                        print(f'    词典路径: {config[\"keep_words_path\"]}')
                    if 'synonyms_path' in config:
                        print(f'    同义词路径: {config[\"synonyms_path\"]}')
                    if 'updateable' in config:
                        print(f'    可更新: {config[\"updateable\"]}')
                    print()
        else:
            print('\\n未找到分析器配置')
    else:
        print('索引不存在或配置异常')
except Exception as e:
    print('解析失败: ' + str(e))
" 2>/dev/null || echo "  无法解析索引设置"
    else
        print_error "获取索引设置失败"
    fi
}

# 获取索引映射
get_index_mappings() {
    print_section "索引映射分析"
    
    response=$(curl -s -k -u "$USERNAME:$PASSWORD" "$OPENSEARCH_URL/$INDEX_NAME/_mappings")
    
    if [ $? -eq 0 ]; then
        print_info "成功获取索引映射"
        
        # 解析并显示映射配置
        echo "$response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if '$INDEX_NAME' in data and 'mappings' in data['$INDEX_NAME']:
        mappings = data['$INDEX_NAME']['mappings']
        
        if 'properties' in mappings:
            print('字段映射:')
            print('==========')
            
            def analyze_properties(props, indent=0):
                for field_name, field_config in props.items():
                    field_type = field_config.get('type', 'N/A')
                    analyzer = field_config.get('analyzer', 'N/A')
                    search_analyzer = field_config.get('search_analyzer', 'N/A')
                    
                    print('  ' * indent + f'{field_name}:')
                    print('  ' * (indent + 1) + f'类型: {field_type}')
                    
                    if analyzer != 'N/A':
                        print('  ' * (indent + 1) + f'索引分析器: {analyzer}')
                    if search_analyzer != 'N/A':
                        print('  ' * (indent + 1) + f'搜索分析器: {search_analyzer}')
                    
                    # 递归分析嵌套字段
                    if 'properties' in field_config:
                        print('  ' * (indent + 1) + '嵌套字段:')
                        analyze_properties(field_config['properties'], indent + 2)
                    
                    print()
            
            analyze_properties(mappings['properties'])
        else:
            print('未找到字段映射')
    else:
        print('索引不存在或映射异常')
except Exception as e:
    print('解析失败: ' + str(e))
" 2>/dev/null || echo "  无法解析索引映射"
    else
        print_error "获取索引映射失败"
    fi
}

# 测试分析器分词
test_analyzers() {
    print_section "分析器分词测试"
    
    local query_text="艾达灵族有多少支派武士"
    print_info "测试查询: \"$query_text\""
    echo ""
    
    # 获取索引中定义的所有分析器
    response=$(curl -s -k -u "$USERNAME:$PASSWORD" "$OPENSEARCH_URL/$INDEX_NAME/_settings")
    
    if [ $? -eq 0 ]; then
        analyzers=$(echo "$response" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if '$INDEX_NAME' in data and 'settings' in data['$INDEX_NAME']:
        settings = data['$INDEX_NAME']['settings']
        if 'index' in settings and 'analysis' in settings['index']:
            analysis = settings['index']['analysis']
            if 'analyzer' in analysis:
                analyzer_names = list(analysis['analyzer'].keys())
                print(' '.join(analyzer_names))
except:
    pass
" 2>/dev/null)
        
        if [ -n "$analyzers" ]; then
            print_info "发现分析器: $analyzers"
            echo ""
            
            for analyzer in $analyzers; do
                print_subsection "测试分析器: $analyzer"
                test_single_analyzer "$query_text" "$analyzer"
                echo ""
            done
        else
            print_warning "未找到分析器配置"
        fi
    else
        print_error "无法获取分析器配置"
    fi
}

# 测试单个分析器
test_single_analyzer() {
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
        print_info "✅ 分析器测试成功"
        
        # 解析并显示分词结果
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
        print('  分词结果: ' + ', '.join(tokens))
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
        print_error "❌ 分析器测试失败，HTTP状态码: $http_code"
        echo "错误详情:"
        echo "$response_body"
    fi
}

# 主函数
main() {
    print_info "OpenSearch索引分析器配置分析脚本启动"
    echo ""
    
    # 检查OpenSearch连接
    if ! check_opensearch; then
        exit 1
    fi
    
    # 检查索引是否存在
    if ! check_index_exists; then
        exit 1
    fi
    
    # 分析索引配置
    get_index_settings
    echo ""
    
    get_index_mappings
    echo ""
    
    # 如果指定了测试选项，则测试分析器
    if [ "$TEST_ANALYZERS" = "true" ]; then
        test_analyzers
    fi
    
    print_info "分析完成！"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--index)
            INDEX_NAME="$2"
            shift 2
            ;;
        -t|--test)
            TEST_ANALYZERS="true"
            shift
            ;;
        -c|--config)
            TEST_ANALYZERS="false"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
        *)
            # 非选项参数，作为索引名称
            INDEX_NAME="$1"
            shift
            ;;
    esac
done

# 执行主函数
main
