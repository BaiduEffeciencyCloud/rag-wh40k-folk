# 环境变量配置模板

## 敏感信息配置（必须设置）

### OpenAI API配置
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Pinecone配置
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_pinecone_index_name_here
```

### Neo4j数据库配置
```bash
# 生产环境Neo4j配置
NEO4J_URI=bolt://your_neo4j_host:7687
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password

# 测试环境Neo4j配置（可选）
NEO4J_TEST_URI=bolt://your_test_neo4j_host:7688
NEO4J_TEST_USERNAME=your_test_neo4j_username
NEO4J_TEST_PASSWORD=your_test_neo4j_password
```

### 模型配置
```bash
# 嵌入模型（可选，有默认值）
EMBEDDING_MODEL=text-embedding-ada-002

# LLM模型（可选，有默认值）
LLM_MODEL=gpt-3.5-turbo

# 重排序模型（可选，有默认值）
RERANK_MODEL=bge-reranker-v2-m3

# Hybrid检索配置（可选，有默认值）
HYBRID_ALPHA=0.3
```

## 非敏感配置（可选，有默认值）

### 系统参数配置
```bash
# 默认温度
DEFAULT_TEMPERATURE=0.7

# 默认返回结果数
DEFAULT_TOP_K=15

# 查询超时时间（秒）
QUERY_TIMEOUT=30

# 答案生成温度
ANSWER_GENERATION_TEMPERATURE=0.3

# 答案生成最大token数
ANSWER_MAX_TOKENS=2000

# 去重阈值
DEDUPLICATION_THRESHOLD=0.8

# 默认最大并发工作线程数
DEFAULT_MAX_WORKERS=4

# 答案生成时使用的最大上下文结果数
MAX_CONTEXT_RESULTS=10
```

### 日志配置
```bash
# 日志格式
LOG_FORMAT=%(asctime)s - %(levelname)s - %(message)s

# 日志文件
LOG_FILE=app.log

# 日志级别
LOG_LEVEL=INFO
```

### 应用配置
```bash
# 应用标题
APP_TITLE=战锤40K规则助手

# 应用图标
APP_ICON=⚔️

# 应用头部
APP_HEADER=
```

## 使用说明

1. **复制模板**：将上述配置复制到项目根目录的 `.env` 文件中
2. **填写敏感信息**：替换所有 `your_*_here` 为实际的值
3. **保护文件**：确保 `.env` 文件不会被提交到版本控制系统
4. **验证配置**：运行应用时检查日志，确保所有必需的配置都已正确设置

## 安全注意事项

- **不要提交 `.env` 文件**：确保 `.env` 文件在 `.gitignore` 中
- **定期轮换密钥**：定期更新API密钥和密码
- **最小权限原则**：使用具有最小必要权限的账户
- **环境隔离**：生产环境和测试环境使用不同的配置 