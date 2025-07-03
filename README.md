# 查询处理系统

## 项目概述
本项目实现了一个WH40K（桌游）规则和数据查询处理系统，目前只实现了query扩展模式

## 功能特点
- **查询扩展**：生成查询变体，选择最贴合用户意图的查询生成完整答案。

## 项目结构
- `app.py`：主应用程序，处理用户查询并返回结果。
- `vector_search.py`：实现向量搜索功能。
- `query_expander.py`：实现查询扩展功能。
- `config.py`：配置文件，包含API密钥和模型配置。

## 使用方法
1. 确保已安装所有依赖项：
   ```bash
   pip3 install -r requirements.txt
   ```

2. 配置环境变量：
   - 复制 `document/env_template.md` 中的配置模板到项目根目录的 `.env` 文件
   - 填写所有必需的敏感信息（API密钥、数据库连接信息等）
   - 确保 `.env` 文件不会被提交到版本控制系统

3. 验证配置：
   ```bash
   python3 test/test_config.py
   ```

4. 运行应用程序：
   ```bash
   python3 app.py
   ```

4. 使用命令行参数选择查询模式：
   ```bash
   python3 app.py --mode vector  # 基础向量检索模式，目前废弃
   python3 app.py --mode expand  # 查询扩展模式，根据用户query扩展出3个query,选择一个最合适的
   python3 app.py --mode enhance  # 增强RAG FUSION模式，目前废弃
   ```

## 配置说明

### 必需配置（敏感信息）
所有敏感信息必须通过环境变量配置，不支持默认值：

- `OPENAI_API_KEY`：OpenAI API密钥
- `PINECONE_API_KEY`：Pinecone API密钥
- `PINECONE_INDEX`：Pinecone索引名称
- `NEO4J_URI`：Neo4j数据库连接URI
- `NEO4J_USERNAME`：Neo4j用户名
- `NEO4J_PASSWORD`：Neo4j密码

### 可选配置（有默认值）
- `EMBEDDING_MODEL`：嵌入模型名称（默认：text-embedding-ada-002）
- `LLM_MODEL`：LLM模型名称（默认：gpt-3.5-turbo）
- `RERANK_MODEL`：重排序模型名称（默认：bge-reranker-v2-m3）
- `DEFAULT_TEMPERATURE`：默认温度参数（默认：0.7）
- `DEFAULT_TOP_K`：默认返回结果数（默认：15）
- `QUERY_TIMEOUT`：查询超时时间（默认：30秒）

详细配置说明请参考 `document/env_template.md`

## 日志配置
- 日志级别设置为 `INFO`，格式为 `%(asctime)s - %(name)s - %(levelname)s - %(message)s`。
## 可使用的桌游文档
- 总规则中文版: https://assets.warhammer-community.com/warhammer40000_core&key_corerules_chn_24.09-czgt3rq37v.pdf
- 总规FAQ：https://assets.warhammer-community.com/chi_wh40k_core&key_core_rules_updates_commentary_dec2024-r4d3ehrv6m-lbyydmwvw2.pdf
- 赛季补丁: https://assets.warhammer-community.com/chi_04-06_warhammer_40000_core_rules_balance_dataslate-mcbpvplq49-fd0mqwnlxf.pdf 

## 注意事项
- **安全第一**：确保 `.env` 文件包含所有必需的敏感信息，且不会被提交到版本控制系统
- **配置验证**：运行前使用 `python3 test/test_config.py` 验证配置是否正确
- **数据库连接**：确保Neo4j数据库和Pinecone索引已创建并可用
- **API密钥**：确保所有API密钥有效且有足够的配额

## 贡献
欢迎提交问题和改进建议！ 
