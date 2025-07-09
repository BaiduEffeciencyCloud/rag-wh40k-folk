# 战锤40K 规则文档向量化项目

这个项目用于将战锤40K（Warhammer 40K）的规则文档进行向量化处理，并存储到 Pinecone 向量数据库中，以便后续进行语义搜索和问答。

## 功能特点

- 使用 LangChain 进行智能文本分割
- 支持 Markdown 格式文档处理
- 自动提取文档标题结构
- 使用 OpenAI Embeddings 进行文本向量化
- 支持异步操作，提高处理效率
- 可配置的文本块大小和重叠度
- 支持 Pinecone 向量数据库存储

## 环境要求

- Python 3.8+
- OpenAI API Key
- Pinecone API Key

## 安装依赖

```bash
pip install pinecone-client langchain openai
```

## 使用方法

### 1. 配置 API Keys

在 `upsert.py` 中设置你的 API Keys：

```python
PINECONE_API_KEY = "你的 Pinecone API Key"
OPENAI_API_KEY = "你的 OpenAI API Key"
```

### 2. 上传文档到 Pinecone

```bash
python upsert.py --source content.md --faction "派系名称" --chunk-size 1500 --chunk-overlap 150 --output result.txt
```

参数说明：
- `--source`: 源文件路径（默认：content.md）
- `--faction`: 派系名称（必需）
- `--chunk-size`: 文本块大小（默认：1500）
- `--chunk-overlap`: 文本块重叠大小（默认：150）
- `--output`: 输出文件名（默认：result.txt）

### 3. 文本分割

项目使用 LangChain 的 `MarkdownHeaderTextSplitter` 和 `RecursiveCharacterTextSplitter` 进行智能文本分割：

- 首先按 Markdown 标题结构分割
- 然后对每个部分进行递归分割
- 保持文本的语义完整性
- 支持自定义分割参数

## 项目结构

```
.
├── README.md
├── upsert.py              # 主程序：处理文档并上传到 Pinecone
├── langchain_splitter.py  # 文本分割工具
└── content.md            # 示例文档
```

## 注意事项

1. 确保有足够的 OpenAI API 额度
2. Pinecone 索引需要提前创建
3. 文档最好使用 Markdown 格式，以保持结构完整性
4. 建议先用小文件测试，确认效果后再处理大文件

## 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 许可证

MIT License 