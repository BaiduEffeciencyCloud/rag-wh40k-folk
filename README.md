
---

```markdown
# 战锤40K 检索系统（RAG Hybrid 检索）

本项目实现了面向战锤40K规则文档的高质量 Hybrid 检索系统，支持 Pinecone 向量数据库，融合 Dense（语义）与 Sparse（BM25）检索，具备业界一流的白名单（保护性短语）召回能力，工程结构清晰，测试闭环完善。

---

## 主要功能

- **Hybrid 检索主流程**：支持 dense+sparse 融合，alpha 权重调节，异常自动降级，参数透传。
- **BM25Manager 白名单保护**：所有白名单短语无论在语料中出现与否，均能被整体召回、分配 idf、参与向量生成。
- **词典/模型自动生成**：支持从指定文件夹批量提取短语、生成 BM25 词典和保护性 userdict。
- **Pinecone 向量数据库对接**：支持 hybrid 检索、dense 检索、sparse 检索，结果格式统一。
- **工程自查与测试驱动开发**：主流程、评估、单元测试、集成测试全链路闭环，边界/异常/兜底逻辑100%一致。
- **详细日志与本地调试**：所有关键分词、vocab、idf、稀疏向量等信息可本地持久化，便于问题追查。

---

## 工程结构

```
.
├── dataupload/              # 主要数据处理与检索逻辑
│   ├── bm25_manager.py      # BM25Manager主类，白名单保护、fit/增量/向量生成
│   ├── gen_userdict.py      # 词典/保护性短语自动生成
│   ├── vector_upload.py     # 文档分块与向量上传
│   └── ...                 
├── searchengine/            # 检索引擎主流程
│   ├── hybrid_search.py     # HybridSearchEngine主流程
│   └── dense_search.py      # Dense检索链路
├── test/                    # 所有测试用例
│   ├── test_bm25_manager.py # BM25Manager单元测试
│   └── searchengine/        # 检索主流程集成测试
├── models/                  # 训练好的BM25模型
├── dict/                    # 词典、保护性短语文件
├── document/                # 生成的文档、报告
├── .env                     # 环境变量配置
├── .gitignore               # 忽略文件配置
└── README.md                # 项目说明文档
```

---

## 快速上手

### 1. 环境准备

- Python 3.8+
- 安装依赖：
  ```bash
  pip3 install -r requirements.txt
  ```

### 2. 生成词典和保护性短语

```bash
python3 dataupload/gen_userdict.py --input_dir test/testdata --output_dir dict
```

### 3. 训练 BM25 模型

```python
from dataupload.bm25_manager import BM25Manager
bm25 = BM25Manager(user_dict_path="dict/all_docs_dict_xxxxxx.txt")
bm25.fit([...])  # 传入所有语料文本
bm25.save_model("models/bm25_model.pkl", "dict/bm25_vocab_xxxxxx.json")
```

### 4. 启动 Hybrid 检索主流程

```python
from searchengine.hybrid_search import HybridSearchEngine
engine = HybridSearchEngine()
results = engine.search("艾达灵族凤凰领主", top_k=5, alpha=0.5)
```

---

## 关键工程规范

- **白名单短语保护**：fit 阶段自动拼接未出现的白名单短语为虚拟文本，确保 BM25Okapi 分配 idf。
- **测试驱动开发**：所有新功能、重构、边界/异常case先写测试，主流程与评估/测试逻辑100%一致。
- **日志与调试**：所有分词、vocab、idf、sparse_vec 关键数据可本地持久化（如 record.log、query_sparse.log）。
- **.gitignore**：所有测试数据、临时文件、模型文件等敏感/大文件已加入忽略，避免泄漏和冗余。

---

## 常见问题

- **query 稀疏向量为空？**
  - 检查 BM25 模型 idf 是否包含目标短语，必要时重新 fit 并保存模型。
- **白名单短语未召回？**
  - 确认保护性词典和 BM25Manager 白名单机制已生效，fit 阶段有虚拟文本兜底。

---

## 贡献与测试

- 测试用例全部位于 test/ 目录，覆盖主流程、边界、异常、兜底等所有场景。
- 推荐每次提交前运行：

  ```bash

  python3 -m unittest discover test
  
  ```

---

## 更新日志

- 2025-07-14：补充 Hybrid 检索、BM25Manager 白名单保护、日志调试、工程结构等说明。

---

> 文档最后更新：2025-07-14
```