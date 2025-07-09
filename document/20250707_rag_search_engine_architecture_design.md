### 2.x 流程调度器（Scheduler/Controller）设计

#### 1. 设计定位与职责
- 作为前端请求的统一入口，负责参数解析、流程分发、模块调度和最终结果汇总。
- 解耦前端与后端各检索/处理模块，便于灵活扩展和策略切换。
- 支持多种query处理、检索、融合策略的动态配置。

#### 2. 接口参数设计
| 参数名   | 类型    | 说明                                                                                   | 取值示例/约定 |
|----------|---------|----------------------------------------------------------------------------------------|---------------|
| --query  | str     | 用户检索关键词/问题                                                                    | "阿巴顿的黑色远征" |
| --qp     | str     | query processor类型：<br>origin（原始query）、expand（扩写）、cot（思维链）、if-kg（是否抽取KG实体） | origin/expand/cot/if-kg |
| --se     | str     | search engine类型：<br>dense（向量）、sparse（稀疏）、hybrid（多路混合，含KG）           | dense/sparse/hybrid |
| --other  | dict    | 预留参数，便于未来扩展（如答案融合策略、召回数量等）                                    | -             |

> 说明：  
> - --qp 支持多选（如"expand+if-kg"），可用逗号分隔或布尔参数。  
> - --se 决定主检索路径，hybrid时自动调度多路召回与融合。

#### 3. 典型流程伪代码
```python
def scheduler(request):
    # 1. 解析参数
    query = request['query']
    qp_type = request.get('qp', 'origin')
    se_type = request.get('se', 'hybrid')
    # 2. Query处理
    if qp_type == 'origin':
        processed_query = query
    elif qp_type == 'expand':
        processed_query = query_expansion(query)
    elif qp_type == 'cot':
        processed_query = chain_of_thought(query)
    # KG实体识别
    if 'if-kg' in qp_type:
        entities, relations = extract_entities_relations(processed_query)
    else:
        entities, relations = None, None
    # 3. 检索调度
    if se_type == 'dense':
        results = dense_search(processed_query)
    elif se_type == 'sparse':
        results = sparse_search(processed_query)
    elif se_type == 'hybrid':
        results = []
        if entities:
            results += kg_search(entities, relations)
        results += dense_search(processed_query)
        results += sparse_search(processed_query)
        results = aggregate_and_dedup(results)
    # 4. 答案生成（融合策略可扩展）
    answer = answer_generate(results)
    return answer
```

#### 4. 可扩展性建议
- **参数可扩展**：预留future参数（如答案融合策略、召回数量、评估开关等）。
- **模块可插拔**：query processor、search engine、KG召回、答案融合均可通过参数动态切换。
- **流程可追踪**：每步处理结果可选输出，便于调试和评估。
- **接口规范**：建议采用RESTful或gRPC接口，便于前后端解耦和多语言支持。

#### 5. 示例请求
```json
{
  "query": "阿巴顿的黑色远征有哪些关键事件？",
  "qp": "expand,if-kg",
  "se": "hybrid"
}
``` 