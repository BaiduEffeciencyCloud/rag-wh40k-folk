# 多路融合策略研究（250707）

## 1. 简单合并与去重（Union & Deduplication）
- **做法**：将各路召回（dense、sparse、KG等）结果简单合并，去重后作为候选集。
- **适用场景**：召回规模不大、对融合策略要求不高时。
- **优点**：实现简单，易于工程落地。
- **缺点**：无法充分利用各路召回的分数和特征，相关性排序不佳。

---

## 2. 线性加权融合（Weighted Sum/Linear Fusion）
- **做法**：为每一路召回结果分配权重（如dense分数×α + sparse分数×β + KG分数×γ），对候选文档加权打分，按总分排序。
- **主流实现**：百度、知乎等大厂RAG系统常用，权重可人工设定或通过自动调参获得。
- **优点**：可控性强，易于调优。
- **缺点**：权重设置依赖经验或调参，难以适应复杂场景。

---

## 3. 学习排序（Learning to Rank, LTR）
- **做法**：用机器学习模型（如LightGBM、XGBoost、神经网络等）对多路召回的特征（如各路分数、实体匹配、上下文相关性等）进行融合排序。
- **主流实现**：Google、微软、阿里等大厂检索系统广泛采用。
- **流程**：
  1. 构建训练集（query-文档对及其多路特征、人工标注的相关性标签）
  2. 训练LTR模型
  3. 线上融合时用模型对候选集打分排序
- **优点**：融合能力强，能自动学习最优排序策略。
- **缺点**：需要标注数据和特征工程，工程复杂度较高。

---

## 4. 多路召回+重排序（Rerank）
- **做法**：先用多路召回合并候选集，再用更强的模型（如cross-encoder、BERT/LLM reranker）对候选集进行重排序。
- **主流实现**：OpenAI、百度、知乎等RAG系统常用"bi-encoder召回 + cross-encoder重排"。
- **优点**：最终排序质量高，能充分利用上下文和语义信息。
- **缺点**：重排序模型推理成本高，适合候选集较小场景。

---

## 5. KG特征融合与证据链增强
- **做法**：将KG召回的结构化特征（如实体链、关系链、节点度等）作为融合排序的特征，或直接作为答案生成的"证据链"输入。
- **主流实现**：微软、阿里等在复杂问答和推理场景下常用。
- **优点**：提升可解释性和复杂推理能力。
- **缺点**：特征设计和融合策略需结合具体业务。

---

## 6. LLM/Agentic融合（答案生成前的智能聚合）
- **做法**：将多路召回结果（文本+KG片段）全部输入LLM/Agent，利用大模型的理解和推理能力自动聚合、筛选、重组信息，生成最终答案。
- **主流实现**：OpenAI、百度等RAG系统在复杂问答、COT推理场景下常用。
- **优点**：融合灵活，能处理复杂信息整合和推理。
- **缺点**：成本高，结果可控性弱，需配合评估体系。

---

## 7. 典型融合流程伪代码
```python
# 1. 简单合并去重
candidates = list(set(dense_results + sparse_results + kg_results))

# 2. 线性加权融合
def weighted_score(doc):
    return 0.5 * dense_score(doc) + 0.3 * sparse_score(doc) + 0.2 * kg_score(doc)
candidates.sort(key=weighted_score, reverse=True)

# 3. LTR融合
features = extract_features(candidates)
scores = ltr_model.predict(features)
candidates = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]

# 4. Cross-encoder重排序
rerank_scores = cross_encoder.predict(query, candidates)
final_candidates = [doc for _, doc in sorted(zip(rerank_scores, candidates), reverse=True)]

# 5. LLM/Agentic融合
final_answer = llm_generate_answer(query, final_candidates, kg_evidence)
```

---

## 8. 业界参考
- 百度/知乎/阿里：多路召回+LTR融合+cross-encoder重排
- OpenAI：多路召回+LLM聚合+人工评测
- 微软：KG证据链+多路召回+可解释性增强 