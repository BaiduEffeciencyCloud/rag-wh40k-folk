# 250702-LLM测评Token与价格统计

## 一、测评流程LLM调用次数
- 每条query会调用cot_search主流程：
  1. CoT扩展器（QueryCoTExpander）调用LLM生成子查询（约1次）。
  2. HierarchicalAnswerGenerator调用LLM生成最终答案（约1次）。
- **每个query大约请求LLM 2次**。
- N条query，总LLM请求次数 ≈ 2 × N。

---

## 二、Token消耗估算
- CoT扩展器：每次约500 tokens（prompt+输出）。
- 答案生成：每次约1200 tokens（prompt+输出）。
- **每个query总token ≈ 1700 tokens**。
- N条query总token ≈ 1700 × N。
- 以20条query为例，总token ≈ 34,000 tokens。

---

## 三、OpenAI官方定价（2024年7月）

### gpt-3.5-turbo
- 输入token：$0.0005/1K tokens
- 输出token：$0.0015/1K tokens
- 估算时可按平均$0.001/1K tokens粗略计算

### gpt-4-1106-preview
- 输入token：$0.01/1K tokens
- 输出token：$0.03/1K tokens
- 估算时可按平均$0.02/1K tokens粗略计算

---

## 四、费用估算

### gpt-3.5-turbo
- 总token：34,000 tokens
- 费用 ≈ 34 × $0.001 = **$0.034**（约0.25元人民币）

### gpt-4-1106-preview
- 总token：34,000 tokens
- 费用 ≈ 34 × $0.02 = **$0.68**（约5元人民币）

---

## 五、业务建议
- gpt-3.5-turbo：一次20条query的完整测评，成本极低（几分钱）。
- gpt-4：一次20条query的完整测评，成本约5元人民币。
- 如果测评量大（如1000条query），建议优先用gpt-3.5-turbo，或分批测评，避免高额费用。
- 如需更精确估算，可在代码中统计实际token用量（OpenAI API返回usage字段）。

---

如需自动统计token用量、生成费用明细表，或对不同模型/批量测评做成本分析，可进一步扩展脚本实现。 