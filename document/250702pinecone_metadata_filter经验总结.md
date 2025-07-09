# Pinecone Metadata Filter 经验总结

## 1. 支持的Filter操作符
Pinecone 支持在向量检索时对 metadata 字段进行过滤，常用操作符如下：

- `$eq`：等于。例如：`{"faction": {"$eq": "aeldari"}}`
- `$ne`：不等于。例如：`{"faction": {"$ne": "aeldari"}}`
- `$in`：包含于列表。例如：`{"faction": {"$in": ["aeldari", "corefaq"]}}`
- `$nin`：不包含于列表。例如：`{"faction": {"$nin": ["aeldari", "corefaq"]}}`
- `$gt`/`$gte`/`$lt`/`$lte`：大于/大于等于/小于/小于等于（用于数值字段）

## 2. 不支持的操作符
- **不支持 `$contains`**：不能用于字符串的"包含"过滤。
- **不支持正则表达式**：不能用正则做模糊匹配。
- **不支持字符串的模糊/部分匹配**：只能精确匹配或列表包含。

## 3. 常见设计误区
- 不能用`{"hierarchy": {"$contains": "猎鹰坦克"}}`来查找包含某实体的切片，会报错：
  > {"code":3,"message":"$contains is not a valid operator","details":[]}
- 不能用`{"text": {"$contains": "xxx"}}`做全文模糊检索。

## 4. 推荐的最佳实践
- **实体/关键词字段建议用list存储**：如`entities: ["猎鹰坦克", "火力支援"]`，检索时用`{"entities": {"$in": ["猎鹰坦克"]}}`。
- **层级/标签等结构化信息建议拆分为多个字段**，避免用长字符串做filter。
- **数值字段可用范围操作符**，如`{"token_count": {"$gt": 100}}`。

## 5. 典型用法示例
```python
# 检索faction为aeldari的所有切片
{"faction": {"$eq": "aeldari"}}

# 检索entities字段包含"猎鹰坦克"的切片
{"entities": {"$in": ["猎鹰坦克"]}}

# 检索token_count大于200的切片
{"token_count": {"$gt": 200}}
```

## 6. 经验教训
- 设计metadata schema时要考虑检索需求，提前规划好哪些字段需要filter、哪些需要支持多值。
- 不要指望Pinecone支持字符串模糊/包含/正则过滤，所有"包含"需求都应转为list+`$in`。
- filter写法不对时Pinecone会直接报错，务必查阅官方文档和报错信息。

---

**结论：**
Pinecone的metadata filter非常高效，但只支持结构化、精确、列表式的过滤。设计schema时要为后续检索留好扩展空间，避免后期大规模重构。 