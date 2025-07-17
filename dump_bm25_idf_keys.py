import pickle

# 加载 BM25 模型
with open('models/bm25_model.pkl', 'rb') as f:
    bm25_model = pickle.load(f)

# 提取 idf keys
idf_keys = bm25_model.idf.keys()

# 保存到 modelkeys.log
with open('modelkeys.log', 'w', encoding='utf-8') as f:
    for key in idf_keys:
        f.write(str(key) + '\n')

print(f"已保存 {len(list(idf_keys))} 个 idf key 到 modelkeys.log") 