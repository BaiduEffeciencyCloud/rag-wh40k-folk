import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "请先在环境变量中配置OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY

MODEL = "text-embedding-3-large"
TEXT = "测试OpenAI embedding维度"

def get_embedding(text, model, dimensions=None):
    params = {
        "input": text,
        "model": model
    }
    if dimensions:
        params["dimensions"] = dimensions
    response = openai.embeddings.create(**params)
    return response.data[0].embedding

if __name__ == "__main__":
    print("--- 测试 3072 维 ---")
    emb_3072 = get_embedding(TEXT, MODEL, 3072)
    print(f"3072维向量长度: {len(emb_3072)}")

    print("--- 测试 1024 维 ---")
    emb_1024 = get_embedding(TEXT, MODEL, 1024)
    print(f"1024维向量长度: {len(emb_1024)}") 