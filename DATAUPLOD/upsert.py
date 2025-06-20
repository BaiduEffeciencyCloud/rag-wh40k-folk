import asyncio
from pinecone import Pinecone
from data_vector import DataVector
from langchain_openai import OpenAIEmbeddings
import os
from datetime import datetime
import argparse
from dotenv import load_dotenv
from data_vector_v2 import DocumentChunker
from unify import TextUnifier
import json

# 加载.env文件
load_dotenv('../.env')

# 从.env文件获取API密钥
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
# 检查API密钥是否存在
if not PINECONE_API_KEY:
    raise ValueError("未找到PINECONE_API_KEY，请在.env文件中设置")
if not OPENAI_API_KEY:
    raise ValueError("未找到OPENAI_API_KEY，请在.env文件中设置")

PINECONE_ENVIRONMENT = "aped-4627-b74a"  # 修改为你的环境

# 初始化 Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

async def upsert_to_pinecone(texts: list, faction: str, source_file: str):
    """
    使用新版 Pinecone SDK 上传向量
    """
    # 初始化 OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # 获取当前时间
    current_time = datetime.now().isoformat()
    
    # 准备向量数据
    vectors = []
    for i, text in enumerate(texts):
        # 获取文本的向量表示（新版 OpenAIEmbeddings 仍支持异步）
        embedding = await embeddings.aembed_query(text)
        # 构建丰富的 metadata
        metadata = {
            "text": text,
            "source_file": source_file,
            "chunk_index": i,
            "faction": faction,
            "total_chunks": len(texts),
            "timestamp": current_time,
            "content_type": "markdown",
            "chunk_size": len(text),
            "language": "zh"
        }
        
        vectors.append({
            "id": f"doc_{source_file}_{i}_{current_time}",
            "values": embedding,
            "metadata": metadata
        })
    
    # 使用新版 Pinecone SDK 上传向量（每批最多100个）
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"成功上传 {len(vectors)} 个文档到 Pinecone！")
    stats = index.describe_index_stats()
    print("索引统计信息：", stats)

def remove_text_from_metadata(chunk, source_file, faction, chunk_index, timestamp):
    metadata = {k: v for k, v in chunk.items() if k != 'text'}
    metadata['source_file'] = source_file
    metadata['faction'] = faction
    metadata['chunk_index'] = chunk_index
    metadata['timestamp'] = timestamp
    return metadata

async def up_to_pinecone(chunks: list, faction: str, source_file: str = "content.md"):
    """
    将切片（每个为data_vector_v2.py中chunk_text生成的metadata字典）向量化并上传到Pinecone。
    每个切片的metadata直接使用chunk本身（去除text字段，text字段作为embedding输入）。
    新增：在上传前进行统一的token化与归一化处理。
    """
    # 初始化文本统一化处理器
    unifier = TextUnifier()
    
    # 对chunks进行统一化处理
    print("正在进行文本统一化处理...")
    unified_chunks = unifier.process_chunks(chunks)
    print(f"统一化处理完成，共处理 {len(unified_chunks)} 个chunks")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    current_time = datetime.now().isoformat()
    
    vectors = []
    for i, chunk in enumerate(unified_chunks):
        # 使用统一化后的文本进行向量化
        text = chunk.get('text', '')
        metadata = remove_text_from_metadata(chunk, source_file, faction, i, current_time)
        embedding = await embeddings.aembed_query(text)
        vectors.append({
            "id": f"doc_{os.path.basename(source_file)}_{i}_{current_time}",
            "values": embedding,
            "metadata": metadata
        })
    
    # 使用新版 Pinecone SDK 上传向量（每批最多100个）
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    
    print(f"成功上传 {len(vectors)} 个文档到 Pinecone！")
    stats = index.describe_index_stats()
    print("索引统计信息：", stats)

def load_special_patterns(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def main(source_file: str, faction: str, chunk_size: int = 512, chunk_overlap: int = 50, output_file: str = "result.txt", chunker_type: str = "v1"):
    """
    主函数，支持选择切片方法
    
    Args:
        source_file (str): 源文件路径
        faction (str): 派系名称
        chunk_size (int): 文本块大小
        chunk_overlap (int): 文本块重叠大小
        output_file (str): 输出文件名
        chunker_type (str): 切片方法选择，'v1' 使用 DataVector，'v2' 使用 DocumentChunker
    """
    chunks = []
    
    if chunker_type == "v1":
        # 使用原有的 DataVector 切片
        vectorizer = DataVector()
        chunks = vectorizer.process_markdown(source_file, faction)
        print(f"使用 DataVector(v1) 切片，共生成 {len(chunks)} 个块")
        # 保存分割结果到文件
        recursive_to_txt([chunk['content'] for chunk in chunks], output_file)
        # 上传到 Pinecone
        await upsert_to_pinecone([chunk['content'] for chunk in chunks], faction, source_file)
    
    elif chunker_type == "v2":
        # 读取special_patterns配置
        special_patterns_path = os.path.join(os.path.dirname(__file__), 'special_patterns.json')
        special_patterns = load_special_patterns(special_patterns_path)
        # 读取文件内容
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建切片器实例
        chunker = DocumentChunker(
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
            special_patterns=special_patterns
        )
        
        # 添加额外metadata
        extra_metadata = {
            'faction': faction,
            'source_file': os.path.basename(source_file)
        }
        
        # 执行切片
        chunks = chunker.chunk_text(content, extra_metadata)
        print(f"使用 DocumentChunker(v2) 切片，共生成 {len(chunks)} 个块")
        # 保存分割结果到文件
        recursive_to_txt(chunks, output_file)
        # 上传到 Pinecone
        await up_to_pinecone(chunks, faction, source_file)
    
    else:
        raise ValueError(f"不支持的切片方法: {chunker_type}，请使用 'v1' 或 'v2'")

def recursive_to_txt(chunks: list, output_file: str):
    """
    将分割后的文本块保存为 TXT 文件，使用 ----{chunk number}---- 作为分隔符
    支持v1格式 {'content': text} 和 v2格式 {'text': text, ...}
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"总块数: {len(chunks)}\n\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"----{i}----\n")
            if isinstance(chunk, dict):
                # v1格式使用'content'字段
                if 'content' in chunk:
                    f.write(chunk['content'].strip())
                # v2格式使用'text'字段
                elif 'text' in chunk:
                    f.write(chunk['text'].strip())
                else:
                    f.write(str(chunk).strip())
            else:
                f.write(str(chunk).strip())
            f.write("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理文档并上传到 Pinecone')
    parser.add_argument('--source', type=str, default='content.md', help='源文件路径')
    parser.add_argument('--faction', type=str, required=True, help='派系名称')
    parser.add_argument('--chunk-size', type=int, default=512, help='文本块大小')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='文本块重叠大小')
    parser.add_argument('--output', type=str, default='result.txt', help='输出文件名')
    parser.add_argument('--chunker', type=str, choices=['v1', 'v2'], default='v1', help='切片方法选择：v1=DataVector, v2=DocumentChunker')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        source_file=args.source,
        faction=args.faction,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_file=args.output,
        chunker_type=args.chunker
    ))
