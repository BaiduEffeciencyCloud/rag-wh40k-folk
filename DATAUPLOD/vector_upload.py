import os
import re
import math
import asyncio
from typing import List, Dict, Any
import tiktoken
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import pinecone
from tenacity import retry, wait_exponential, stop_after_attempt
import sys
sys.path.append('..')
from config import EMBADDING_MODEL, PINECONE_API_KEY, LLM_MODEL

# Constants or environment config
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
# 使用统一的嵌入模型配置
OPENAI_EMBED_MODEL = EMBADDING_MODEL

# Tokenizer for token-based splitting
TOKEN_ENCODER = tiktoken.encoding_for_model(LLM_MODEL)

# Minimum characters per chunk
MIN_CHARS = 100

class DataVector:
    def __init__(self, index_name: str):
        # Initialize Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index = pinecone.Index(index_name)

        # Set up splitters
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,          # token-based approx -> ~500 tokens
            chunk_overlap=50,
            length_function=lambda txt: len(TOKEN_ENCODER.encode(txt)),
            separators=["\n\n", "\n", " ", "----"]
        )

    def detect_override(self, text: str) -> bool:
        keywords = ["替代", "而非", "允许", "覆盖", "改为", "6寸"]
        return any(kw in text for kw in keywords)

    def classify_chunk(self, text: str) -> str:
        if re.search(r"技能：|当本模型", text):
            return "unit_ability"
        if "部署在" in text and "寸" in text:
            return "deployment_rule"
        if re.search(r"军阵|分队", text):
            return "detachment_rule"
        return "general"

    def estimate_tokens(self, text: str) -> int:
        return len(TOKEN_ENCODER.encode(text))

    def chunk_document(self, text: str, source_id: str = "doc.md") -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        # 1) Header-based splitting
        md_splits = self.header_splitter.split_text(text)
        for split in md_splits:
            headers_meta = {k: v for k, v in split.metadata.items() if k.startswith('h')}
            # 2) Token-aware recursive splitting
            for part in self.char_splitter.split_text(split.page_content):
                content = part.strip()
                if len(content) < MIN_CHARS:
                    continue  # skip too-short chunks
                metadata = {
                    "source_id": f"{source_id}#{len(chunks)}",
                    "headers": headers_meta,
                    "tags": ["override_rule"] if self.detect_override(content) else [],
                    "chunk_type": self.classify_chunk(content),
                    "length_chars": len(content),
                    "length_tokens": self.estimate_tokens(content)
                }
                chunks.append({"content": content, "metadata": metadata})
        return chunks

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    async def _upsert_batch(self, vectors: List[Dict[str, Any]]):
        self.index.upsert(vectors=vectors)

    async def upsert_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Batch upsert chunks into Pinecone index with retry and async concurrency.
        """
        tasks = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = []
            for c in batch:
                emb = await self._get_embedding_async(c["content"])
                vectors.append({
                    "id": c["metadata"]["source_id"],
                    "values": emb,
                    "metadata": c["metadata"]
                })
            # schedule upload
            tasks.append(self._upsert_batch(vectors))
        # run all upserts concurrently
        await asyncio.gather(*tasks)

    async def _get_embedding_async(self, text: str) -> List[float]:
        # Using OpenAI's async client if available
        from openai import OpenAI
        client = OpenAI()
        response = await client.embeddings.acreate(
            model=OPENAI_EMBED_MODEL,
            input=text
        )
        return response.data[0].embedding

# Example usage:
# async def main():
#     dv = DataVector(index_name="wh40k-index")
#     text = open("aeldaricodex.md").read()
#     chunks = dv.chunk_document(text, source_id="aeldaricodex.md")
#     await dv.upsert_chunks(chunks)
#
# if __name__ == "__main__":
#     asyncio.run(main())
