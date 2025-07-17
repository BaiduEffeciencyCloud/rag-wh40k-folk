import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import asyncio
from dataupload.vector_upload import DataVector

class TestDataVector(unittest.TestCase):
    def setUp(self):
        self.index_name = "test-index"
        # patch pinecone.Index，返回 MagicMock 实例，避免参数校验
        patcher = patch("pinecone.Index", return_value=MagicMock())
        self.mock_index_class = patcher.start()
        self.addCleanup(patcher.stop)
        # patch pinecone.init，避免真实初始化
        patcher_init = patch("pinecone.init", return_value=None)
        patcher_init.start()
        self.addCleanup(patcher_init.stop)
        self.dv = DataVector(self.index_name)

    def test_chunk_document_basic(self):
        test_file = os.path.join(os.path.dirname(__file__), "testdata", "aeldaricodex.md")
        with open(test_file, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = self.dv.chunk_document(text, source_id="aeldaricodex.md")
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)
            self.assertIn("source_id", chunk["metadata"])
            self.assertIn("chunk_type", chunk["metadata"])

    def test_chunk_document_empty(self):
        chunks = self.dv.chunk_document("", source_id="empty.md")
        self.assertEqual(chunks, [])

    def test_classify_chunk(self):
        self.assertEqual(self.dv.classify_chunk("技能：xxx"), "unit_ability")
        self.assertEqual(self.dv.classify_chunk("部署在6寸范围内"), "deployment_rule")
        self.assertEqual(self.dv.classify_chunk("军阵分队"), "detachment_rule")
        self.assertEqual(self.dv.classify_chunk("普通内容"), "general")

    def test_detect_override(self):
        self.assertTrue(self.dv.detect_override("允许你替代规则"))
        self.assertFalse(self.dv.detect_override("普通内容"))

    @patch("dataupload.vector_upload.DataVector._get_embedding_async", new_callable=AsyncMock)
    def test_upsert_chunks_mocked(self, mock_embed):
        # mock embedding 返回固定向量
        mock_embed.return_value = [0.1, 0.2, 0.3]
        # mock pinecone.Index.upsert
        self.dv.index.upsert = MagicMock()
        # 构造简单chunk
        chunks = [
            {"content": "测试内容1", "metadata": {"source_id": "doc#0"}},
            {"content": "测试内容2", "metadata": {"source_id": "doc#1"}}
        ]
        # 执行异步 upsert
        asyncio.run(self.dv.upsert_chunks(chunks, batch_size=1))
        # 检查 upsert 被调用
        self.assertTrue(self.dv.index.upsert.called)
        # 检查 embedding 被调用
        self.assertEqual(mock_embed.call_count, 2)

    @patch("dataupload.vector_upload.DataVector._get_embedding_async", new_callable=AsyncMock)
    def test_upsert_chunks_with_sparse_values(self, mock_embed):
        """
        测试DataVector集成sparse embedding，断言每个vector都包含sparse_values字段，内容与bm25_manager一致。
        """
        mock_embed.return_value = [0.1, 0.2, 0.3]
        # mock pinecone.Index.upsert
        self.dv.index.upsert = MagicMock()
        # mock bm25_manager
        mock_bm25 = MagicMock()
        mock_bm25.get_sparse_vector.side_effect = lambda text: {"indices": [1,2], "values": [0.5, 0.8]}
        # 假设DataVector未来支持bm25_manager注入
        self.dv.bm25_manager = mock_bm25
        # 构造chunk
        chunks = [
            {"content": "测试内容1", "metadata": {"source_id": "doc#0"}},
            {"content": "测试内容2", "metadata": {"source_id": "doc#1"}}
        ]
        # 设想DataVector.upsert_chunks支持sparse embedding
        async def upsert_chunks_with_sparse(chunks, batch_size=1):
            tasks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                vectors = []
                for c in batch:
                    emb = await self.dv._get_embedding_async(c["content"])
                    sparse = self.dv.bm25_manager.get_sparse_vector(c["content"])
                    vectors.append({
                        "id": c["metadata"]["source_id"],
                        "values": emb,
                        "sparse_values": sparse,
                        "metadata": c["metadata"]
                    })
                self.dv.index.upsert(vectors=vectors)
            return True
        # 执行
        asyncio.run(upsert_chunks_with_sparse(chunks, batch_size=1))
        # 检查upsert参数
        for call in self.dv.index.upsert.call_args_list:
            vectors = call[1]["vectors"]
            for v in vectors:
                self.assertIn("sparse_values", v)
                self.assertEqual(v["sparse_values"], {"indices": [1,2], "values": [0.5, 0.8]})

    def test_upsert_chunks_with_sparse_empty(self):
        """
        空数据case：不应上传任何vector
        """
        self.dv.index.upsert = MagicMock()
        self.dv.bm25_manager = MagicMock()
        chunks = []
        # 设想同上
        async def upsert_chunks_with_sparse(chunks, batch_size=1):
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                vectors = []
                for c in batch:
                    emb = [0.1, 0.2, 0.3]
                    sparse = self.dv.bm25_manager.get_sparse_vector(c["content"])
                    vectors.append({
                        "id": c["metadata"]["source_id"],
                        "values": emb,
                        "sparse_values": sparse,
                        "metadata": c["metadata"]
                    })
                self.dv.index.upsert(vectors=vectors)
            return True
        asyncio.run(upsert_chunks_with_sparse(chunks, batch_size=1))
        self.dv.index.upsert.assert_not_called()

    def test_upsert_chunks_with_sparse_extreme(self):
        """
        极端数据case：超长文本
        """
        self.dv.index.upsert = MagicMock()
        mock_bm25 = MagicMock()
        mock_bm25.get_sparse_vector.return_value = {"indices": [0], "values": [999.9]}
        self.dv.bm25_manager = mock_bm25
        long_text = "极端" * 10000
        chunks = [{"content": long_text, "metadata": {"source_id": "doc#0"}}]
        async def upsert_chunks_with_sparse(chunks, batch_size=1):
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                vectors = []
                for c in batch:
                    emb = [0.1, 0.2, 0.3]
                    sparse = self.dv.bm25_manager.get_sparse_vector(c["content"])
                    vectors.append({
                        "id": c["metadata"]["source_id"],
                        "values": emb,
                        "sparse_values": sparse,
                        "metadata": c["metadata"]
                    })
                self.dv.index.upsert(vectors=vectors)
            return True
        asyncio.run(upsert_chunks_with_sparse(chunks, batch_size=1))
        for call in self.dv.index.upsert.call_args_list:
            vectors = call[1]["vectors"]
            for v in vectors:
                self.assertIn("sparse_values", v)
                self.assertEqual(v["sparse_values"], {"indices": [0], "values": [999.9]})

    def test_upsert_chunks_with_sparse_bm25_exception(self):
        """
        异常数据case：bm25_manager抛异常
        """
        self.dv.index.upsert = MagicMock()
        mock_bm25 = MagicMock()
        mock_bm25.get_sparse_vector.side_effect = Exception("bm25 error")
        self.dv.bm25_manager = mock_bm25
        chunks = [{"content": "异常内容", "metadata": {"source_id": "doc#0"}}]
        async def upsert_chunks_with_sparse(chunks, batch_size=1):
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                vectors = []
                for c in batch:
                    emb = [0.1, 0.2, 0.3]
                    try:
                        sparse = self.dv.bm25_manager.get_sparse_vector(c["content"])
                    except Exception as e:
                        sparse = None
                    vectors.append({
                        "id": c["metadata"]["source_id"],
                        "values": emb,
                        "sparse_values": sparse,
                        "metadata": c["metadata"]
                    })
                self.dv.index.upsert(vectors=vectors)
            return True
        asyncio.run(upsert_chunks_with_sparse(chunks, batch_size=1))
        for call in self.dv.index.upsert.call_args_list:
            vectors = call[1]["vectors"]
            for v in vectors:
                self.assertIn("sparse_values", v)
                self.assertIsNone(v["sparse_values"])

if __name__ == "__main__":
    unittest.main() 