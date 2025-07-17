import os
import json
import shutil
from dataupload.storage.raw_chunk_local_storage import RawChunkStorage, RAW_CHUNKS_FILE, RAW_FOLDER
from test.base_test import TestBase

class TestRawChunkStorage(TestBase):
    def setUp(self):
        super().setUp()
        # 用临时目录替换RAW_FOLDER，避免污染真实数据
        self.test_dir = self.temp_dir
        self.storage = RawChunkStorage(base_dir=self.test_dir)
        self.file_path = os.path.join(self.test_dir, RAW_CHUNKS_FILE)

    def tearDown(self):
        super().tearDown()

    def test_load_empty(self):
        # 文件不存在时应自动创建空文件并返回空列表
        result = self.storage.load()
        self.assertEqual(result, [])
        self.assertTrue(os.path.exists(self.file_path))
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.assertEqual(json.load(f), [])

    def test_storage_and_load(self):
        data = ["chunk1", "chunk2"]
        self.storage.storage(data)
        loaded = self.storage.load()
        self.assertEqual(set(loaded), set(data))

    def test_storage_append_and_dedup(self):
        data1 = ["chunk1", "chunk2"]
        data2 = ["chunk2", "chunk3"]
        self.storage.storage(data1)
        self.storage.storage(data2)
        loaded = self.storage.load()
        # 应合并去重
        self.assertEqual(set(loaded), {"chunk1", "chunk2", "chunk3"})

    def test_storage_with_invalid_json(self):
        # 手动写入非法json，storage应兜底
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write("not a json")
        data = ["chunk4"]
        self.storage.storage(data)
        loaded = self.storage.load()
        self.assertIn("chunk4", loaded) 