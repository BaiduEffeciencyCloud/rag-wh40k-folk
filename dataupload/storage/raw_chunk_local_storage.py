import os
import json
from typing import List
from .storage_interface import StorageInterface

# 获取项目根目录（假设本文件在 dataupload/storage/ 下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_CHUNKS_FILE = "raw_chunks.json"
RAW_FOLDER = os.path.join(PROJECT_ROOT, "raw_chunk")

class RawChunkStorage(StorageInterface):
    """
    本地文件存储实现
    """
    def __init__(self, base_dir: str = RAW_FOLDER):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def storage(self, data: List[str], filename: str = RAW_CHUNKS_FILE):
        """
        将字符串列表数据保存为本地json文件，若文件已存在则合并去重后保存
        """
        file_path = os.path.join(self.base_dir, filename)
        if os.path.exists(file_path):
            # 读取原有内容
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    old_data = json.load(f)
                except Exception:
                    old_data = []
            # 合并去重
            merged = list(dict.fromkeys(old_data + data))
        else:
            merged = data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        return file_path

    def load(self, filename: str = RAW_CHUNKS_FILE) -> List[str]:
        file_path = os.path.join(self.base_dir, filename)
        if not os.path.exists(file_path):
            # 文件不存在则新建空文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)