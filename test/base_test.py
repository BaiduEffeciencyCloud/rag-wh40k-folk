import tempfile
import shutil
import os
import unittest
import builtins
from send2trash import send2trash

class TestBase(unittest.TestCase):
    """所有测试用例的基类，自动保护项目文件"""
    
    def setUp(self):
        # 只在系统临时目录创建测试文件
        self.temp_dir = tempfile.mkdtemp(dir=tempfile.gettempdir())
        self._created_temp_dirs = [self.temp_dir]
        super().setUp()
    
    def tearDown(self):
        # 只删除系统临时目录下的内容，区分文件和目录
        for temp_path in getattr(self, "_created_temp_dirs", []):
            if temp_path.startswith(tempfile.gettempdir()):
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
                elif os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
        super().tearDown()
    
    def create_temp_file(self, prefix="test_", suffix=".tmp"):
        """安全的临时文件创建方法"""
        temp_file = tempfile.NamedTemporaryFile(
            prefix=prefix, 
            suffix=suffix, 
            dir=tempfile.gettempdir(),
            delete=False
        )
        self._created_temp_dirs.append(temp_file.name)
        return temp_file.name 

# ========== 全局删除保护（使用send2trash） ========== 

_original_rmtree = shutil.rmtree
_original_remove = os.remove

def safe_rmtree(path, *args, **kwargs):
    temp_root = tempfile.gettempdir()
    abs_path = os.path.abspath(path)
    if abs_path.startswith(temp_root):
        return _original_rmtree(path, *args, **kwargs)
    else:
        print(f"安全保护：将目录移动到回收站 {abs_path}")
        return send2trash(abs_path)

def safe_remove(path, *args, **kwargs):
    temp_root = tempfile.gettempdir()
    abs_path = os.path.abspath(path)
    if abs_path.startswith(temp_root):
        return _original_remove(path, *args, **kwargs)
    else:
        print(f"安全保护：将文件移动到回收站 {abs_path}")
        return send2trash(abs_path)

shutil.rmtree = safe_rmtree
os.remove = safe_remove 