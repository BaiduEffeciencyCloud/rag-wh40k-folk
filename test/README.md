# 测试体系说明

## 1. 测试分类

本项目的测试按照功能模块进行分类，每个模块都有独立的测试目录：

```
test/
├── query_processing/       # 查询处理测试
│   ├── test_qprocessor.py  # qProcessor模块测试
│   └── test_suite.py       # 查询处理测试套件
├── retrieval/              # 检索召回测试
│   ├── test_dense_search.py
│   ├── test_sparse_search.py
│   └── test_suite.py
├── aggregation/            # 答案融合测试
│   ├── test_weighted_fusion.py
│   └── test_suite.py
├── dictionary/             # 词典测试
│   ├── test_bm25_dict.py
│   └── test_suite.py
├── upload/                 # 上传测试
│   ├── test_file_upload.py
│   └── test_suite.py
├── integration/            # 继承测试
│   ├── test_end_to_end.py
│   └── test_suite.py
├── utils/                  # 测试工具
│   ├── test_helpers.py
│   └── test_suite.py
├── run_all_tests.py        # 全局测试套件管理器
└── README.md              # 本文件
```

## 2. 测试套件使用

### 2.1 运行单个模块的测试

```bash
# 运行查询处理模块测试
python3 test/query_processing/test_suite.py

# 运行检索召回模块测试
python3 test/retrieval/test_suite.py

# 运行答案融合模块测试
python3 test/aggregation/test_suite.py
```

### 2.2 运行所有测试

```bash
# 运行所有模块的测试
python3 test/run_all_tests.py

# 运行指定模块的测试
python3 test/run_all_tests.py query_processing
python3 test/run_all_tests.py retrieval
```

### 2.3 运行单个测试文件

```bash
# 直接运行单个测试文件
python3 test/query_processing/test_qprocessor.py
```

## 3. 测试编写规范

### 3.1 测试类命名

- 测试类以 `Test` 开头
- 类名应该清晰描述被测试的功能
- 例如：`TestStraightforwardProcessor`、`TestExpanderProcessor`

### 3.2 测试方法命名

- 测试方法以 `test_` 开头
- 方法名应该描述测试的具体场景
- 例如：`test_processor_creation`、`test_process_single_query`

### 3.3 测试结构

```python
import unittest
from your_module import YourClass

class TestYourClass(unittest.TestCase):
    """测试YourClass的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.instance = YourClass()
    
    def test_some_functionality(self):
        """测试某个功能"""
        # 准备测试数据
        test_input = "test data"
        
        # 执行被测试的方法
        result = self.instance.some_method(test_input)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result, expected_value)
    
    def tearDown(self):
        """测试后的清理工作"""
        pass
```

## 4. 测试套件开发

### 4.1 创建新的测试套件

1. 在对应的测试目录下创建 `test_suite.py`
2. 继承 `QueryProcessingTestSuite` 的模式
3. 实现自动发现和加载测试类的功能

### 4.2 测试套件模板

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名称测试套件
"""

import unittest
import sys
import os
import importlib
import inspect
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ModuleTestSuite:
    """模块测试套件"""
    
    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.test_loader = unittest.TestLoader()
        self.current_dir = Path(__file__).parent
    
    def discover_tests(self):
        """发现当前目录下的所有测试文件"""
        test_files = []
        for file in self.current_dir.glob("test_*.py"):
            if file.name != "test_suite.py":
                test_files.append(file)
        return test_files
    
    def build_test_suite(self):
        """构建测试套件"""
        # 实现测试发现和加载逻辑
        pass
    
    def run_tests(self, verbosity=2):
        """运行测试套件"""
        # 实现测试运行逻辑
        pass

def main():
    """主函数"""
    test_suite = ModuleTestSuite()
    result = test_suite.run_tests()
    return 0 if result and not (result.failures or result.errors) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
```

## 5. 最佳实践

### 5.1 测试覆盖

- 每个功能模块都应该有对应的测试
- 测试应该覆盖正常情况、边界情况和异常情况
- 测试用例应该独立，不依赖其他测试的执行结果

### 5.2 测试数据

- 使用固定的测试数据，避免随机性
- 测试数据应该具有代表性，覆盖各种场景
- 大型测试数据可以放在单独的文件中

### 5.3 测试性能

- 单元测试应该快速执行
- 避免在单元测试中进行网络请求或数据库操作
- 使用mock对象模拟外部依赖

### 5.4 测试维护

- 定期运行测试，确保代码质量
- 及时修复失败的测试
- 随着功能变化更新测试用例

## 6. 持续集成

建议在CI/CD流程中集成测试：

```yaml
# .github/workflows/test.yml 示例
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python3 test/run_all_tests.py
```

## 7. 故障排除

### 7.1 常见问题

1. **导入错误**: 确保项目根目录在Python路径中
2. **测试发现失败**: 检查测试文件命名是否符合规范
3. **测试套件加载失败**: 确保test_suite.py有正确的main函数

### 7.2 调试技巧

- 使用 `python3 -v` 查看详细的导入信息
- 在测试中添加 `print` 语句进行调试
- 使用 `pdb` 进行断点调试 