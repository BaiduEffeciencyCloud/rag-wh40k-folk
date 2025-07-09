#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Processing 模块测试套件
自动发现和加载该文件夹下的所有测试类
"""

import unittest
import os
import sys
import importlib
import inspect
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class QueryProcessingTestSuite:
    """Query Processing 测试套件"""
    
    def __init__(self):
        self.test_suite = unittest.TestSuite()
        self.test_loader = unittest.TestLoader()
        self.current_dir = Path(__file__).parent
    
    def discover_tests(self):
        """发现当前目录下的所有测试文件"""
        print("正在发现 Query Processing 测试文件...")
        
        # 获取当前目录下的所有Python文件
        test_files = []
        for file in self.current_dir.glob("test_*.py"):
            if file.name != "test_suite.py":  # 排除测试套件本身
                test_files.append(file)
        
        print(f"发现测试文件: {[f.name for f in test_files]}")
        return test_files
    
    def load_test_modules(self, test_files):
        """加载测试模块"""
        modules = []
        for test_file in test_files:
            try:
                # 将文件路径转换为模块名
                module_name = f"test.query_processing.{test_file.stem}"
                module = importlib.import_module(module_name)
                modules.append(module)
                print(f"成功加载模块: {module_name}")
            except ImportError as e:
                print(f"加载模块失败 {test_file.name}: {e}")
            except Exception as e:
                print(f"加载模块时出现错误 {test_file.name}: {e}")
        
        return modules
    
    def find_test_classes(self, modules):
        """从模块中查找测试类"""
        test_classes = []
        for module in modules:
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, unittest.TestCase) and 
                    obj != unittest.TestCase):
                    test_classes.append(obj)
                    print(f"发现测试类: {obj.__name__}")
        
        return test_classes
    
    def build_test_suite(self):
        """构建测试套件"""
        # 发现测试文件
        test_files = self.discover_tests()
        
        if not test_files:
            print("未发现测试文件")
            return self.test_suite
        
        # 加载测试模块
        modules = self.load_test_modules(test_files)
        
        # 查找测试类
        test_classes = self.find_test_classes(modules)
        
        # 将测试类添加到套件中
        for test_class in test_classes:
            tests = self.test_loader.loadTestsFromTestCase(test_class)
            self.test_suite.addTests(tests)
            print(f"添加测试类到套件: {test_class.__name__}")
        
        return self.test_suite
    
    def run_tests(self, verbosity=2):
        """运行测试套件"""
        print("\n" + "="*50)
        print("开始运行 Query Processing 测试套件")
        print("="*50)
        
        # 构建测试套件
        suite = self.build_test_suite()
        
        if suite.countTestCases() == 0:
            print("没有发现测试用例")
            return
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # 输出测试结果摘要
        print("\n" + "="*50)
        print("测试结果摘要")
        print("="*50)
        print(f"运行测试数: {result.testsRun}")
        print(f"失败数: {len(result.failures)}")
        print(f"错误数: {len(result.errors)}")
        print(f"跳过数: {len(result.skipped)}")
        
        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
        
        return result

def main():
    """主函数"""
    # 创建测试套件
    test_suite = QueryProcessingTestSuite()
    
    # 运行测试
    result = test_suite.run_tests()
    
    # 返回退出码
    if result and (result.failures or result.errors):
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 