#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局测试套件管理器
可以运行所有模块的测试
"""

import unittest
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def discover_test_suites():
    """发现所有测试套件"""
    test_dir = Path(__file__).parent
    test_suites = []
    
    # 查找所有test_suite.py文件
    for suite_file in test_dir.rglob("test_suite.py"):
        module_path = suite_file.parent
        module_name = module_path.name
        
        # 跳过根目录的test_suite.py（如果存在）
        if module_name == "test":
            continue
            
        test_suites.append({
            "name": module_name,
            "path": module_path,
            "suite_file": suite_file
        })
    
    return test_suites

def run_single_suite(suite_info):
    """运行单个测试套件"""
    print(f"\n{'='*60}")
    print(f"运行 {suite_info['name']} 测试套件")
    print(f"{'='*60}")
    
    # 切换到测试套件目录
    original_cwd = os.getcwd()
    os.chdir(suite_info['path'])
    
    try:
        # 导入并运行测试套件
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_suite", suite_info['suite_file'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 如果模块有main函数，调用它
        if hasattr(module, 'main'):
            result = module.main()
            return result
        else:
            print(f"警告：{suite_info['name']} 测试套件没有main函数")
            return None
            
    except Exception as e:
        print(f"运行 {suite_info['name']} 测试套件时出错: {e}")
        return None
    finally:
        os.chdir(original_cwd)

def run_all_tests():
    """运行所有测试套件"""
    print("开始运行所有测试套件...")
    
    # 发现所有测试套件
    test_suites = discover_test_suites()
    
    if not test_suites:
        print("未发现任何测试套件")
        return 1
    
    print(f"发现 {len(test_suites)} 个测试套件:")
    for suite in test_suites:
        print(f"  - {suite['name']}")
    
    # 运行所有测试套件
    results = []
    for suite in test_suites:
        result = run_single_suite(suite)
        results.append((suite['name'], result))
    
    # 输出总体结果
    print(f"\n{'='*60}")
    print("所有测试套件运行完成")
    print(f"{'='*60}")
    
    success_count = 0
    for suite_name, result in results:
        if result == 0:
            print(f"✅ {suite_name}: 通过")
            success_count += 1
        elif result == 1:
            print(f"❌ {suite_name}: 失败")
        else:
            print(f"⚠️  {suite_name}: 未知状态")
    
    print(f"\n总计: {len(results)} 个测试套件，{success_count} 个通过")
    
    # 返回总体结果
    if success_count == len(results):
        return 0
    else:
        return 1

def run_specific_suite(suite_name):
    """运行指定的测试套件"""
    test_suites = discover_test_suites()
    
    for suite in test_suites:
        if suite['name'] == suite_name:
            return run_single_suite(suite)
    
    print(f"未找到测试套件: {suite_name}")
    print("可用的测试套件:")
    for suite in test_suites:
        print(f"  - {suite['name']}")
    return 1

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行指定的测试套件
        suite_name = sys.argv[1]
        return run_specific_suite(suite_name)
    else:
        # 运行所有测试套件
        return run_all_tests()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 