#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动更新requirements.txt脚本
扫描项目中的所有Python文件，识别使用的第三方库，并更新requirements.txt
"""

import os
import re
import subprocess
import sys
from typing import Set, Dict, List
from pathlib import Path

# 标准库列表（不需要包含在requirements.txt中）
STDLIB_MODULES = {
    'os', 'sys', 're', 'json', 'logging', 'argparse', 'datetime', 'time',
    'typing', 'dataclasses', 'collections', 'enum', 'glob', 'math', 'string',
    'asyncio', 'concurrent', 'subprocess', 'pathlib', 'unicodedata', 'traceback'
}

# 项目内部模块（不需要包含在requirements.txt中）
INTERNAL_MODULES = {
    'config', 'utils', 'dataupload', 'evaltool', 'orchestrator', 'qProcessor',
    'searchengine', 'aAggregation', 'vector_search', 'query_expander', 'query_cot',
    'cot_search', 'query_processor_v2', 'bm25_manager', 'bm25_config', 'vector_builder',
    'knowledge_graph_manager', 'pinecone_manager', 'data_vector', 'data_vector_v2',
    'data_vector_v3', 'unify', 'gen_userdict', 'seed_neo4j', 'phrase_weight',
    'visualization', 'vocab_intrinsic_eval', 'embedding_eval', 'llm_analysis',
    'ragas_evaluator', 'dedup_agg', 'processor', 'processorfactory', 'search_factory',
    'aggfactory', 'agginterface', 'subquery_retriever', 'llm_utils'
}

def find_python_files(directory: str) -> List[str]:
    """查找目录下的所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 跳过虚拟环境和缓存目录
        dirs[:] = [d for d in dirs if d not in {'venv', '__pycache__', '.git', '.pytest_cache'}]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path: str) -> Set[str]:
    """从Python文件中提取import语句"""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 匹配 import 语句
        import_patterns = [
            r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # import module
            r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import',  # from module import
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                continue
                
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    module = match.group(1)
                    # 过滤掉标准库和内部模块
                    if module not in STDLIB_MODULES and module not in INTERNAL_MODULES:
                        imports.add(module)
                        
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return imports

def get_package_versions(packages: Set[str]) -> Dict[str, str]:
    """获取包的版本信息"""
    versions = {}
    
    for package in packages:
        try:
            # 尝试导入包并获取版本
            result = subprocess.run([
                sys.executable, '-c', 
                f'import {package}; print({package}.__version__)'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                versions[package] = version
            else:
                # 如果无法获取版本，使用默认版本
                versions[package] = 'latest'
                
        except Exception as e:
            print(f"获取 {package} 版本时出错: {e}")
            versions[package] = 'latest'
    
    return versions

def update_requirements_txt(versions: Dict[str, str], output_file: str = 'requirements.txt'):
    """更新requirements.txt文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for package, version in sorted(versions.items()):
                if version == 'latest':
                    f.write(f"{package}\n")
                else:
                    f.write(f"{package}=={version}\n")
        
        print(f"✅ requirements.txt 已更新: {output_file}")
        print(f"📦 包含 {len(versions)} 个包")
        
    except Exception as e:
        print(f"❌ 更新 requirements.txt 失败: {e}")

def main():
    """主函数"""
    print("🔍 开始扫描项目依赖...")
    
    # 查找所有Python文件
    python_files = find_python_files('.')
    print(f"📁 找到 {len(python_files)} 个Python文件")
    
    # 提取所有import
    all_imports = set()
    for file_path in python_files:
        imports = extract_imports(file_path)
        all_imports.update(imports)
    
    print(f"📦 发现 {len(all_imports)} 个第三方包:")
    for imp in sorted(all_imports):
        print(f"  - {imp}")
    
    # 获取版本信息
    print("\n🔍 获取包版本信息...")
    versions = get_package_versions(all_imports)
    
    # 更新requirements.txt
    print("\n📝 更新 requirements.txt...")
    update_requirements_txt(versions)
    
    print("\n✅ 完成！")

if __name__ == "__main__":
    main() 