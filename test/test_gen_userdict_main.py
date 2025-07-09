#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys
import os
import tempfile
import shutil
import subprocess
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestGenUserdictMain(unittest.TestCase):
    """测试gen_userdict.py的main函数"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用项目中现有的真实测试数据
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.input_dir = os.path.join(self.base_dir, 'test', 'testdata')
        self.output_dir = tempfile.mkdtemp()
        
        # 确保输入目录存在且有数据
        self.assertTrue(os.path.exists(self.input_dir), f"测试数据目录不存在: {self.input_dir}")
        test_files = os.listdir(self.input_dir)
        self.assertGreater(len(test_files), 0, f"测试数据目录为空: {self.input_dir}")
    
    def test_main_basic_functionality(self):
        """测试main函数基本功能"""
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--min-freq', '1',
            '--min-pmi', '0',
            '--max-len', '6'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 检查命令执行成功
        self.assertEqual(result.returncode, 0, f"命令执行失败: {result.stderr}")
        
        # 检查输出文件
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0, "应该生成输出文件")
        
        # 检查词典文件
        dict_files = [f for f in output_files if f.endswith('.txt') and not f.endswith('.meta.json')]
        self.assertGreater(len(dict_files), 0, "应该生成词典文件")
        
        # 检查词汇表文件
        vocab_files = [f for f in output_files if f.endswith('.json') and not f.endswith('.meta.json')]
        self.assertGreater(len(vocab_files), 0, "应该生成词汇表文件")
        
        # 检查meta文件
        meta_files = [f for f in output_files if f.endswith('.meta.json')]
        self.assertGreater(len(meta_files), 0, "应该生成meta文件")
    
    def test_main_with_filter(self):
        """测试main函数使用--filter参数"""
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--min-freq', '1',
            '--min-pmi', '0',
            '--max-len', '6',
            '--filter'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"命令执行失败: {result.stderr}")
        
        # 检查输出文件
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)
    
    def test_main_with_ac_mask(self):
        """测试main函数使用--ac-mask参数"""
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--min-freq', '1',
            '--min-pmi', '0',
            '--max-len', '6',
            '--ac-mask'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"命令执行失败: {result.stderr}")
        
        # 检查输出文件
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)
    
    def test_main_with_auto_threshold(self):
        """测试main函数使用--auto-threshold参数"""
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--min-freq', '1',
            '--min-pmi', '0',
            '--max-len', '6',
            '--auto-threshold'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"命令执行失败: {result.stderr}")
        
        # 检查输出文件
        output_files = os.listdir(self.output_dir)
        self.assertGreater(len(output_files), 0)
    
    def test_main_invalid_input_dir(self):
        """测试main函数无效输入目录"""
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', '/nonexistent/path',
            '--output-dir', self.output_dir,
            '--min-freq', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        # gen_userdict.py检测到目录不存在时打印错误但返回码是0
        self.assertEqual(result.returncode, 0, "无效输入目录应该正常退出")
        self.assertIn("输入文件夹不存在", result.stdout, "应该输出错误信息")
    
    def test_main_empty_input_dir(self):
        """测试main函数空输入目录"""
        empty_dir = os.path.join(self.base_dir, 'test', 'testdata', 'empty')
        os.makedirs(empty_dir, exist_ok=True)
        
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', empty_dir,
            '--output-dir', self.output_dir,
            '--min-freq', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        # 重构后空目录会优雅处理，返回码为0
        self.assertEqual(result.returncode, 0, "空目录应该能正常处理")
        self.assertIn("未读取到任何文本", result.stdout, "应该输出空数据警告")
        self.assertIn("未生成任何文件", result.stdout, "应该输出未生成文件警告")
    
    def test_main_output_file_structure(self):
        """测试main函数输出文件结构"""
        cmd = [
            'python3', 'dataupload/gen_userdict.py',
            '--input-dir', self.input_dir,
            '--output-dir', self.output_dir,
            '--min-freq', '1',
            '--min-pmi', '0',
            '--max-len', '6'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        
        # 检查词典文件内容
        dict_files = [f for f in os.listdir(self.output_dir) if f.endswith('.txt') and not f.endswith('.meta.json')]
        if dict_files:
            dict_file = os.path.join(self.output_dir, dict_files[0])
            with open(dict_file, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertGreater(len(content.strip()), 0, "词典文件应该有内容")
        
        # 检查词汇表文件内容
        vocab_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json') and not f.endswith('.meta.json')]
        if vocab_files:
            vocab_file = os.path.join(self.output_dir, vocab_files[0])
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.assertIsInstance(vocab_data, dict, "词汇表文件应该是有效的JSON字典")
            self.assertGreater(len(vocab_data), 0, "词汇表应该有内容")
        
        # 检查meta文件内容
        meta_files = [f for f in os.listdir(self.output_dir) if f.endswith('.meta.json')]
        if meta_files:
            meta_file = os.path.join(self.output_dir, meta_files[0])
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            self.assertIsInstance(meta_data, dict, "meta文件应该是有效的JSON字典")
    
    def test_main_parameter_effects(self):
        """测试main函数参数对结果的影响"""
        # 测试不同的min_freq值
        for min_freq in [1, 2, 3]:
            # 清空输出目录
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
            
            cmd = [
                'python3', 'dataupload/gen_userdict.py',
                '--input-dir', self.input_dir,
                '--output-dir', self.output_dir,
                '--min-freq', str(min_freq),
                '--min-pmi', '0',
                '--max-len', '6'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"min_freq={min_freq}时命令执行失败")
            
            # 检查是否生成了文件
            output_files = os.listdir(self.output_dir)
            self.assertGreater(len(output_files), 0, f"min_freq={min_freq}时应该生成文件")
    
    def tearDown(self):
        """清理测试文件"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main() 