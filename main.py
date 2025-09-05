#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统主入口
"""

import argparse
import logging
from typing import Dict, Any

from orchestrator.master import MasterController


class RAGMainController:
    """RAG系统主控制器"""
    
    def __init__(self):
        """初始化主控制器"""
        self.master_controller = MasterController()
    
    def parse_arguments(self) -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='RAG系统检索入口')
        parser.add_argument('query', type=str, help='查询内容')
        parser.add_argument('--advance', action='store_true', help='启用高级模式（意图识别）')
        
        return parser.parse_args()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """处理查询"""
        try:
            # 调用master controller进行意图识别和配置生成
            result = self.master_controller.process_query(query)
            
            if result and 'intent_type' in result and 'processor_config' in result:
                logging.info(f"查询处理成功: 意图={result['intent_type']}")
                return result
            else:
                logging.error("master controller返回结果格式错误")
                return {
                    'intent_type': 'query',
                    'processor_config': {},
                    'query': query,
                    'error': '结果格式错误'
                }
                
        except Exception as e:
            logging.error(f"查询处理失败: {e}")
            return {
                'intent_type': 'query',
                'processor_config': {},
                'query': query,
                'error': str(e)
            }
    
    def execute_query(self, query: str, advance: bool = False) -> Dict[str, Any]:
        """执行查询：调用master controller执行完整的查询流程"""
        try:
            return self.master_controller.execute_query(query, advance)
        except Exception as e:
            logging.error(f"执行查询失败: {e}")
            return {
                'intent_type': 'query',
                'processor_config': {},
                'query': query,
                'error': str(e),
                'results': None
            }
    
    def main(self):
        """主函数"""
        try:
            # 解析命令行参数
            args = self.parse_arguments()
            
            # 设置日志
            self.setup_logging()
            
            logging.info(f"开始处理查询: {args.query}")
            logging.info(f"模式: {'高级' if args.advance else '简单'}")
            
            # 处理查询
            result = self.execute_query(args.query, args.advance)
            
            # 输出结果
            if result:
                print(f"\n查询: {args.query}")
                if 'intent_type' in result:
                    print(f"识别意图: {result['intent_type']}")
                if 'processor_config' in result:
                    config = result['processor_config']
                    print(f"处理器配置: qp={config.get('qp', 'N/A')}, se={config.get('se', 'N/A')}, ps={config.get('ps', 'N/A')}, agg={config.get('agg', 'N/A')}")
                
                # 输出查询结果
                if 'results' in result and result['results']:
                    print(f"\n查询结果:")
                    # 如果结果包含final_answer，优先显示并确保转义字符正确显示
                    if isinstance(result['results'], dict) and 'final_answer' in result['results']:
                        print(result['results']['final_answer'])
                    else:
                        print(result['results'])
                elif 'error' in result:
                    print(f"查询出错: {result['error']}")
                else:
                    print("未找到查询结果")
                
                return result
            else:
                logging.error("查询执行失败，未返回结果")
                return None
                
        except Exception as e:
            logging.error(f"主函数执行失败: {e}")
            return None


def main():
    """程序入口点"""
    controller = RAGMainController()
    return controller.main()


if __name__ == "__main__":
    main()
