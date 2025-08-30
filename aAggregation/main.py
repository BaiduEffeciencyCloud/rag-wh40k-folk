import argparse
from .aggfactory import AggFactory
from typing import List
import sys


def main():
    parser = argparse.ArgumentParser(description="DynamicAggregation Prompt 测试脚本")
    parser.add_argument('--template', type=str, default='default', help='模板名，默认为default')
    parser.add_argument('--query', type=str, required=True, help='用户问题')
    parser.add_argument('--context', type=str, action='append', required=True, help='上下文，可多次传入')
    args = parser.parse_args()

    aggregator = AggFactory.get_aggregator('simple', template=args.template)
    prompt = aggregator._load_prompt(args.context, args.query)
    print("\n===== 生成的Prompt =====\n")
    print(prompt)

if __name__ == '__main__':
    main() 