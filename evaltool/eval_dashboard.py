import argparse
import os
import sys
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from embedding_eval import run_embedding_eval
from answer_eval import run_answer_eval_v2
from vocab_intrinsic_eval import run_vocab_eval

def main():
    parser = argparse.ArgumentParser(description='RAG系统效果评估中控台')
    parser.add_argument('--mode', choices=['embedding', 'answer', 'dict'], required=True, help='评估模式：embedding/answer/dict')
    parser.add_argument('--csv', help='输入的csv文件路径')
    parser.add_argument('--f', help='输入文件路径，多个文件用逗号分隔（仅dict模式）')
    parser.add_argument('--config', default=None, help='可选，配置文件路径')
    parser.add_argument('--model', default=os.getenv('EMBEDDING_MODEL', 'default-model'), help='embedding模型名称，优先从环境变量读取')
    parser.add_argument('--top_k', type=int, default=int(os.getenv('EVAL_TOP_K', 5)), help='召回top_k，默认5')
    parser.add_argument('--userdict', default=None, help='短语保护userdict.txt路径（仅dict模式）')
    args = parser.parse_args()

    # 日期文件夹管理
    from datetime import datetime
    today = datetime.now().strftime('%y%m%d')
    out_dir = os.path.join(os.path.dirname(__file__), today)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.mode == 'embedding':
        print(f"[INFO] 进入embedding效果评估模式，模型：{args.model}")
        if not args.csv:
            print("[ERROR] embedding模式下--csv为必填")
            sys.exit(1)
        run_embedding_eval(args.csv, model_name=args.model, top_k=args.top_k, config_file=args.config)
    elif args.mode == 'answer':
        print("[INFO] 进入最终答案效果评估模式")
        if not args.csv:
            print("[ERROR] answer模式下--csv为必填")
            sys.exit(1)
        run_answer_eval_v2(args.csv, top_k=args.top_k)
    elif args.mode == 'dict':
        print("[INFO] 进入BM25词典效果评估模式")
        if not args.f:
            print("[ERROR] dict模式下--f为必填，多个文件用逗号分隔")
            sys.exit(1)
        files = [f.strip() for f in args.f.split(',') if f.strip()]
        run_vocab_eval(files, out_dir, user_dict_path=args.userdict)
    else:
        print("[ERROR] 未知评估模式")
        sys.exit(1)

if __name__ == '__main__':
    main() 