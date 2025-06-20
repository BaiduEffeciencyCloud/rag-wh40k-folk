import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
import re
import json
from datetime import datetime

class DataVector:
    def __init__(self, output_dir: str = "chunks"):
        """
        初始化DataVector类
        
        Args:
            output_dir: 输出目录，用于保存切分后的文本文件
        """
        # 加载环境变量
        load_dotenv('../.env')
        
        # 初始化输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化文本分割器
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3")
            ]
        )
    
    def process_markdown(self, file_path: str, faction: str) -> List[Dict[str, Any]]:
        """
        处理Markdown文件，提取内容并保存为JSON格式
        
        Args:
            file_path: Markdown文件路径
            faction: 派系名称
            
        Returns:
            List[Dict[str, Any]]: 处理后的文本切片列表
        """
        print(f"开始处理文件: {file_path}")
        
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用MarkdownHeaderTextSplitter分割文本
            splits = self.header_splitter.split_text(content)
            
            # 处理分割后的文本
            processed_chunks = []
            for i, split in enumerate(splits):
                # 获取元数据
                metadata = split.metadata
                
                # 处理文本内容
                text = split.page_content.strip()
                
                # 跳过空内容
                if not text:
                    continue
                
                # 创建切片数据
                chunk_data = {
                    "content": text,
                    "metadata": {
                        "source_file": os.path.basename(file_path),
                        "chunk_index": i,
                        "faction": faction,
                        "total_chunks": len(splits),
                        "timestamp": datetime.now().isoformat(),
                        "content_type": "markdown",
                        "chunk_size": len(text),
                        "language": "zh",
                        **metadata
                    }
                }
                processed_chunks.append(chunk_data)
            
            print(f"成功处理 {len(processed_chunks)} 个切片")
            return processed_chunks
            
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            raise
    
    def save_chunks(self, chunks: List[Dict[str, Any]], faction: str, source_file: str) -> str:
        """
        将处理后的切片保存为JSON文件
        
        Args:
            chunks: 处理后的文本切片列表
            faction: 派系名称
            source_file: 源文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            # 确保输出目录存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(source_file))[0]
            output_file = os.path.join(
                self.output_dir,
                f"{base_name}_{faction}_{timestamp}.json"
            )
            
            # 保存为JSON文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "source_file": source_file,
                        "faction": faction,
                        "total_chunks": len(chunks),
                        "timestamp": timestamp,
                        "version": "1.0"
                    },
                    "chunks": chunks
                }, f, ensure_ascii=False, indent=2)
            
            print(f"切片已保存到: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"保存切片时出错: {str(e)}")
            raise

async def main(file_path: str, faction: str):
    """
    主函数
    
    Args:
        file_path: 要处理的文档路径
        faction: 文档类型（如：eldar、space_marines等）
    """
    # 创建DataVector实例
    vectorizer = DataVector()
    
    # 处理文档
    chunks = vectorizer.process_markdown(file_path, faction)
    
    # 保存到文件
    output_file = vectorizer.save_chunks(chunks, faction, file_path)

if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) != 3:
        print("使用方法: python3 data_vector.py <文件路径> <faction>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    faction = sys.argv[2]
    
    asyncio.run(main(file_path, faction)) 