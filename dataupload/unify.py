import re
import unicodedata
from typing import List, Dict, Any, Optional
import tiktoken
import os
import json
from data_vector_v2 import DocumentChunker
import data_vector_v2

class TextUnifier:
    """
    文本统一化处理类，负责token化与编码，保留专有名词、数字+单位等特殊内容
    """
    
    def __init__(self, encoding_name: str = 'cl100k_base'):
        """
        初始化文本统一化处理器
        
        Args:
            encoding_name: tiktoken编码器名称
        """
        self.encoder = tiktoken.get_encoding(encoding_name)
        
        # 定义需要保护的专有名词模式
        self.protected_patterns = {
            # 战锤40K专有名词
            'faction_names': [
                r'\b(?:Eldar|Aeldari|Space Marines|Imperial Guard|Astra Militarum|Orks|Orcs|Tau|Necrons|Tyranids|Chaos|Dark Eldar|Drukhari|Adeptus Mechanicus|Adepta Sororitas|Grey Knights|Deathwatch|Blood Angels|Ultramarines|Dark Angels|Space Wolves|White Scars|Iron Hands|Salamanders|Raven Guard|Imperial Fists|Black Templars|Crimson Fists)\b',
                r'\b(?:艾达|灵族|星际战士|帝国卫队|帝国军|兽人|钛族|死灵|泰伦虫族|混沌|黑暗灵族|机械教|战斗修女|灰骑士|死亡守望|血天使|极限战士|暗黑天使|太空野狼|白疤|钢铁之手|火蜥蜴|渡鸦守卫|帝国之拳|黑色圣堂|深红之拳)\b'
            ],
            
            # 武器和装备
            'weapons_equipment': [
                r'\b(?:Bolter|Plasma Gun|Melta Gun|Flamer|Heavy Bolter|Lascannon|Missile Launcher|Power Sword|Chainsword|Power Fist|Thunder Hammer|Storm Shield|Terminator Armor|Power Armor|Jump Pack|Land Raider|Rhino|Predator|Vindicator|Whirlwind|Thunderhawk|Stormraven|Stormtalon|Stormhawk)\b',
                r'\b(?:爆弹枪|等离子枪|热熔枪|火焰喷射器|重型爆弹枪|激光炮|导弹发射器|动力剑|链锯剑|动力拳|雷神之锤|风暴盾|终结者装甲|动力装甲|跳跃背包|兰德掠袭者|犀牛|捕食者|复仇者|旋风|雷鹰|风暴渡鸦|风暴爪|风暴鹰)\b'
            ],
            
            # 能力和特性
            'abilities_traits': [
                r'\b(?:Feel No Pain|Furious Charge|Fleet|Stealth|Infiltrate|Scout|Deep Strike|Jump Infantry|Fast Attack|Heavy Support|Troops|Elites|HQ|Warlord|Chapter Master|Captain|Lieutenant|Sergeant|Veteran|Terminator|Dreadnought|Tactical Squad|Assault Squad|Devastator Squad)\b',
                r'\b(?:无畏|狂怒冲锋|快速|潜行|渗透|侦察|深空打击|跳跃步兵|快速攻击|重型支援|部队|精英|指挥部|战团长|连长|中尉|军士|老兵|终结者|无畏机甲|战术小队|突击小队|毁灭者小队)\b'
            ],
            
            # 数字+单位组合
            'number_units': [
                r'\b\d+(?:寸|英寸|cm|mm|m|km|kg|g|t|°C|°F|%|S\d+|T\d+|W\d+|L\d+|A\d+|Sv\d+|BS\d+|WS\d+|I\d+|LD\d+|W\d+|A\d+|Sv\d+)\b',
                r'\b(?:S\d+|T\d+|W\d+|L\d+|A\d+|Sv\d+|BS\d+|WS\d+|I\d+|LD\d+)\b',  # 战锤属性缩写
                r'\b\d+(?:d6|d3|d10|d20)\b',  # 骰子
                r'\b\d+(?:AP|AP-|AP\+\d+)\b',  # 穿甲值
                r'\b\d+(?:D\d+|D\d+)\b',  # 伤害值
            ],
            
            # 特殊符号和缩写
            'special_symbols': [
                r'\b(?:AP|AP-|AP\+\d+|D\d+|S\d+|T\d+|W\d+|L\d+|A\d+|Sv\d+|BS\d+|WS\d+|I\d+|LD\d+)\b',
                r'\b(?:HQ|Troops|Elites|Fast Attack|Heavy Support|Dedicated Transport|Lord of War)\b',
                r'\b(?:WS|BS|S|T|W|I|A|Ld|Sv)\b',
            ]
        }
        
        # 编译正则表达式
        self.compiled_patterns = {}
        for category, patterns in self.protected_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def protect_special_content(self, text: str) -> Dict[str, List[str]]:
        """
        保护特殊内容，返回保护的内容和替换后的文本
        
        Args:
            text: 原始文本
            
        Returns:
            包含保护内容的字典和替换后的文本
        """
        protected_content = {}
        processed_text = text
        
        # 为每个类别生成唯一的占位符
        placeholder_counter = 0
        
        for category, patterns in self.compiled_patterns.items():
            protected_content[category] = []
            
            for pattern in patterns:
                matches = pattern.finditer(processed_text)
                for match in matches:
                    original = match.group(0)
                    placeholder = f"__PROTECTED_{placeholder_counter}__"
                    protected_content[category].append({
                        'original': original,
                        'placeholder': placeholder,
                        'position': match.span()
                    })
                    processed_text = processed_text.replace(original, placeholder, 1)
                    placeholder_counter += 1
        
        return {
            'protected_content': protected_content,
            'processed_text': processed_text
        }
    
    def restore_special_content(self, text: str, protected_content: Dict[str, List[str]]) -> str:
        """
        恢复被保护的特殊内容
        
        Args:
            text: 处理后的文本
            protected_content: 保护的内容字典
            
        Returns:
            恢复后的文本
        """
        restored_text = text
        
        # 按位置排序，从后往前替换，避免位置偏移
        all_placeholders = []
        for category, items in protected_content.items():
            for item in items:
                all_placeholders.append(item)
        
        # 按位置排序
        all_placeholders.sort(key=lambda x: x['position'][0], reverse=True)
        
        for item in all_placeholders:
            restored_text = restored_text.replace(item['placeholder'], item['original'])
        
        return restored_text
    
    def normalize_text(self, text: str) -> str:
        """
        文本标准化处理
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        # 1. 保护特殊内容
        protection_result = self.protect_special_content(text)
        processed_text = protection_result['processed_text']
        protected_content = protection_result['protected_content']
        
        # 2. 基础文本清理
        # 统一空白字符
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        # 统一标点符号
        processed_text = unicodedata.normalize('NFKC', processed_text)
        
        # 清理多余的空格
        processed_text = processed_text.strip()
        
        # 3. 恢复特殊内容
        restored_text = self.restore_special_content(processed_text, protected_content)
        
        return restored_text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        对文本进行token化
        
        Args:
            text: 文本
            
        Returns:
            token列表
        """
        return self.encoder.encode(text)
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 文本
            
        Returns:
            token数量
        """
        return len(self.tokenize_text(text))
    
    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个chunk，统一化其文本内容
        
        Args:
            chunk: 包含文本的chunk字典
            
        Returns:
            处理后的chunk字典
        """
        processed_chunk = chunk.copy()
        
        # 处理文本内容
        if 'text' in chunk:
            original_text = chunk['text']
            normalized_text = self.normalize_text(original_text)
            processed_chunk['text'] = normalized_text
            processed_chunk['original_text'] = original_text  # 保留原始文本
            processed_chunk['token_count'] = self.count_tokens(normalized_text)
        
        elif 'content' in chunk:
            original_content = chunk['content']
            normalized_content = self.normalize_text(original_content)
            processed_chunk['content'] = normalized_content
            processed_chunk['original_content'] = original_content  # 保留原始内容
            processed_chunk['token_count'] = self.count_tokens(normalized_content)
        
        return processed_chunk
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理chunks
        
        Args:
            chunks: chunk列表
            
        Returns:
            处理后的chunk列表
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            processed_chunk = self.process_chunk(chunk)
            processed_chunk['unified_chunk_index'] = i  # 添加统一化后的索引
            processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def save_chunks_to_json(self, chunks: List[Dict[str, Any]], input_filename: str, output_dir: str = "chunks") -> str:
        """
        保存处理后的chunks为JSON文件，文件名为after_{input}.json
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        base = os.path.basename(input_filename)
        name, _ = os.path.splitext(base)
        output_file = os.path.join(output_dir, f"after_{name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"处理后的chunks已保存到: {output_file}")
        return output_file


def main():
    """
    测试函数
    """
    # 创建统一化处理器
    unifier = TextUnifier()
    
    # 测试文本
    test_file = "../test/testdata/aeldaricodex.md"
    if not os.path.exists(test_file):
        print(f"未找到测试文件: {test_file}")
        return
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 切片

    chunker = DocumentChunker()
    chunks = chunker.chunk_text(content)
    print(f"原始chunk数: {len(chunks)}")
    
    # 统一化处理
    processed_chunks = unifier.process_chunks(chunks)
    print(f"统一化后chunk数: {len(processed_chunks)}")
    # 保存初始切片后的chunks
    data_vector_v2.save_chunks_to_json(processed_chunks)
    # 保存处理后的chunks
    unifier.save_chunks_to_json(processed_chunks, test_file)
    
    # 其余演示
    print("前2个处理后chunk示例:")
    for i, chunk in enumerate(processed_chunks[:2]):
        print(json.dumps(chunk, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 