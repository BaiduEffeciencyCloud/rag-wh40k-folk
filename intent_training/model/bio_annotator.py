"""
BIO标注转换器
将槽位标签转换为BIO标注格式，支持序列标注训练
"""

import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class BIOAnnotator:
    """BIO标注转换器"""
    
    def __init__(self):
        """初始化BIO标注器"""
        self.bio_tags = ['O', 'B', 'I']  # Outside, Beginning, Inside
    
    def convert_slots_to_bio(self, query: str, slots: Dict[str, str]) -> List[str]:
        """
        将槽位字典转换为BIO标注序列
        
        Args:
            query: 查询文本
            slots: 槽位字典 {slot_name: slot_value}
            
        Returns:
            BIO标注序列
        """
        # 初始化所有token为O
        tokens = list(query)
        bio_labels = ['O'] * len(tokens)
        
        # 为每个槽位找到在查询中的位置并标注
        for slot_name, slot_value in slots.items():
            if not slot_value:
                continue
                
            # 查找槽位值在查询中的位置
            positions = self._find_slot_positions(query, slot_value)
            
            for start_pos, end_pos in positions:
                # 标注B-{slot_name} (开始)
                if start_pos < len(bio_labels):
                    bio_labels[start_pos] = f"B-{slot_name}"
                
                # 标注I-{slot_name} (中间)
                for pos in range(start_pos + 1, min(end_pos, len(bio_labels))):
                    bio_labels[pos] = f"I-{slot_name}"
        
        return bio_labels
    
    def _find_slot_positions(self, query: str, slot_value: str) -> List[Tuple[int, int]]:
        """
        查找槽位值在查询中的所有位置
        
        Args:
            query: 查询文本
            slot_value: 槽位值
            
        Returns:
            位置列表 [(start_pos, end_pos), ...]
        """
        positions = []
        start = 0
        
        while True:
            pos = query.find(slot_value, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(slot_value)))
            start = pos + 1
        
        return positions
    
    def convert_bio_to_slots(self, query: str, bio_labels: List[str]) -> Dict[str, List[Tuple[str, int, int]]]:
        """
        将BIO标注序列转换回槽位字典
        
        Args:
            query: 查询文本
            bio_labels: BIO标注序列
            
        Returns:
            槽位字典 {slot_name: [(value, start_pos, end_pos), ...]}
        """
        slots = {}
        current_slot = None
        current_start = -1
        
        for i, label in enumerate(bio_labels):
            if label == 'O':
                # 结束当前槽位
                if current_slot:
                    slot_value = query[current_start:i]
                    if current_slot not in slots:
                        slots[current_slot] = []
                    slots[current_slot].append((slot_value, current_start, i))
                    current_slot = None
                    current_start = -1
            
            elif label.startswith('B-'):
                # 结束前一个槽位，开始新槽位
                if current_slot:
                    slot_value = query[current_start:i]
                    if current_slot not in slots:
                        slots[current_slot] = []
                    slots[current_slot].append((slot_value, current_start, i))
                
                # 开始新槽位
                current_slot = label[2:]  # 去掉'B-'前缀
                current_start = i
            
            elif label.startswith('I-'):
                # 继续当前槽位
                if current_slot and label[2:] == current_slot:
                    continue  # 继续当前槽位
                else:
                    # 槽位不匹配，结束当前槽位
                    if current_slot:
                        slot_value = query[current_start:i]
                        if current_slot not in slots:
                            slots[current_slot] = []
                        slots[current_slot].append((slot_value, current_start, i))
                    current_slot = None
                    current_start = -1
        
        # 处理最后一个槽位
        if current_slot:
            slot_value = query[current_start:len(bio_labels)]
            if current_slot not in slots:
                slots[current_slot] = []
            slots[current_slot].append((slot_value, current_start, len(bio_labels)))
        
        return slots
    
    def validate_bio_sequence(self, bio_labels: List[str]) -> bool:
        """
        验证BIO标注序列的合法性
        
        Args:
            bio_labels: BIO标注序列
            
        Returns:
            是否合法
        """
        for i, label in enumerate(bio_labels):
            if label == 'O':
                continue
            
            if label.startswith('B-'):
                # B标签后应该跟I标签或O标签
                continue
            
            elif label.startswith('I-'):
                # I标签前应该有B标签或I标签
                if i == 0:
                    return False  # I标签不能出现在开头
                
                prev_label = bio_labels[i-1]
                if prev_label == 'O':
                    return False  # I标签前不能是O标签
                
                if prev_label.startswith('B-') or prev_label.startswith('I-'):
                    # 检查槽位名称是否一致
                    current_slot = label[2:]
                    prev_slot = prev_label[2:]
                    if current_slot != prev_slot:
                        return False  # 槽位名称不一致
        
        return True
    
    def get_unique_slots(self, bio_labels: List[str]) -> List[str]:
        """
        从BIO标注序列中提取唯一的槽位名称
        
        Args:
            bio_labels: BIO标注序列
            
        Returns:
            唯一槽位名称列表
        """
        slots = set()
        for label in bio_labels:
            if label.startswith('B-') or label.startswith('I-'):
                slot_name = label[2:]
                slots.add(slot_name)
        return list(slots)
    
    def create_training_data(self, queries: List[str], slot_labels: List[Dict[str, str]]) -> List[Tuple[List[str], List[str]]]:
        """
        创建训练数据：查询token序列和对应的BIO标注序列
        
        Args:
            queries: 查询列表
            slot_labels: 槽位标签列表
            
        Returns:
            训练数据列表 [(tokens, bio_labels), ...]
        """
        training_data = []
        
        for query, slots in zip(queries, slot_labels):
            tokens = list(query)
            bio_labels = self.convert_slots_to_bio(query, slots)
            
            # 验证BIO序列的合法性
            if self.validate_bio_sequence(bio_labels):
                training_data.append((tokens, bio_labels))
            else:
                logger.warning(f"无效的BIO序列: {bio_labels} for query: {query}")
        
        return training_data 