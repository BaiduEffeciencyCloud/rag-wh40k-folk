from abc import ABC, abstractmethod
from typing import List, Dict, Any

class PostSearchInterface(ABC):
    """Post-Search 处理器接口基类"""
    
    @abstractmethod
    def process(self, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        处理检索结果
        Args:
            results: 检索结果列表，格式与dense_search.py的search方法一致
            **kwargs: 其他参数（如mode、query_info等）
        Returns:
            处理后的结果列表
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        获取处理器名称
        Returns:
            处理器名称字符串
        """
        pass
    
    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        获取配置模式
        Returns:
            配置模式字典，描述该处理器支持的参数
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        验证配置是否符合处理器要求
        Args:
            config: 配置字典
        Returns:
            (is_valid, error_messages): 验证结果和错误信息列表
        """
        schema = self.get_config_schema()
        errors = []
        
        for param_name, param_config in schema.items():
            if param_name in config:
                value = config[param_name]
                type_valid = True
                
                # 类型检查
                if param_config['type'] == 'float' and not isinstance(value, (int, float)):
                    errors.append(f"{param_name} 必须是数值类型，当前类型: {type(value)}")
                    type_valid = False
                elif param_config['type'] == 'integer' and not isinstance(value, int):
                    errors.append(f"{param_name} 必须是整数类型，当前类型: {type(value)}")
                    type_valid = False
                elif param_config['type'] == 'boolean' and not isinstance(value, bool):
                    errors.append(f"{param_name} 必须是布尔类型，当前类型: {type(value)}")
                    type_valid = False
                elif param_config['type'] == 'string' and not isinstance(value, str):
                    errors.append(f"{param_name} 必须是字符串类型，当前类型: {type(value)}")
                    type_valid = False
                
                # 只有在类型检查通过后才进行范围检查
                if type_valid:
                    # 范围检查
                    if 'min' in param_config and value < param_config['min']:
                        errors.append(f"{param_name} 不能小于 {param_config['min']}，当前值: {value}")
                    if 'max' in param_config and value > param_config['max']:
                        errors.append(f"{param_name} 不能大于 {param_config['max']}，当前值: {value}")
                    
                    # 选项检查（仅对字符串类型）
                    if param_config['type'] == 'string' and 'options' in param_config:
                        if value not in param_config['options']:
                            errors.append(f"{param_name} 必须是以下选项之一: {param_config['options']}，当前值: {value}")
        
        return len(errors) == 0, errors
