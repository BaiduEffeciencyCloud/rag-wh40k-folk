import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

class ConfigError(Exception):
    pass

def load_config(config_path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    _validate_config(config)
    return config

def _validate_config(config: dict):
    # 基础字段校验
    required_fields = [
        'intent_data_path', 'slot_data_path', 'test_size', 'val_size', 'random_state',
        'intent_feature', 'slot_feature', 'classifier_type', 'model_path', 'uncertainty'
    ]
    for field in required_fields:
        if field not in config:
            raise ConfigError(f"缺少必需配置项: {field}")
    # 类型校验
    if not isinstance(config['test_size'], float) or not (0 < config['test_size'] < 1):
        raise ConfigError('test_size 必须为0-1之间的小数')
    if not isinstance(config['val_size'], float) or not (0 <= config['val_size'] < 1):
        raise ConfigError('val_size 必须为0-1之间的小数')
    if not isinstance(config['random_state'], int):
        raise ConfigError('random_state 必须为整数')
    # 其他可扩展校验
    # ... 