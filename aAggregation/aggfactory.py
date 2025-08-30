from .agginterface import AggregationInterface
from typing import Dict, Any
from .agg import DynamicAggregation

class AggFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, agg_cls):
        cls._registry[name] = agg_cls

    @classmethod
    def get_aggregator(cls, name: str = 'simple', template: str = 'default') -> AggregationInterface:
        if name not in cls._registry:
            raise ValueError(f"未注册聚合策略: {name}")
        # 兼容旧接口: 若聚合器支持template参数则传递，否则无参数
        agg_cls = cls._registry[name]
        try:
            return agg_cls(template=template)
        except TypeError:
            return agg_cls()

AggFactory.register('default', DynamicAggregation)
AggFactory.register('query', DynamicAggregation)
AggFactory.register('list', DynamicAggregation)
AggFactory.register('compare', DynamicAggregation)
AggFactory.register('rule', DynamicAggregation) 