from .agginterface import AggregationInterface
from typing import Dict, Any
from .dedup_agg import DedupAggregation

class AggFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, agg_cls):
        cls._registry[name] = agg_cls

    @classmethod
    def get_aggregator(cls, name: str) -> AggregationInterface:
        if name not in cls._registry:
            raise ValueError(f"未注册聚合策略: {name}")
        return cls._registry[name]()

AggFactory.register('simple', DedupAggregation) 