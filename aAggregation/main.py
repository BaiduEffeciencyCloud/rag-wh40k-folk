from .aggfactory import AggFactory
from typing import List, Dict, Any

def aggregate_results(results: List[Dict[str, Any]], agg_type: str = 'dedup', params: Dict[str, Any] = None) -> Dict[str, Any]:
    aggregator = AggFactory.get_aggregator(agg_type)
    return aggregator.aggregate(results, params) 