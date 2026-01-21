"""Aggregation Strategies for Federated Learning"""

from tfl_hpl.aggregation.median_aggregation import MedianAggregator
from tfl_hpl.aggregation.robust_aggregation import RobustAggregator

__all__ = [
    "MedianAggregator",
    "RobustAggregator",
]
