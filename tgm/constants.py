from typing import Final

PADDED_NODE_ID: Final[int] = -1
"""Sentinel node ID used to mark invalid or padded neighbors in a graph.

Notes:
    - A tgm.exceptions.InvalidNodeIDError will be thrown if this value
    appears as an ID in edge or node event tensors will constructing DGData.
"""

RECIPE_TGB_LINK_PRED: Final[str] = 'TGB_LINK_PROPERTY_PREDICTION'
"""Recipe identifier for TGB link property prediction task.
"""

METRIC_TGB_LINKPROPPRED: Final[str] = 'mrr'
"""Official metric used in TGB link prediction task.
"""

METRIC_TGB_NODEPROPPRED: Final[str] = 'ndcg'
"""Official metric used in TGB node prediction task.
"""
