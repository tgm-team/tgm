from typing import Final, List

PADDED_NODE_ID: Final[int] = -1
"""Sentinel node ID used to mark invalid or padded neighbors in a graph.

Notes:
    - A tgm.exceptions.InvalidNodeIDError will be thrown if this value
    appears as an ID in edge or node event tensors will constructing DGData.
"""

RECIPE_TGB_LINK_PRED: Final[str] = 'TGB_LINKPROPPRED_SETTING'
"""Recipe identifier for TGB link property prediction task.
"""

SUPPORTED_RECIPES: Final[List[str]] = [RECIPE_TGB_LINK_PRED]
"""A list of supported recipes that can be built from HookManager.

Notes:
    - A tgm.exceptions.UnsupportRecipe will be thrown if user build recipe
    from unsupported recipe
"""

METRIC_TGB_LINKPROPPRED: Final[str] = 'mrr'
"""Official metric used in TGB link prediction task.
"""
