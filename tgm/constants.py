from typing import Final

PADDED_NODE_ID: Final[int] = -1
"""Sentinel node ID used to mark invalid or padded neighbors in a graph.

Notes:
    - A tgm.exceptions.InvalidNodeIDError will be thrown if this value
    appears as an ID in edge or node event tensors will constructing DGData.
"""

DEFAULT_KEY_HOOK_MANAGER: Final[str] = '_SHARED_HOOK'
