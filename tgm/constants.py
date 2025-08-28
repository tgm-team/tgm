from typing import Final

PADDED_NODE_ID: Final[int] = -1
"""Sentinel node ID used to mark invalid or padded neighbors in a graph.

Notes:
    - This value should **not** appear in actual node IDs in edge or node event tensors.
"""
