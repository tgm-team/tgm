"""OpenDG library for fast ML on temporal graphs."""

from .data import CTDG, DTDG, BaseData

__version__ = '0.1.0'

__all__ = [
    'BaseData',
    'CTDG',
    'DTDG',
    '__version__',
]
