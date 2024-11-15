""" OpenDG library for fast ML on temporal graphs."""

from .data.data import BaseData,CTDG,DTDG
from .data.storage import EventStore

__version__ = '0.1.0'

__all__ = [
    'BaseData',
    '__version__',
]
