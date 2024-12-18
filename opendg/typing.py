import os
import pathlib
from typing import Tuple, Union

FileType = Union[str, os.PathLike, pathlib.Path]

Event = Tuple[int, int, int]  # Source Vertex, Target Vertex, Time

Snapshot = int
