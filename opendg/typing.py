from typing import Dict, List, Tuple

from opendg.timedelta import TimeDeltaTG

Event = Tuple[int, int, int]  # Timestamp, Source Vertex, Target Vertex

EventsDict = Dict[int, List[Tuple[int, int]]]  # Timestamp -> List of edges

