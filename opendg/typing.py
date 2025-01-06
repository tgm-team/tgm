from typing import Dict, List, Tuple

Event = Tuple[int, int, int]  # Timestamp, Source Vertex, Target Vertex

EventsDict = Dict[int, List[Tuple[int, int]]]  # Timestamp -> List of edges
