from typing import List, Optional

import pandas as pd
import torch

from opendg.events import EdgeEvent, Event


def read_pandas(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    time_col: Optional[str] = None,
    edge_feature_col: Optional[str] = None,
) -> List[Event]:
    events: List[Event] = []

    def _construct_event_from_row(row: pd.Series) -> None:
        src_id = row[src_col]
        dst_id = row[dst_col]

        if time_col is not None:
            time = row[time_col]
        else:
            time = len(events)

        if edge_feature_col is not None:
            edge_features = torch.tensor(row[edge_feature_col])
        else:
            edge_features = None

        event = EdgeEvent(time, (src_id, dst_id), edge_features)
        events.append(event)

    # Note: This is not thread-safe. Should not be run without the GIL
    df.apply(_construct_event_from_row, axis=1)

    return events
