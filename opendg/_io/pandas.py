from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from opendg.events import EdgeEvent, Event


def read_pandas(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    time_col: str,
    edge_feature_col: Optional[str] = None,
) -> List[Event]:
    # TODO: Node Events not supported

    # Pre-allocating buffer of events with the right size, and adding an index column
    # into the dataframe, so that the df.apply() call below is thread safe, and can
    # be run without the GIL. This index ensures that everything is processed in order.
    _mock_event = EdgeEvent(time=-1, edge=(-1, -1))
    events: List[Event] = [_mock_event] * len(df)
    df['index'] = np.arange(len(df))

    def _construct_event_from_row(row: pd.Series) -> None:
        src_id = int(row[src_col])
        dst_id = int(row[dst_col])
        time = int(row[time_col])
        idx = int(row['index'])

        if edge_feature_col is not None:
            edge_features = torch.tensor(row[edge_feature_col])
        else:
            edge_features = None

        events[idx] = EdgeEvent(time, (src_id, dst_id), edge_features)

    df.apply(_construct_event_from_row, axis=1)
    df.drop('index', axis=1)  # Clean up temporary index column

    return events
