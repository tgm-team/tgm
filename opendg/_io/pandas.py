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
    _check_pandas_import()
    # TODO: Node Events not supported

    # Pre-allocating buffer of events with the right size, and adding an index column
    # into the dataframe, so that the df.apply() call below is thread safe, and can
    # be run without the GIL. This index ensures that everything is processed in order.
    _mock_event = EdgeEvent(t=-1, src=-1, dst=-1)
    events: List[Event] = [_mock_event] * len(df)
    df['index'] = np.arange(len(df))

    def _construct_event_from_row(row: pd.Series) -> None:
        src = int(row[src_col])
        dst = int(row[dst_col])
        t = int(row[time_col])
        i = int(row['index'])

        if edge_feature_col is not None:
            features = torch.tensor(row[edge_feature_col])
        else:
            features = None

        events[i] = EdgeEvent(t=t, src=src, dst=dst, global_id=i, features=features)

    df.apply(_construct_event_from_row, axis=1)
    df.drop('index', axis=1)  # Clean up temporary index column
    return events


def _check_pandas_import(min_version_number: Optional[str] = None) -> None:
    try:
        import pandas

        user_pandas_version = pandas.__version__
    except ImportError:
        user_pandas_version = None

    err_msg = 'User requires pandas '
    if min_version_number is not None:
        err_msg += f'>={min_version_number} '
    err_msg += 'to initialize a DGraph using DGraph.from_pandas()'

    if user_pandas_version is None:
        raise ImportError(err_msg)
    elif min_version_number is not None and user_pandas_version < min_version_number:
        err_msg += f', found pandas=={user_pandas_version} < {min_version_number}'
        raise ImportError(err_msg)
