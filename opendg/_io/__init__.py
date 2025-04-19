from opendg._io.csv import read_csv
from opendg._io.pandas import read_pandas
from opendg._io.tgb import read_tgb, TIME_DELTA_DICT

from typing import Any, List, Union
import pathlib

import pandas as pd
from opendg.timedelta import TimeDeltaDG
from opendg.events import Event


def read_events(
    data: Union[str, pathlib.Path, pd.DataFrame], **kwargs: Any
) -> List[Event]:
    if isinstance(data, pd.DataFrame):
        return read_pandas(data, **kwargs)

    if isinstance(data, (str, pathlib.Path)):
        data_str = str(data)

        if data_str.startswith('tgbl-'):
            return read_tgb(name=data_str, **kwargs)
        if data_str.endswith('.csv'):
            return read_csv(data, **kwargs)

        raise ValueError(f'Unsupported file format or dataset identifier: {data_str}')

    raise ValueError(f'Cannot read events from type {type(data).__name__}')


def read_time_delta(
    name: str,
) -> TimeDeltaDG:
    if name.startswith('tgbl-'):
        return TIME_DELTA_DICT[name]
    else:
        return TimeDeltaDG('r')
