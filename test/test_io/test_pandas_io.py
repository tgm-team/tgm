import pandas as pd
import pytest
import torch

from opendg._io import read_pandas
from opendg.events import EdgeEvent


def test_read_pandas_no_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
    }
    events_df = pd.DataFrame(events_dict)

    events = read_pandas(events_df, src_col='src', dst_col='dst', time_col='t')
    assert len(events) == len(events_df)
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].global_id == i
        assert events[i].t == events_df.t.iloc[i]
        assert events[i].edge == (events_df.src.iloc[i], events_df.dst.iloc[i])


def test_read_pandas_with_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    events = read_pandas(
        events_df,
        src_col='src',
        dst_col='dst',
        time_col='t',
        edge_feature_col='edge_features',
    )
    assert len(events) == len(events_df)
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].global_id == i
        assert events[i].t == events_df.t.iloc[i]
        assert events[i].edge == (events_df.src.iloc[i], events_df.dst.iloc[i])
        torch.testing.assert_close(
            events[i].features.tolist(), events_df.edge_features.iloc[i]
        )


@pytest.mark.parametrize(
    'bad_col_type', ['src_col', 'dst_col', 'time_col', 'edge_feature_col']
)
def test_read_pandas_bad_col_name(bad_col_type):
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    read_pandas_kwargs = {
        'src_col': 'src',
        'dst_col': 'dst',
        'time_col': 't',
        'edge_feature_col': 'edge_features',
    }
    read_pandas_kwargs[bad_col_type] = 'Mock Bad Column Name'

    with pytest.raises(KeyError):
        _ = read_pandas(events_df, **read_pandas_kwargs)
