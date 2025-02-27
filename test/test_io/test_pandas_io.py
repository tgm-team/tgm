import pandas as pd
import pytest
import torch

from opendg._io import read_pandas
from opendg.events import EdgeEvent


def test_read_pandas_no_features_no_features_no_timestamp():
    events_dict = {
        'src_id': [2, 10],
        'dst_id': [3, 20],
    }
    events_df = pd.DataFrame(events_dict)

    events = read_pandas(events_df, src_col='src_id', dst_col='dst_id')
    assert len(events) == len(events_df)
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].time == i
        assert events[i].edge == (events_df.src_id.iloc[i], events_df.dst_id.iloc[i])


def test_read_pandas_no_features_with_timestamp():
    events_dict = {
        'src_id': [2, 10],
        'dst_id': [3, 20],
        'time': [1337, 1338],
    }
    events_df = pd.DataFrame(events_dict)

    events = read_pandas(events_df, src_col='src_id', dst_col='dst_id', time_col='time')
    assert len(events) == len(events_df)
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].time == events_df.time.iloc[i]
        assert events[i].edge == (events_df.src_id.iloc[i], events_df.dst_id.iloc[i])


def test_read_pandas_with_features_no_timestamp():
    events_dict = {
        'src_id': [2, 10],
        'dst_id': [3, 20],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    events = read_pandas(
        events_df, src_col='src_id', dst_col='dst_id', edge_feature_col='edge_features'
    )
    assert len(events) == len(events_df)
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].time == i
        assert events[i].edge == (events_df.src_id.iloc[i], events_df.dst_id.iloc[i])
        torch.testing.assert_close(
            events[i].features.tolist(), events_df.edge_features.iloc[i]
        )


def test_read_pandas_with_features_with_timestamp():
    events_dict = {
        'src_id': [2, 10],
        'dst_id': [3, 20],
        'time': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    events = read_pandas(
        events_df,
        src_col='src_id',
        dst_col='dst_id',
        edge_feature_col='edge_features',
        time_col='time',
    )
    assert len(events) == len(events_df)
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].time == events_df.time.iloc[i]
        assert events[i].edge == (events_df.src_id.iloc[i], events_df.dst_id.iloc[i])
        torch.testing.assert_close(
            events[i].features.tolist(), events_df.edge_features.iloc[i]
        )


@pytest.mark.parametrize(
    'bad_col_type', ['src_col', 'dst_col', 'time_col', 'edge_feature_col']
)
def test_read_pandas_bad_col_name(bad_col_type):
    events_dict = {
        'src_id': [2, 10],
        'dst_id': [3, 20],
        'time': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    read_pandas_kwargs = {
        'src_col': 'src_id',
        'dst_col': 'dst_id',
        'time_col': 'time',
        'edge_feature_col': 'edge_features',
    }
    read_pandas_kwargs[bad_col_type] = 'Mock Bad Column Name'

    with pytest.raises(KeyError):
        _ = read_pandas(events_df, **read_pandas_kwargs)
