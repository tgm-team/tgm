import pandas as pd
import pytest
import torch

from opendg._io import read_pandas
from opendg.data import DGData


def test_read_pandas_no_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
    }
    events_df = pd.DataFrame(events_dict)

    data = read_pandas(events_df, src_col='src', dst_col='dst', time_col='t')
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [1337, 1338]


def test_read_pandas_with_edge_features():
    events_dict = {
        'src': [2, 10],
        'dst': [3, 20],
        't': [1337, 1338],
        'edge_features': [torch.rand(5).tolist(), torch.rand(5).tolist()],
    }
    events_df = pd.DataFrame(events_dict)

    data = read_pandas(
        events_df,
        src_col='src',
        dst_col='dst',
        time_col='t',
        edge_feature_col='edge_features',
    )
    assert isinstance(data, DGData)
    assert data.edge_index.tolist() == [[2, 3], [10, 20]]
    assert data.timestamps.tolist() == [1337, 1338]
    torch.testing.assert_close(
        data.edge_feats.tolist(), events_df.edge_features.tolist()
    )


@pytest.mark.skip('TODO: Add node features to IO')
def test_read_pandas_with_node_events():
    pass


@pytest.mark.skip('TODO: Add node features to IO')
def test_read_pandas_with_node_features():
    pass
