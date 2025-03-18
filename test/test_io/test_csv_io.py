import tempfile

import pytest
import torch

from opendg._io import read_csv, write_csv
from opendg.events import EdgeEvent


def test_csv_conversion_no_features():
    events = [
        EdgeEvent(t=1, src=2, dst=3),
        EdgeEvent(t=1, src=10, dst=20),
    ]

    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, **col_names)
        recovered_events = read_csv(f.name, **col_names)
    assert events == recovered_events


def test_csv_conversion_with_features():
    events = [
        EdgeEvent(t=1, src=2, dst=3, msg=torch.rand(5)),
        EdgeEvent(t=5, src=10, dst=20, msg=torch.rand(5)),
    ]

    edge_feature_col = [f'dim_{i}' for i in range(5)]
    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, edge_feature_col=edge_feature_col, **col_names)
        recovered_events = read_csv(
            f.name, edge_feature_col=edge_feature_col, **col_names
        )

    assert len(events) == len(recovered_events)
    for i in range(len(recovered_events)):
        assert isinstance(events[i], EdgeEvent)
        assert isinstance(recovered_events[i], EdgeEvent)
        assert events[i].t == recovered_events[i].t
        assert events[i].edge == recovered_events[i].edge
        torch.testing.assert_close(events[i].msg, recovered_events[i].msg)


def test_csv_conversion_with_features_no_feature_cols_provided():
    events = [
        EdgeEvent(t=1, src=2, dst=3, msg=torch.rand(5)),
        EdgeEvent(t=5, src=10, dst=20, msg=torch.rand(5)),
    ]
    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(events, f.name, **col_names)


def test_csv_conversion_with_features_bad_feature_col_shape():
    events = [
        EdgeEvent(t=1, src=2, dst=3, msg=torch.rand(5)),
        EdgeEvent(t=5, src=10, dst=20, msg=torch.rand(5)),
    ]
    edge_feature_col = [f'dim_{i}' for i in range(3)]  # Bad: should have 5 names
    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(
                events, f.name, edge_feature_col=edge_feature_col, **col_names
            )
