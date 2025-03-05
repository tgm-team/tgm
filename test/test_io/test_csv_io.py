import tempfile

import pytest
import torch

from opendg._io import read_csv, write_csv
from opendg.events import EdgeEvent


def test_csv_conversion_no_features():
    events = [
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=5, edge=(10, 20)),
    ]

    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}
    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, **col_names)
        recovered_events = read_csv(f.name, **col_names)

    _assert_events_equal(events, recovered_events)


def test_csv_conversion_with_features():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(5)),
    ]

    edge_feature_col = [f'dim_{i}' for i in range(5)]
    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}

    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, edge_feature_col=edge_feature_col, **col_names)
        recovered_events = read_csv(
            f.name, edge_feature_col=edge_feature_col, **col_names
        )

    _assert_events_equal(events, recovered_events)


def test_csv_conversion_with_features_no_feature_cols_provided():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(1, 3, 37)),
    ]
    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(events, f.name, **col_names)


def test_csv_conversion_with_features_bad_feature_col_shape():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(37)),
    ]
    edge_feature_col = [f'dim_{i}' for i in range(3)]  # Bad: should have 37 names
    col_names = {'src_col': 'src_id', 'dst_col': 'dst_id', 'time_col': 'time'}
    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(
                events, f.name, edge_feature_col=edge_feature_col, **col_names
            )


def _assert_events_equal(expected_events, actual_events):
    assert len(expected_events) == len(actual_events)
    for i in range(len(expected_events)):
        expected_event = expected_events[i]
        actual_event = actual_events[i]

        assert isinstance(expected_event, EdgeEvent)
        assert isinstance(actual_event, EdgeEvent)

        assert expected_event.time == actual_event.time
        assert expected_event.edge == actual_event.edge
        torch.testing.assert_close(expected_event.features, actual_event.features)
