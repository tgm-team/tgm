import tempfile

import pytest
import torch

from opendg._events import EdgeEvent
from opendg._io import read_csv, write_csv


def test_csv_conversion_no_features():
    events = [
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=5, edge=(10, 20)),
    ]

    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name)
        recovered_events = read_csv(f.name)

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(len(events)):
        # Only works for edge events
        assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        assert events[i].features == recovered_events[i].features


def test_csv_conversion_no_features_custom_cols():
    events = [
        EdgeEvent(time=1, edge=(2, 3)),
        EdgeEvent(time=5, edge=(10, 20)),
    ]

    col_names = {'src_node_id_col': 'foo', 'dst_node_id_col': 'bar', 'time_col': 'baz'}

    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, **col_names)
        recovered_events = read_csv(f.name, **col_names)

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(len(events)):
        # Only works for edge events
        assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        assert events[i].features == recovered_events[i].features


def test_csv_conversion_with_features():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(5)),
    ]

    feature_cols = [f'dim_{i}' for i in range(5)]

    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, feature_cols=feature_cols)
        recovered_events = read_csv(f.name, feature_cols=feature_cols)

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(len(events)):
        # Only works for edge events
        assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        torch.testing.assert_close(events[i].features, recovered_events[i].features)


def test_csv_conversion_with_features_custom_cols():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(5)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(5)),
    ]

    col_names = {'src_node_id_col': 'foo', 'dst_node_id_col': 'bar', 'time_col': 'baz'}
    feature_cols = [f'dim_{i}' for i in range(5)]

    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, feature_cols=feature_cols, **col_names)
        recovered_events = read_csv(f.name, feature_cols=feature_cols, **col_names)

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(
        len(events)
    ):  # Only works for edge events assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        torch.testing.assert_close(events[i].features, recovered_events[i].features)


def test_csv_conversion_with_features_no_feature_cols_provided():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(1, 3, 37)),
    ]

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(events, f.name)


def test_csv_conversion_with_features_bad_feature_col_shape():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(37)),
    ]
    feature_cols = [f'dim_{i}' for i in range(3)]  # Bad: should have 37 names
    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(events, f.name, feature_cols=feature_cols)
