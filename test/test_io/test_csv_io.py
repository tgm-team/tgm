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

    with tempfile.NamedTemporaryFile() as f:
        write_csv(
            events, f.name, src_node_id_col='foo', dst_node_id_col='bar', time_col='baz'
        )
        recovered_events = read_csv(
            f.name, src_node_id_col='foo', dst_node_id_col='bar', time_col='baz'
        )

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(len(events)):
        # Only works for edge events
        assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        assert events[i].features == recovered_events[i].features


@pytest.mark.skip('Not implemented')
def test_csv_conversion_with_features():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(3, 37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(3, 37)),
    ]

    with tempfile.NamedTemporaryFile() as f:
        write_csv(events, f.name, feature_cols=['dim_0', 'dim_1', 'dim_2'])
        recovered_events = read_csv(f.name, feature_cols=['dim_0', 'dim_1', 'dim_2'])

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(len(events)):
        # Only works for edge events
        assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        assert events[i].features == recovered_events[i].features


@pytest.mark.skip('Not implemented')
def test_csv_conversion_with_features_custom_cols():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(1, 3, 37)),
    ]

    with tempfile.NamedTemporaryFile() as f:
        write_csv(
            events,
            f.name,
            src_node_id_col='foo',
            dst_node_id_col='bar',
            time_col='baz',
            feature_cols=['dim_0', 'dim_1', 'dim_2'],
        )
        recovered_events = read_csv(
            f.name,
            src_node_id_col='foo',
            dst_node_id_col='bar',
            time_col='baz',
            feature_cols=['dim_0', 'dim_1', 'dim_2'],
        )

    # Need an event __eq__
    assert len(events) == len(recovered_events)
    for i in range(len(events)):
        # Only works for edge events
        assert events[i].time == recovered_events[i].time
        assert events[i].edge == recovered_events[i].edge
        assert events[i].features == recovered_events[i].features


def test_csv_conversion_with_features_no_feature_cols_provided():
    events = [
        EdgeEvent(time=1, edge=(2, 3), features=torch.rand(1, 3, 37)),
        EdgeEvent(time=5, edge=(10, 20), features=torch.rand(1, 3, 37)),
    ]

    with tempfile.NamedTemporaryFile() as f:
        with pytest.raises(ValueError):
            _ = write_csv(events, f.name)
