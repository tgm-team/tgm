from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from opendg._io import read_tgb
from opendg.events import EdgeEvent

# Constants for split sizes
num_events = 157474
num_train = 110232
num_val = 23621
num_test = 23621

# Index boundaries for each split
train_indices = np.arange(0, num_train)
val_indices = np.arange(num_train, num_train + num_val)
test_indices = np.arange(num_train + num_val, num_events)


@pytest.mark.parametrize(
    'split,expected_indices',
    [
        ('train', train_indices),
        ('valid', val_indices),
        ('test', test_indices),
        ('all', np.arange(num_events)),
    ],
)
@patch('opendg._io.tgb.LinkPropPredDataset')
def test_tgb_conversion(mock_dataset_cls, split, expected_indices):
    sources = np.random.randint(0, 1000, size=num_events)
    destinations = np.random.randint(0, 1000, size=num_events)
    timestamps = np.random.randint(1_000_000_000, 2_000_000_000, size=num_events)
    edge_feat = None

    train_mask = np.zeros(num_events, dtype=bool)
    train_mask[train_indices] = True

    val_mask = np.zeros(num_events, dtype=bool)
    val_mask[val_indices] = True

    test_mask = np.zeros(num_events, dtype=bool)
    test_mask[test_indices] = True

    mock_dataset = MagicMock()
    mock_dataset.full_data = {
        'sources': sources,
        'destinations': destinations,
        'timestamps': timestamps,
        'edge_feat': edge_feat,
    }
    mock_dataset.train_mask = train_mask
    mock_dataset.val_mask = val_mask
    mock_dataset.test_mask = test_mask
    mock_dataset.num_edges = num_events

    mock_dataset_cls.return_value = mock_dataset

    # Run the function
    events = read_tgb(name='tgbl-wiki', split=split)

    # Assertions
    assert len(events) == len(expected_indices)

    for i, idx in enumerate(expected_indices[:5]):  # sample a few for sanity check
        assert isinstance(events[i], EdgeEvent)
        assert events[i].t == int(timestamps[idx])
        assert events[i].edge == (int(sources[idx]), int(destinations[idx]))

    # Confirm correct dataset instantiation
    mock_dataset_cls.assert_called_once_with(name='tgbl-wiki')
