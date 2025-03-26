from unittest.mock import MagicMock, patch

import numpy as np

from opendg._io import read_tgb
from opendg.events import EdgeEvent


@patch('opendg._io.tgb.LinkPropPredDataset')
def test_tgb_conversion(mock_dataset_cls):
    num_events = 157474

    # Simulate large mock data
    sources = np.random.randint(0, 1000, size=num_events)
    destinations = np.random.randint(0, 1000, size=num_events)
    timestamps = np.random.randint(1000000000, 2000000000, size=num_events)
    edge_feat = None  # or np.random.rand(num_events, feat_dim)

    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.full_data = {
        'sources': sources,
        'destinations': destinations,
        'timestamps': timestamps,
        'edge_feat': edge_feat,
    }
    mock_dataset.train_mask = np.ones(num_events, dtype=bool)
    mock_dataset.val_mask = np.zeros(num_events, dtype=bool)
    mock_dataset.test_mask = np.zeros(num_events, dtype=bool)
    mock_dataset.num_edges = num_events

    mock_dataset_cls.return_value = mock_dataset

    events = read_tgb(name='tgbl-wiki', split='all')

    assert len(events) == num_events
    for i in range(5):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].t == int(timestamps[i])
        assert events[i].edge == (int(sources[i]), int(destinations[i]))

    # Confirm correct call
    mock_dataset_cls.assert_called_once_with(name='tgbl-wiki')
