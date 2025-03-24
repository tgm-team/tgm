from tgb.linkproppred.dataset import LinkPropPredDataset

from opendg._io.tgb import read_tgb
from opendg.events import EdgeEvent


def test_tgb_conversion():
    name = 'tgbl-wiki'
    dataset = LinkPropPredDataset(name=name, root='datasets', preprocess=True)
    data = dataset.full_data
    sources = data['sources']
    destinations = data['destinations']
    timestamps = data['timestamps']

    events = read_tgb(name=name, split='all')
    assert len(events) == dataset.num_edges
    for i in range(len(events)):
        assert isinstance(events[i], EdgeEvent)
        assert events[i].t == int(timestamps[i])
        assert events[i].edge == (int(sources[i]), int(destinations[i]))
