from typing import Any, List

import torch
from tgb.linkproppred.dataset import LinkPropPredDataset

from opendg.events import EdgeEvent, Event
from opendg.timedelta import TimeDeltaDG

TIME_DELTA_DICT = {
    'tgbl-wiki': TimeDeltaDG('s'),
    'tgbl-subreddit': TimeDeltaDG('r'),
    'tgbl-lastfm': TimeDeltaDG('r'),
    'tgbl-review': TimeDeltaDG('s'),
    'tgbl-coin': TimeDeltaDG('r'),
    'tgbl-flight': TimeDeltaDG('r'),
    'tgbl-comment': TimeDeltaDG('r'),
    'tgbn-trade': TimeDeltaDG('r'),
    'tgbn-genre': TimeDeltaDG('r'),
    'tgbn-reddit': TimeDeltaDG('r'),
    'tgbn-token': TimeDeltaDG('r'),
    'tkgl-polecat': TimeDeltaDG('r'),
    'tkgl-icews': TimeDeltaDG('r'),
    'tkgl-yago': TimeDeltaDG('r'),
    'tkgl-wikidata': TimeDeltaDG('r'),
    'tkgl-smallpedia': TimeDeltaDG('r'),
    'thgl-myket': TimeDeltaDG('r'),
    'thgl-github': TimeDeltaDG('r'),
    'thgl-forum': TimeDeltaDG('r'),
    'thgl-software': TimeDeltaDG('r'),
}


def read_tgb(
    name: str,
    split: str = 'all',  # Options: 'train', 'valid', 'test', 'all'
    **kwargs: Any,
) -> List[Event]:
    # TODO: Node Events not supported
    if name.startswith('tgbl-'):
        dataset = LinkPropPredDataset(name=name, **kwargs)
    elif name.startswith('tgbn-'):
        raise ValueError(f'Not Implemented dataset: {name}')
    else:
        raise ValueError(f'Unknown dataset: {name}')
    data = dataset.full_data

    split_masks = {
        'train': dataset.train_mask,
        'valid': dataset.val_mask,
        'test': dataset.test_mask,
    }

    if split == 'all':
        mask = slice(None)  # selects everything
    elif split in split_masks:
        mask = split_masks[split]
    else:
        raise ValueError(f'Unknown split: {split}')

    sources = data['sources'][mask]
    destinations = data['destinations'][mask]
    timestamps = data['timestamps'][mask]
    edge_feats = data['edge_feat'][mask] if data['edge_feat'] is not None else None

    events: List[Event] = []
    for i, (src, dst, t, feat) in enumerate(
        zip(
            sources,
            destinations,
            timestamps,
            edge_feats if edge_feats is not None else [None] * len(sources),
        )
    ):
        features = torch.tensor(feat, dtype=torch.float) if feat is not None else None
        event = EdgeEvent(
            t=int(t), src=int(src), dst=int(dst), global_id=i, features=features
        )
        events.append(event)

    return events
