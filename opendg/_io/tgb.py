from typing import Any

import numpy as np
import torch
from tgb.linkproppred.dataset import LinkPropPredDataset

from opendg.data import DGData
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
) -> DGData:
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

    src = data['sources'][mask]
    dst = data['destinations'][mask]
    edge_index = torch.from_numpy(np.stack([src, dst], axis=1)).long()
    timestamps = torch.from_numpy(data['timestamps'][mask]).long()
    if data['edge_feat'] is None:
        edge_feats = None
    else:
        edge_feats = torch.from_numpy(data['edge_feat'][mask])
    return DGData(edge_index, timestamps, edge_feats)
