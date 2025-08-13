from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

from tgm import DGraph
from tgm.timedelta import TimeDeltaDG


@dataclass(frozen=True)
class DGDataset:
    r"""Immutable container storing some combination of train, validation and test DGraphs."""

    train: DGraph
    val: Optional[DGraph] = None
    test: Optional[DGraph] = None


@dataclass(frozen=True)
class TGBDataset(DGDataset):
    name: str = ''

    TGB_TIME_DELTAS: ClassVar[Dict[str, TimeDeltaDG]] = {
        'tgbl-wiki': TimeDeltaDG('s'),
        'tgbl-subreddit': TimeDeltaDG('s'),
        'tgbl-lastfm': TimeDeltaDG('s'),
        'tgbl-review': TimeDeltaDG('s'),
        'tgbl-coin': TimeDeltaDG('s'),
        'tgbl-flight': TimeDeltaDG('s'),
        'tgbl-comment': TimeDeltaDG('s'),
        'tgbn-trade': TimeDeltaDG('Y'),
        'tgbn-genre': TimeDeltaDG('s'),
        'tgbn-reddit': TimeDeltaDG('s'),
        'tgbn-token': TimeDeltaDG('s'),
    }
