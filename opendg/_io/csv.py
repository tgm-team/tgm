import csv
import pathlib
from typing import List, Optional

import torch

from opendg.data import DGData


def read_csv(
    file_path: str | pathlib.Path,
    src_col: str,
    dst_col: str,
    time_col: str,
    edge_feature_col: Optional[List[str]] = None,
) -> DGData:
    # TODO: Node Events not supported

    file_path = str(file_path) if isinstance(file_path, pathlib.Path) else file_path
    with open(file_path, newline='') as f:
        reader = list(csv.DictReader(f))  # Assumes the whole things fits in memory
        num_edges = len(reader)

    edge_index = torch.empty((num_edges, 2), dtype=torch.long)
    timestamps = torch.empty(num_edges, dtype=torch.long)
    edge_feats = None
    if edge_feature_col is not None:
        edge_feats = torch.empty((num_edges, len(edge_feature_col)))

    for i, row in enumerate(reader):
        edge_index[i, 0] = int(row[src_col])
        edge_index[i, 1] = int(row[dst_col])
        timestamps[i] = int(row[time_col])
        if edge_feature_col is not None:
            # This is likely better than creating a tensor copy for every event
            for j, col in enumerate(edge_feature_col):
                edge_feats[i, j] = float(row[col])  # type: ignore

    return DGData(edge_index, timestamps, edge_feats)
