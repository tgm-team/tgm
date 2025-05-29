import csv
import pathlib
from typing import List, Optional

from torch import Tensor

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
    edge_index_, timestamps_, edge_features_ = [], [], []
    with open(file_path, newline='') as f:
        for row in csv.DictReader(f):
            src = int(row[src_col])
            dst = int(row[dst_col])
            t = int(row[time_col])

            edge_index_.append([src, dst])
            timestamps_.append(t)

            if edge_feature_col is not None:
                features = [float(row[feature_col]) for feature_col in edge_feature_col]
                edge_features_.append(features)

    edge_index = Tensor(edge_index_)
    timestamps = Tensor(timestamps_)
    edge_features = Tensor(edge_features_) if edge_feature_col is not None else None
    return DGData(edge_index, timestamps, edge_features)
