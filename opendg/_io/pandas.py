from typing import Optional

import pandas as pd
import torch

from opendg.data import DGData


def read_pandas(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    time_col: str,
    edge_feature_col: Optional[str] = None,
) -> DGData:
    _check_pandas_import()
    # TODO: Node Events not supported

    edge_index = torch.from_numpy(df[[src_col, dst_col]].to_numpy()).long()
    timestamps = torch.from_numpy(df[time_col].to_numpy()).long()
    if edge_feature_col is None:
        edge_features = None
    else:
        edge_features = torch.from_numpy(df[edge_feature_col].to_numpy())
    return DGData(edge_index, timestamps, edge_features)


def _check_pandas_import(min_version_number: Optional[str] = None) -> None:
    try:
        import pandas

        user_pandas_version = pandas.__version__
    except ImportError:
        user_pandas_version = None

    err_msg = 'User requires pandas '
    if min_version_number is not None:
        err_msg += f'>={min_version_number} '
    err_msg += 'to initialize a DGraph using DGraph.from_pandas()'

    if user_pandas_version is None:
        raise ImportError(err_msg)
    elif min_version_number is not None and user_pandas_version < min_version_number:
        err_msg += f', found pandas=={user_pandas_version} < {min_version_number}'
        raise ImportError(err_msg)
