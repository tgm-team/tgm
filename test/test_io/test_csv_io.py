import csv
import tempfile

import torch

from opendg._io import read_csv
from opendg.data import DGData


def test_csv_conversion_no_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 1])
    data = DGData(edge_index, timestamps)

    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        _write_csv(data, f.name, **col_names)
        recovered_data = read_csv(f.name, **col_names)

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
    assert recovered_data.edge_feats is None
    assert recovered_data.node_feats is None


def test_csv_conversion_with_features():
    edge_index = torch.LongTensor([[2, 3], [10, 20]])
    timestamps = torch.LongTensor([1, 5])
    edge_feats = torch.rand(2, 5)
    data = DGData(edge_index, timestamps, edge_feats)

    edge_feature_col = [f'dim_{i}' for i in range(5)]
    col_names = {'src_col': 'src', 'dst_col': 'dst', 'time_col': 't'}
    with tempfile.NamedTemporaryFile() as f:
        _write_csv(data, f.name, edge_feature_col=edge_feature_col, **col_names)
        recovered_data = read_csv(
            f.name, edge_feature_col=edge_feature_col, **col_names
        )

    torch.testing.assert_close(data.edge_index, recovered_data.edge_index)
    torch.testing.assert_close(data.timestamps, recovered_data.timestamps)
    torch.testing.assert_close(data.edge_feats, recovered_data.edge_feats)
    assert recovered_data.node_feats is None


def _write_csv(data, fp, src_col, dst_col, time_col, edge_feature_col=None):
    with open(fp, 'w', newline='') as f:
        fieldnames = [src_col, dst_col, time_col]
        if edge_feature_col is not None:
            fieldnames += edge_feature_col
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(data.edge_index)):
            row = {
                src_col: int(data.edge_index[i][0]),
                dst_col: int(data.edge_index[i][1]),
                time_col: int(data.timestamps[i]),
            }
            if data.edge_feats is not None:
                if edge_feature_col is None:
                    raise ValueError(
                        'No feature column provided but events had features'
                    )

                feats = data.edge_feats[i]

                if len(feats.shape) > 1:
                    raise ValueError('Multi-dimensional features not supported')

                if len(feats) != len(edge_feature_col):
                    raise ValueError(
                        f'Got {len(feats)}-dimensional feature tensor but only '
                        f'specified {len(edge_feature_col)} feature column names.'
                    )

                features_list = feats.tolist()
                for feature_col, feature_val in zip(edge_feature_col, features_list):
                    row[feature_col] = feature_val

            writer.writerow(row)
