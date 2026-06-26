import pytest
import torch

from tgm.exceptions import BadAggregatorProtocolError
from tgm.nn import LinkPredictor
from tgm.nn.modules import ConcatMerge, LearnableSumMerge


class FooMerge:
    # Bad merge operation implementation: missing implementation of def dim() -> int:
    def __call__(self, z_src: torch.Tensor, z_dst: torch.Tensor):
        return z_src


@pytest.fixture
def edge_factory():
    src = torch.rand(200, 128)
    dst = torch.rand(200, 128)
    return src, dst


def test_cat_merge():
    src = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    dst = torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    merge_op = ConcatMerge(dim=5)
    assert merge_op.out_channels == 5 * 2
    merge_result = merge_op(src, dst)

    expected = torch.tensor(
        [[1, 2, 3, 4, 5, 11, 12, 13, 14, 15], [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]]
    )

    assert torch.equal(expected, merge_result)


def test_sum_merge():
    src = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).float()
    dst = torch.tensor([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]).float()
    merge_op = LearnableSumMerge(dim=5)
    assert merge_op.out_channels == 5
    merge_result = merge_op(src, dst)
    assert list(merge_result.shape) == [2, 5]
    assert not torch.isnan(merge_result).any()


@pytest.mark.parametrize('merge_op_name', ['concat', 'sum'])
def test_output(edge_factory, merge_op_name):
    merge_op = None
    if merge_op_name == 'concat':
        merge_op = ConcatMerge(128)
    else:
        merge_op = LearnableSumMerge(128)

    decoder = LinkPredictor(node_dim=128, nlayers=5, hidden_dim=64, merge_op=merge_op)
    src, dst = edge_factory

    out = decoder(src, dst)

    assert not torch.isnan(out).any()
    assert len(decoder.model) == 9  # 5 layers + 4 ReLU
    assert list(out.shape) == [200]

    # check the first layer
    if merge_op_name == 'concat':
        assert decoder.model[0].in_features == 128 * 2  # concat 2 nodes embeddings
    else:
        assert decoder.model[0].in_features == 128
    assert decoder.model[0].out_features == 64  # concat 2 nodes embeddings

    # check the last layer
    assert decoder.model[-1].in_features == 64
    assert decoder.model[-1].out_features == 1

    for i in range(
        2, len(decoder.model) - 1, 2
    ):  # exclude the first and the last layer
        assert decoder.model[i].in_features == 64
        assert decoder.model[i].out_features == 64


def test_bad_merge():
    with pytest.raises(BadAggregatorProtocolError):
        LinkPredictor(node_dim=128, nlayers=5, hidden_dim=64, merge_op=FooMerge())
