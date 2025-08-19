import pytest

from tgm.split import RatioSplit, TimeSplit


def test_time_split_bad_args():
    with pytest.raises(ValueError):
        TimeSplit(val_time=-1, test_time=0)
    with pytest.raises(ValueError):
        TimeSplit(val_time=2, test_time=1)


def test_time_split():
    pass


def test_time_split_with_node_feats():
    pass


def test_time_split_no_val_split():
    pass


def test_ratio_split_bad_args():
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=0.1)
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=-1, val_ratio=0, test_ratio=1)
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=0.1, val_ratio=0.1, test_ratio=0.1)
    with pytest.raises(ValueError):
        RatioSplit(train_ratio=0.4, val_ratio=0.4, test_ratio=0.4)


def test_ratio_split():
    pass


def test_ratio_split_with_node_feats():
    pass


def test_ratio_split_no_train_split():
    pass


def test_ratio_split_no_val_split():
    pass


def test_ratio_split_train_only_split():
    pass


def test_tgbl_split():
    pass


def test_tgbn_split():
    pass
