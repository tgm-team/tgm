from typing import Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch

from tgm import DGraph, RecipeRegistry
from tgm.constants import RECIPE_TGB_LINK_PRED
from tgm.data import DGData
from tgm.exceptions import UndefinedRecipe
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    TGBNegativeEdgeSamplerHook,
)


@pytest.fixture
def dg():
    edge_index = torch.LongTensor([[1, 10], [1, 11], [1, 12], [1, 13]])
    edge_timestamps = torch.LongTensor([1, 1, 2, 2])
    data = DGData.from_raw(edge_timestamps, edge_index)
    return DGraph(data)


@pytest.fixture
def tgb_dataset_factory():
    tgb_dataset = MagicMock()
    neg_sampler = MagicMock()
    neg_sampler.eval_set = {'val': [], 'test': []}
    tgb_dataset.negative_sampler = neg_sampler
    return tgb_dataset


@patch('tgm.recipe.PyGLinkPropPredDataset')
def test_bad_build_recipe(mock_dataset_cls, tgb_dataset_factory, dg):
    mock_dataset = tgb_dataset_factory()
    mock_dataset_cls.return_value = mock_dataset

    with pytest.raises(UndefinedRecipe):
        hm, register_keys = RecipeRegistry.build(
            'foo', dataset_name='tgbl-foo', train_dg=dg
        )


@patch('tgm.recipe.PyGLinkPropPredDataset')
def test_build_recipe_tgb_link_pred(mock_dataset_cls, tgb_dataset_factory, dg):
    mock_dataset = tgb_dataset_factory()
    mock_dataset_cls.return_value = mock_dataset

    hm, register_keys = RecipeRegistry.build(
        RECIPE_TGB_LINK_PRED, dataset_name='tgbl-foo', train_dg=dg
    )
    train_hooks = hm._key_to_hooks['train']
    val_hooks = hm._key_to_hooks['val']
    test_hooks = hm._key_to_hooks['test']

    assert len(register_keys) == 3
    assert (
        register_keys[0] == 'train'
        and register_keys[1] == 'val'
        and register_keys[2] == 'test'
    )
    assert len(train_hooks) == len(val_hooks) == len(test_hooks) == 1
    assert isinstance(train_hooks[0], NegativeEdgeSamplerHook)
    assert isinstance(val_hooks[0], TGBNegativeEdgeSamplerHook)
    assert isinstance(test_hooks[0], TGBNegativeEdgeSamplerHook)
    mock_dataset_cls.assert_called_once_with(name='tgbl-foo', root='datasets')
    mock_dataset.load_val_ns.assert_called_once()
    mock_dataset.load_test_ns.assert_called_once()


def test_register_new_recipe():
    state = {'call': 0}

    @RecipeRegistry.register('foo')
    def mock_recipe(input: str) -> Tuple[HookManager, str]:
        state['call'] += 1
        hm = HookManager(['global'])
        return hm, input

    hm, output = RecipeRegistry.build('foo', input='input_1')

    assert isinstance(hm, HookManager)
    assert output == 'input_1'
    assert state['call'] == 1
