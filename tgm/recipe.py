from typing import Any, Callable, Dict

from tgm import DGraph
from tgm.constants import RECIPE_TGB_LINK_PRED
from tgm.exceptions import UndefinedRecipe
from tgm.hooks import (
    HookManager,
    NegativeEdgeSamplerHook,
    TGBNegativeEdgeSamplerHook,
)
from tgm.util._tgb import suppress_output


class RecipeRegistry:
    """Registry recipe for managing all pre-defined recipe in the form of Callable object.

    This class allows you to register your custom recipes perform frequently-used pre-experiment setup used.
    One common frequently-used pre-experiment setup is set up HookManager for TGB linkpropred, which is provided by TGM team.
    Users are welcome to define their own recipe. User-defined recipe need to be registered with `RecipeRegistry` to be able to build.
    Please see tutorial for further information

    Args:
    Raises:
        UndefinedRecipe: If build is call on recipe that is not defined or registered.
    """

    _recipes: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            cls._recipes[name] = func
            return func

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs) -> Any:  # type: ignore
        if name not in cls._recipes:
            raise UndefinedRecipe(
                f'Undefined or not yet registered recipe: {name}. Please select from {cls._recipes}'
            )
        return cls._recipes[name](**kwargs)


@RecipeRegistry.register(RECIPE_TGB_LINK_PRED)
def build_tgb_link_pred(dataset_name: str, train_dg: DGraph) -> HookManager:
    """Build ready-to-use HookManager with default hooks for TGB linkproppred task.

    Args:
        dataset_name (str): The name of the TGB dataset (e.g. 'tgbl-wiki')
        train_dg (DGraph): The training graph, used to setup the `NegativeEdgeSamplerHook` for training

    Returns:
        HookManager with registered keys: ['train', 'val', 'test']
    """
    try:
        from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    except ImportError:
        raise ImportError('TGB required to load TGB data, try `pip install py-tgb`')

    dataset = suppress_output(
        PyGLinkPropPredDataset, name=dataset_name, root='datasets'
    )
    dataset.load_val_ns()
    dataset.load_test_ns()
    _, dst, _ = train_dg.edges
    neg_sampler = dataset.negative_sampler

    hm = HookManager(keys=['train', 'val', 'test'])
    hm.register(
        'train', NegativeEdgeSamplerHook(low=int(dst.min()), high=int(dst.max()))
    )
    hm.register('val', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='val'))
    hm.register('test', TGBNegativeEdgeSamplerHook(neg_sampler, split_mode='test'))

    return hm
