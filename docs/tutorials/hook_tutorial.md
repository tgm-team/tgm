# Hook Management in TGM

Temporal graph learning pipelines often require dynamic transformations on graph batches—like sampling neighbors, generating negative edges, or moving data to GPU. TGM defines `DGHook`s to provide a flexible, composable way to perform these transformations automatically during batch iteration. Think of `DGHook`s as all the necessary data processing and operations before you feed the current batch into the TG ML model.

______________________________________________________________________

## 1. Hooks: The Basics

A `DGHook` is a callable object that takes a `DGBatch` (a batch of graph events) and a `DGraph` (a temporal view over the entire graph) as inputs and returns a transformed `DGBatch`, with additional properties.

See [`tgm.graph.DGBatch`](../api/batch.md) for a full reference of the base `DGBatch` yielded by our `DGDataLoader`.

Hooks declare the following information

- `requires: Set[str]`: Names of attributes that the hook needs to exist on the batch
- `produces: Set[str]`: Names of attributes from the batch that the hook requires
- `has_state: bool`: A flag to denote whether the hook stores state internally (i.e. some memory or attribute that may change upon subsequent invocations of the hook). An example of a stateful hook is a `RecencyNeighborSampler` which keeps track of node interactions over subsequent `__call__`s.

> Note:
> \- `StatelessHook`: only transforms the batch, no internal state (`has_state = False`)
> \- `StatefulHook`: maintains internal state, (`has_state = True`)

### Built-in Hooks

TGM implements several commonly used hooks. The table below summarizes them:

| Hook Name                    | Type      | `requires` | `produces`                           | Description                                                                                          |
| ---------------------------- | --------- | ---------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `NegativeEdgeSamplerHook`    | Stateless | None       | `neg`, `neg_time`                    | Generates random negatives for link prediction                                                       |
| `TGBNegativeEdgeSamplerHook` | Stateless | None       | `neg`, `neg_time`, `neg_batch_list`  | Loads pre-computed negative edges for [TGB](https://tgb.complexdatalab.com/) datasets                |
| `NeighborSamplerHook`        | Stateless | None       | `nbr_nids`, `nbr_times`, `nbr_feats` | Uniform sampler neighbor for a given number of hops                                                  |
| `RecencyNeighborSamplerHook` | Stateful  | None       | `nbr_nids`, `nbr_times`, `nbr_feats` | Recency neighbor sampler for a given number of hops                                                  |
| `PinMemoryHook`              | Stateless | None       | None                                 | Pins all `torch.Tensor` in `DGBatch` for fast CPU-GPU transfer                                       |
| `DeduplicationHook`          | Stateful  | None       | `unique_nids`, `global_to_local`     | Computes unique node ids in `DGBatch` and a mapping from global (graph) to local (batch) coordinates |

### Custom Hooks

Along with the hooks provided by `TGM` team, users are welcome to write custom hooks to perform any operations on `DGBatch` as desired. For instance, if you are developing a new model or new sampling strategy, chances are, all you need to do is define a custom hook. The first step is to think about whether you need internal state. If not, you can subclass `tgm.hooks.StatelessHook`.

For example, the following shows a simple implementation of a negative sampler hook, which add random negative nodes in the range `[10, 100)`, and a corresponding *negative time* which matches the ground truth batch time:

```python
from tgm.hooks import StatelessHook
from tgm import DGBatch, DGraph

class MyNegativeHook(StatelessHook):
    produces = {'my_neg', 'my_neg_time'}
    requires = set()

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.my_neg = torch.randint(10, 100, (len(batch.dst),))
        batch.my_neg_time = batch.time.clone()
        return batch
```

> **Important**: Each hooks adds attributes to the batch. Hooks that run after it may depend on these attributes (defined in `requires`). More on that later.

## 2. HookManager: Orchestrator of Hooks

Typically, a full training and evaluation pipeline will require multiple hooks, perhaps some of which execute conditionally on your workload (e.g. validation vs. test). The `HookManager` manages which hooks are applied to a batch, and in what order. You can think of it like a key-value store where:

- *Keys*: e.g. `'train'`, `'val'`, `'test'`
- *Values*: List of hooks associated with each key

Hooks are executed automatically during data loading, allowing different transformations to occur for different data splits. For instance:

```python
from tgm.hooks import NegativeEdgeSamplerHook # A real negative edge sampler
from tgm.data import DGDataLoader

# Create our graph
train_dg, test_dg = ...

# Initialize a hook manager with 'train' and 'test' keys
hm = HookManager(keys=['train', 'test'])

# Train: Random negatives
hm.register('train', NegativeEdgeSamplerHook(low=0, high=dg.num_nodes))

# Test: Use the dummy class we defined above
hm.register('test', MyNegativeHook())

train_loader = DGDataLoader(train_dg, hook_manager=hm)
test_loader = DGDataLoader(test_dg, hook_manager=hm)
```

**Important**: When creating custom hooks, you need to make sure you follow the correct hook API. See [`tgm.hooks`](../api/hooks/hooks.md) for more information. A `BadHookProtocolError` will be thrown if you accidentlly tried registering a hook with the wrong API. We suggest you write some unit tests to accompany your custom protocols. You can see [some of our hook tests](https://github.com/tgm-team/tgm/tree/main/test/unit/test_hooks) as a starting point. If your hook has general utility to the TG community, we can add it to TGM and enable code re-use for other practitioners.

What now? Well, when we iterate our training graph, we have access to the attributes produced by `NegativeEdgeSamplerHook`, which are `neg` and `neg_time`. In order to see these transformations get applied, we need to *activate* the key we are interested in...

## 3. Context Management

In the previous section, we created a hook manager and added a hook to the 'train' key and another to the 'test' key. If we just try iterating the data, we won't see the attributes we want:

```python
for batch in train_loader:
    assert batch.dst.shape() == batch.neg.shape() # AttributeError! No attribute `neg` in batch

for batch in test_loader:
    assert batch.dst.shape() == batch.my_neg.shape() # AttributeError! No attribute `my_neg` in batch
```

What we have to do is *activate* the keys we want. This allows us to selectively execute the right transformation, depending on which key is active. We can use the `with hm.activate()` context manager to do so:

```python
with hm.activate('train'):
    for batch in train_loader:
        assert batch.dst.shape() == batch.neg.shape() # True

with hm.activate('test'):
    for batch in test_loader:
        assert batch.dst.shape() == batch.my_neg.shape() # True
        assert torch.all(batch.my_neg >= 10) # True
        assert torch.all(batch.my_neg < 100) # True
        assert torch.equal(batch.my_neg_time, batch.time) # True
```

> **Note**: The context manager is just syntactical sugar for the following:

```python
with hm.activate(key):
    ...

#### Equivalent to
hm.set_active_hooks(key)
...
hm.set_active_hooks(None)
```

See [`tgm.hooks.HookManager`](../api/hooks/hook_manager.md) for a full reference.

### State Reset

Often it will happen that hooks with internal memory (stateful hooks) require that some memory is reset, at an end of epoch, for instance. The `HookManager` will automatically walk through all the stateful hooks and call `reset_state()` internally when you issue:

```python
hm.reset_state()
```

You can also selectively reset hooks for a particular key.

```python
hm.reset_state('train')
```

## 4. Shared Hooks

In temporal graph learning, it is common that information you received in the past needed to be used for future prediction. For example, the stored neighbours in the `tgm.hooks.RecencyNeighborSampler` hook is state that must be carried to the validation phase to ensure that the models can access information from the training set. Therefore, this raises the need for sharing hook state of a hook across splits.

For this purpose, we have the notion of `shared hooks`, which are automatically attributed to **all** keys in the `HookManager`:

```python
from tgm.data import DGDataLoader

# Create our graph
train_dg, test_dg = ...

# Initialize a hook manager with 'train' and 'test' keys
hm = HookManager(keys=['train', 'test'])

# Register our dummy hook across both the train and test split
hm.register_shared(MyNegativeHook())
```

> *Note*: Using shared hooks is typically only useful if the hook has state, that needs to be shared across activation keys.

## 5. Hook Resolution

As you may have guessed, hooks add attributes that may depend on other hooks. Formally, the set of `requires` and `produces` attributes defined on `DGBatch` by the list of hooks defines a directed-acylic-graph (*DAG*) for every key in the hook manager. When we activate a key, the hook manager performs a topological sort of the hook list and finds a topological ordering to execute during data loading. This is only done once and cached, until (if) you decide to add more hooks for that key.

The upside is that you shouldn't care what order you register your hooks in, the manager will figure it out. But, it's possible that no valid ordering exists.

For instance, suppose in our dummy hook, we added a requirement that our hook `requires` the batch attribute `foo`:

```python
from tgm.hooks import StatelessHook
from tgm import DGBatch, DGraph

class MyNegativeHookWithFoo(StatelessHook):
    produces = {'my_neg', 'my_neg_time'}
    requires = {'foo'} # This hook depends on batch.foo existing!

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        batch.my_neg = torch.randint(10, 100, (len(batch.dst),))
        batch.my_neg_time = batch.time.clone()
        return batch
```

Now, if we register our hook and try to activate a key that uses it, we'll encounter the `tgm.UnresolvableHookDependenciesError`:

```python

# Register MyNegativeHook on 'train' then activate it and try iterating the data, as before
hm.register('train', MyNegativeHookWithFoo()) # Ok, registered

with hm.activate('train'): # Raises tgm.UnresolvableHookDependenciesError
    ...
```

You will see the error message tell you that the manager could not find a valid ordering of hooks, and that's because no hook *produces* `'foo'`. If you encounter this, chances are you just misspelled either your `requires` or `produces` specification.

> *Note*: You can also manually try to resolve hooks for a specific key without activating anything:

```python
hm.resolve_hooks('train') # Raises tgm.UnresolvableHookDepenenciesError
```

You can inspect the resolved hooks according to the `__str__` method on the `HookManager`, to validate that everything is as expected as well:

```python
print(hm)
```

It might give you something along the lines of:

```
HookManager:
  Shared hooks:
    - DeduplicationHook (requires=set(), produces={'unique_nids', 'global_to_local'})
    - MockHook (requires=set(), produces=set())
  Active key: None
  Keyed hooks:
    train:
      - DeduplicationHook (requires=set(), produces={'unique_nids', 'global_to_local'})
      - MockHook (requires=set(), produces=set())
      - MockHookRequires (requires={'foo'}, produces=set())
      - MockHookWithState (requires=set(), produces=set())
    val:
      - DeduplicationHook (requires=set(), produces={'unique_nids', 'global_to_local'})
      - MockHook (requires=set(), produces=set())
      - MockHookRequires (requires={'foo'}, produces=set())

```

______________________________________________________________________

## 6. Recipes

`TGM` offer a convenient way to setup common `HookManager` configuration by using `RecipeRegistry.build()` with a pre-defined recipe. For example, in the TGB `linkproppred` setting, the `HookManager` must register train, validation, and test hooks as follows:

```python
dataset = PyGLinkPropPredDataset(
    name=dataset_name, root='datasets'
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
```

To minimize boilerplate and avoid accidental typos in this setup process, this procedure can be encapsulated in a function and registered through `RecipeRegistry` as follows:

```python
@RecipeRegistry.register(RECIPE_TGB_LINK_PRED)
def build_tgb_link_pred(dataset_name: str, train_dg: DGraph) -> HookManager:
   try:
       from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
   except ImportError:
       raise ImportError('TGB required to load TGB data, try `pip install py-tgb`')


   dataset = PyGLinkPropPredDataset(
       name=dataset_name, root='datasets'
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
```

`build_tgb_link_pred()` encapsulates procedure to set up `HookManager` for `TGB` linkpropred experiments and is registered to `RecipeRegistry` with the name defined by constant `RECIPE_TGB_LINK_PRED` as follows:

```python
@RecipeRegistry.register(RECIPE_TGB_LINK_PRED)
```

Therefore, all we need to do to set up `HookManager` for `TGB` linkproppred is:

```python
hm = RecipeRegistry.build(
   RECIPE_TGB_LINK_PRED, dataset_name=args.dataset, train_dg=train_dg
)
registered_keys = hm.keys
train_key, val_key, test_key = registered_keys
```

`TGM` team provided the implementation of recipe for `TGB` linkproppred, users are welcome to define their own `Recipe`, register it and build it with `RecipeRegistry.build()`.

## Summary

`DGHook`s are modular transformation applied to batches under the hood during data loading. The `HookManager` orchestrates hooks by key-value pair, and ensures correct execution order given the set of `requires` and `produces` attributes. After activating a given key, the yielded batch from the dataloader will have all the `produces` attributes computed for you.

By sub-classing either the `StatefulHook` or `StatelessHook`, you can define you own custom hooks in `TGM`.
