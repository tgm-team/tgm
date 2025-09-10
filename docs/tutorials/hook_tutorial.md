# Hooks and Hook Management in Temporal Graph Learning

Temporal graph learning pipelines often require dynamic transformations on graph batches—like sampling neighbors, generating negative edges, or moving data to GPU. TGM defines **hooks** to provide a flexible, composable way to perform these transformations automatically during batch iteration. We discuss them in this tutorial.

______________________________________________________________________

## 1. Hooks: The Basics

A `DGHook` is a callable object that receives a `DGBatch` (a batch of graph events), and a `DGraph` (a temporal view over the entire graph) and returns a transformed `DGBatch`, with additional properties.

See [`tgm.graph.DGBatch`](../api/batch.md) for a full reference of the base `DGBatch` yielded by our `DGDataLoader`.

Hooks declare the following information

- `requires`: Set of attributes that the hook needs to exist on the batch
- `produces`: Set of attributes that the hook adds to the batch
- `has_state`: A boolean flag to denote whether the hook stores state internally

Note:

- `StatelessHook`: only transforms the batch, no internal state (`has_state = False`)
- `StatelefullHook`: maintains internal state, (`has_state = False`)

### Built-in Hooks

TGM ships with several commonly used hooks for temporal graph learning.

The table bellow summarizes them:

| Hook Name                    | Type      | `requires` | `produces`                           | Description                                                                                          |
| ---------------------------- | --------- | ---------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `NegativeEdgeSamplerHook`    | Stateless | None       | `neg`, `neg_time`                    | Generates random negatives for link prediction                                                       |
| `TGBNegativeEdgeSamplerHook` | Stateless | None       | `neg`, `neg_time`, `neg_batch_list`  | Loads pre-computed negative edges for TGB datasets                                                   |
| `NeighborSamplerHook`        | Stateless | None       | `nbr_nids`, `nbr_times`, `nbr_feats` | Uniform sampler neighbor for a given number of hops                                                  |
| `RecencyNeighborSamplerHook` | Stateful  | None       | `nbr_nids`, `nbr_times`, `nbr_feats` | Recency neighbor sampler for a given number of hops                                                  |
| `PinMemoryHook`              | Stateless | None       | None                                 | Pins all `torch.Tensor` in `DGBatch` for fast cpu-gpu transfer                                       |
| `DeduplicationHook`          | Stateful  | None       | `unique_nids`, `global_to_local`     | Computes unique node ids in `DGBatch` and a mapping from global (graph) to local (batch) coordinates |

### Custom Hooks

If you are developing a new model or new sampling strategy, chances are, you need to define your own hook. The first step is to think about whether you need internal state. If not, you can subclass `tgm.hooks.StatelessHook`.

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

**Important**: Each hooks adds attributes to the batch. Hooks that run after it may dependend on these attributes (with `requires`). More on that later.

## 2. HookManager: Orchestrator of Hooks

`HookManager` manages which hooks are applied to a batch, and in what order. You can think of it like a key-value store where:

- *Keys*: e.g. `'train'`, `'val'`, `'test'`
- *Values*: List of hooks associated with that key

Hooks are executed automatically during data loading, allowing different transformations to occur for different data splits. For instance:

```python
from tgm.hooks import NegativeEdgeSamplerHook # A real negative edge sampler
from tgm.loader import DGDataLoader

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

**Important**: When creating custom hooks, you need to make sure you follow the correct hook API. See [`tgm.hooks`](../api/hooks/hooks.md) for more information. A `BadHookProtocolError` will be thrown if you accidently tried registering a hook with the wrong API.

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
        assert torch.all(batch.my_neg >= 0) # True
        assert torch.all(batch.my_neg < 10) # True
```

**Note**: The context manager is just syntatical sugar for the following:

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

## 4. Shared Hooks

Sometimes it happens that you truly want the same hook instance (as in, the same object) to be shared among different keys. A good example is the `tgm.hooks.RecencyNeighborSampler` which buffers internal state for events for each node in the graph. We want this hook to be shared between 'train' and 'validation' splits, because we want the neighboures accumulates during training to *warm-start* the recent neighbour in the validation split.

For this purpose, we have the notion of `shared hooks`, which are automatically attributed to all key pairs in the `HookManager`:

```python
from tgm.loader import DGDataLoader

# Create our graph
train_dg, test_dg = ...

# Initialize a hook manager with 'train' and 'test' keys
hm = HookManager(keys=['train', 'test'])

# Register our dummy hook across both the train and test split
hm.register_shared(MyNegativeHook())
```

*Note*: Using shared hooks is typically only useful if the hook has state, that needs to be shared across activation keys.

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

You will see the error message tell you that the manager could not find a valid ordering of hooks, and that's because no hook *produces* `'foo'`. If you encounter this, chances are you just mispelled either your `requires` or `produces` specification.

*Note*: You can also manually try to resolve hooks for a specific key without activating anything:

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

## Summary

`DGHook`s are modular transformation applied to batches under the hood during data loading. The `HookManager` orchestrates hooks by key-value pair, and ensures correct execution order given the set of `requires` and `produces` attributes. After activating a given key, the yielded batch from the dataloader will have all the `produces` attributes computed for you.

By sub-classing either the `StatefullHook` or `StatelessHook`, you can define you own custom transformation and operators in `TGM`.
