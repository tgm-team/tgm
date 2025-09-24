# Quick Start Guide

### Running Pre-packaged Examples

Start by syncing additional dependencies in our example scripts:

```sh
uv sync --group examples
```

For this example, we'll run [TGAT](https://arxiv.org/abs/2002.07962) dynamic link-prediction on [tgbl-wiki](https://tgb.complexdatalab.com/docs/leader_linkprop/#tgbl-wiki-v2). We'll use standard parameters on run on GPU. We show some explicit arguments for clarity:

```
python examples/linkproppred/tgat.py \
  --dataset tgbl-wiki \
  --bsize 200 \
  --device cuda \
  --epochs 1 \
  --n-nbrs 20 20 \
  --sampling recency
```
