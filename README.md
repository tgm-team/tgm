<a id="readme-top"></a>

![image](./docs/img/logo.svg)

<div align="center">
<h3 style="font-size: 22px">Efficient and Modular ML on Temporal Graphs</h3>
<br/>
<br/>

</div>

## About The Project

TGM is a research library designed to accelerate training workloads over dynamic graphs and facilitate prototyping of temporal graph learning methods.

Our main goal is to provide an efficient abstraction over dynamic graphs to enable new practitioners to quickly contribute to research in the field. We natively support both discrete and continuous-time graphs.


### Library Highlights

- :hourglass_flowing_sand: First library to support both discrete and continuous-time graphs
- :wrench: Built-in support for [TGB datasets](https://tgb.complexdatalab.com/), MRR-based link prediction, Node Property Prediction and Graph level tasks
- :sparkles: Modular, intuitive API for rapid model prototyping
- :atom: Efficient dataloading with edge and time-based batching
- :heavy_check_mark: Validated implementations of popular TG methods

### Supported Methods

We aim to support the following temporal graph learning methods.

- :white_check_mark: [EdgeBank](https://arxiv.org/abs/2207.10128)
- :white_check_mark: [GCN](https://arxiv.org/abs/1609.02907)
- :white_check_mark: [GC-LSTM](https://arxiv.org/abs/1812.04206)
- :white_check_mark: [GraphMixer](https://arxiv.org/abs/2302.11636)
- :white_check_mark: [TGAT](https://arxiv.org/abs/2002.07962)
- :white_check_mark: [TGN](https://arxiv.org/abs/2006.10637)
- :white_check_mark: [DygFormer](https://arxiv.org/abs/2303.13047)
- :white_check_mark: [TPNet](https://arxiv.org/abs/2410.04013)
- :white_large_square: [TNCN](https://arxiv.org/abs/2406.07926)
- :white_large_square: [DyGMamba](https://arxiv.org/abs/2408.04713)
- :white_large_square: [NAT](https://arxiv.org/abs/2209.01084)


## Installation

#### Using [uv](https://docs.astral.sh/uv/) (recommended)

```sh
# Create and activate your venv
uv venv my_venv --python 3.10 && source my_venv/bin/activate

# Install the wheels into the venv
uv pip install -e .

# Test the install
uv run python -c 'import tgm; print(tgm.__version__)'
```

### Running Pre-packaged Examples

Start by syncing additional dependencies in our example scripts:

```sh
uv sync --group examples
```

For this example, we'll run [TGAT](https://arxiv.org/abs/2002.07962) dynamic link-prediction on tgbl-wiki
We'll use standard parameters on run on GPU. We show some explicit arguments for clarity:

```
python examples/linkproppred/tgat.py \
  --dataset tgbl-wiki \
  --bsize 200 \
  --device cuda \
  --epochs 1 \
  --n-nbrs 20 20 \
  --sampling recency
```


## Documentation

Documentation along with a quick start guide can be found on the [docs website](xxxxxxxxx).


