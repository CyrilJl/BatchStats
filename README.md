<div align="center">
  <img src="https://raw.githubusercontent.com/CyrilJl/BatchStats/main/docs/source/_static/logo_batchstats.svg" alt="Logo BatchStats" width="200">

[![PyPI Version](https://img.shields.io/pypi/v/batchstats.svg)](https://pypi.org/project/batchstats/)
[![Python Versions](https://img.shields.io/pypi/pyversions/batchstats.svg)](https://pypi.org/project/batchstats/)
[![conda Version](https://anaconda.org/conda-forge/batchstats/badges/version.svg)](https://anaconda.org/conda-forge/batchstats)
[![Documentation Status](https://img.shields.io/readthedocs/batchstats?logo=read-the-docs)](https://batchstats.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml)

</div>

# BatchStats

BatchStats computes statistics on data that arrives in batches, so you can stream or process large datasets without loading everything into memory. Its incremental algorithms expose a small NumPy-friendly API and support merging independently computed accumulators.

BatchStats requires Python 3.10 or newer.

## Installation

```console
pip install batchstats
```

Or with `conda`/`mamba`:

```console
conda install -c conda-forge batchstats
```

## Quick Start

```python
import numpy as np
from batchstats import BatchMean, BatchVar

rng = np.random.default_rng(0)
data_stream = (rng.standard_normal((100, 10)) for _ in range(10))

batch_mean = BatchMean()
batch_var = BatchVar()

for batch in data_stream:
    batch_mean.update_batch(batch)
    batch_var.update_batch(batch)

mean = batch_mean()
variance = batch_var()

print(f"Mean shape: {mean.shape}")
print(f"Variance shape: {variance.shape}")
```

## Available Statistics

* `BatchSum` / `BatchNanSum`
* `BatchWeightedSum`
* `BatchMean` / `BatchNanMean`
* `BatchWeightedMean`
* `BatchMin` / `BatchNanMin`
* `BatchMax` / `BatchNanMax`
* `BatchPeakToPeak` / `BatchNanPeakToPeak`
* `BatchVar`
* `BatchStd`
* `BatchCov`
* `BatchCorr`
* `BatchTopK` / `BatchNanTopK` (exact extreme ranks and linear tail quantiles)

`BatchNanSum`, `BatchNanMean` and the top-k accumulators can be saved and resumed
with `save()` / `load()` using NPZ checkpoints. Use `to_state()` / `from_state()`
to export and restore their state in memory.

```python
from batchstats import BatchNanTopK, required_k

extremes = BatchNanTopK(required_k(0.998, 8784), axis=0)
for block in data_stream:  # (time, *spatial_shape), same spatial positions
    extremes.update_batch(block)
p998 = extremes.quantile(0.998)  # raises if retained capacity is insufficient
extremes.save("extremes.npz")
```

Top-k outputs have a leading rank axis and mask unavailable ranks while preserving
the selected dtype. `BatchTopK` rejects NaNs; `BatchNanTopK` ignores them per cell.
See [the streaming and checkpoint guide](docs/source/extreme_statistics.rst).

Docs: https://batchstats.readthedocs.io

## Optional xarray support

```console
pip install "batchstats[xarray]"
```

Use the classes in `batchstats.xarray` with a `DataArray` or a `Dataset`:

```python
from batchstats.xarray import BatchNanMean

mean = BatchNanMean(dim="time")
for batch in labelled_batches:
    mean.update_batch(batch)
result = mean()  # DataArray or Dataset, with the remaining dimensions/coordinates
```

All the statistics listed above have labelled counterparts. Reduce one or more
named dimensions with `dim="time"` or `dim=("time", "level")`; `dim=None`
reduces all dimensions. Dataset variables are accumulated independently. The
NumPy API and its dependencies remain unchanged; importing `batchstats` does not
import xarray. See [the xarray guide](docs/source/xarray_support.rst) for coordinate
validation, weighted statistics, covariance, top-k and streaming limits.

## Development

Install the development dependencies and run the local quality gates:

```console
python -m pip install -e ".[dev,xarray]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=batchstats
python -m build
python -m twine check dist/*
```
