<div align="center">
  <img src="https://raw.githubusercontent.com/CyrilJl/BatchStats/main/docs/source/_static/logo_batchstats.svg" alt="Logo BatchStats" width="200">

[![PyPI Version](https://img.shields.io/pypi/v/batchstats.svg)](https://pypi.org/project/batchstats/)
[![conda Version](https://anaconda.org/conda-forge/batchstats/badges/version.svg)](https://anaconda.org/conda-forge/batchstats)
[![Documentation Status](https://img.shields.io/readthedocs/batchstats?logo=read-the-docs)](https://batchstats.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/59da873e81d84d9281c58c1a09bc72e9)](https://app.codacy.com/gh/CyrilJl/BatchStats/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

</div>

``batchstats`` is a Python package for computing statistics on data that arrives in batches. It's perfect for streaming data or datasets too large to fit into memory.

For detailed information, please check out the [full documentation](https://batchstats.readthedocs.io).

## Installation

Install ``batchstats`` using ``pip``:

```console
pip install batchstats
```

Or with `conda`:

```console
conda install -c conda-forge batchstats
```

## Quick Start

Here's how to compute the mean and variance of a dataset in batches:

```python
import numpy as np
from batchstats import BatchMean, BatchVar

# Simulate a data stream
data_stream = (np.random.randn(100, 10) for _ in range(10))

# Initialize the stat objects
batch_mean = BatchMean()
batch_var = BatchVar()

# Process each batch
for batch in data_stream:
    batch_mean.update_batch(batch)
    batch_var.update_batch(batch)

# Get the final result
mean = batch_mean()
variance = batch_var()

print(f"Mean shape: {mean.shape}")
print(f"Variance shape: {variance.shape}")
```

## Advanced Usage

`batchstats` handles n-dimensional `np.ndarray` inputs and allows specifying multiple axes for reduction, just like `numpy`.

```python
import numpy as np
from batchstats import BatchMean

# Create a 3D data stream
data_stream = (np.random.rand(10, 5, 8) for _ in range(5))

# Compute the mean over the last two axes (1 and 2)
batch_mean_3d = BatchMean(axis=(1, 2))

for batch in data_stream:
    batch_mean_3d.update_batch(batch)

mean_3d = batch_mean_3d()

print(f"3D Mean shape: {mean_3d.shape}")
```

## Handling NaN Values

``batchstats`` provides `BatchNan*` classes to handle `NaN` values, similar to `numpy`'s `nan*` functions.

```python
import numpy as np
from batchstats import BatchNanMean

# Create data with NaNs
data = np.random.randn(1000, 5)
data[::10] = np.nan

# Compute the mean, ignoring NaNs
nan_mean = BatchNanMean().update_batch(data)()

print(f"NaN-aware mean shape: {nan_mean.shape}")
```

## Available Statistics

``batchstats`` supports a variety of common statistics:

* `BatchSum` / `BatchNanSum`
* `BatchMean` / `BatchNanMean`
* `BatchMin` / `BatchNanMin`
* `BatchMax` / `BatchNanMax`
* `BatchPeakToPeak` / `BatchNanPeakToPeak`
* `BatchVar`
* `BatchStd`
* `BatchCov`

For more details on each class, see the [API Reference](https://batchstats.readthedocs.io/en/latest/api.html).
