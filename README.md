<div align="center">
  <img src="https://raw.githubusercontent.com/CyrilJl/BatchStats/main/docs/source/_static/logo_batchstats.svg" alt="Logo BatchStats" width="200">

[![PyPI Version](https://img.shields.io/pypi/v/batchstats.svg)](https://pypi.org/project/batchstats/)
[![conda Version](https://anaconda.org/conda-forge/batchstats/badges/version.svg)](https://anaconda.org/conda-forge/batchstats)
[![Documentation Status](https://img.shields.io/readthedocs/batchstats?logo=read-the-docs)](https://batchstats.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/59da873e81d84d9281c58c1a09bc72e9)](https://app.codacy.com/gh/CyrilJl/BatchStats/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

</div>

# BatchStats

BatchStats computes statistics on data that arrives in batches, so you can stream or process large datasets without loading everything into memory. Feed batches with `update_batch`, then call the object to get the final result.

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

data_stream = (np.random.randn(100, 10) for _ in range(10))

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

Docs: https://batchstats.readthedocs.io
