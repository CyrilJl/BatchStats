[![PyPI Version](https://img.shields.io/pypi/v/batchstats.svg)](https://pypi.org/project/batchstats/)

# BatchStats

`batchstats` is a Python package designed to compute various statistics of data that arrive batch by batch, making it suitable for streaming input or data too large to fit in memory.

## Installation

You can install `batchstats` using pip:

```
pip install batchstats
```

## Usage

Here's an example of how to use `batchstats` to compute batch mean and variance:

```python
from batchstats import BatchMean, BatchVar

# Initialize BatchMean and BatchVar objects
batchmean = BatchMean()
batchvar = BatchVar()

# Iterate over your generator of data batches
for batch in your_data_generator:
    # Update BatchMean and BatchVar with the current batch of data
    batchmean.update_batch(batch)
    batchvar.update_batch(batch)

# Compute and print the mean and variance
print("Batch Mean:", batchmean())
print("Batch Variance:", batchvar())
```

It is also possible to compute the covariance between two datasets:

```python
import numpy as np
from batchstats import BatchCov

n_samples, m, n = 10_000, 100, 50
data1 = np.random.randn(n_samples, m)
data2 = np.random.randn(n_samples, n)
n_batches = 7

batchcov = BatchCov()
for batch_index in np.array_split(np.arange(n_samples), n_batches):
    batchcov.update_batch(batch1=data1[batch_index], batch2=data2[batch_index])
true_cov = (data1 - data1.mean(axis=0)).T@(data2 - data2.mean(axis=0))/n_samples
np.allclose(true_cov, batchcov()), batchcov().shape
>>> (True, (100, 50))
```

`batchstats` is also flexible in terms of input shapes, with the first dimension always representing the samples and the remaining dimensions representing the features:

```python
import numpy as np
from batchstats import BatchSum

data = np.random.randn(10_000, 80, 90)
n_batches = 7

batchsum = BatchSum()
for batch_data in np.array_split(data, n_batches):
    batchsum.update_batch(batch_data)

true_sum = np.sum(data, axis=0)
np.allclose(true_sum, batchsum()), batchsum().shape
>>> (True, (80, 90))
```

## Available Classes/Stats

- `BatchCov`: Compute the covariance matrix of two datasets (not necessarily square).
- `BatchMax`: Compute the maximum value.
- `BatchMean`: Compute the mean.
- `BatchMin`: Compute the minimum value.
- `BatchSum`: Compute the sum.
- `BatchVar`: Compute the variance.

Each class is tested against numpy results to ensure accuracy. For example:

```python
import numpy as np
from batchstats import BatchMean

def test_mean(data, n_batches):
    true_stat = np.mean(data, axis=0)

    batchmean = BatchMean()
    for batch_data in np.array_split(data, n_batches):
        batchmean.update_batch(batch=batch_data)
    batch_stat = batchmean()
    return np.allclose(true_stat, batch_stat)

data = np.random.randn(1_000_000, 50)
n_batches = 31
test_mean(data, n_batches)
>>> True
```

## Requesting Additional Statistics

If you require additional statistics that are not currently implemented in `batchstats`, feel free to open an issue on the GitHub repository or submit a pull request with your suggested feature. We welcome contributions and feedback from the community to improve `batchstats` and make it more versatile for various data analysis tasks.
