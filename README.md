[![PyPI Version](https://img.shields.io/pypi/v/batchstats.svg)](https://pypi.org/project/batchstats/) [![conda Version](
https://anaconda.org/conda-forge/batchstats/badges/version.svg)](https://anaconda.org/conda-forge/batchstats) [![Documentation Status](https://img.shields.io/readthedocs/batchstats?logo=read-the-docs)](https://batchstats.readthedocs.io/en/latest/?badge=latest)
[![Unit tests](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml/badge.svg)](https://github.com/CyrilJl/BatchStats/actions/workflows/pytest.yml) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/59da873e81d84d9281c58c1a09bc72e9)](https://app.codacy.com/gh/CyrilJl/BatchStats/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# <img src="https://raw.githubusercontent.com/CyrilJl/BatchStats/main/docs/source/_static/logo_batchstats.svg" alt="Logo BatchStats" width="200" height="150" align="right"> BatchStats

`batchstats`  is a Python package designed to compute various statistics of data that arrive batch by batch (in chunks or segments), making it suitable for streaming input or data too large to fit in memory. The classes and methods implemented in batchstats are based on online algorithms—algorithms that process input piece-by-piece in a serial fashion, without requiring the entire input to be available from the start. For covariance and variance calculations, the package employs the celebrated Welford's online algorithm. Special care has been given to ensuring numerical precision, optimizing computation time, and minimizing memory usage.

## Installation

You can install `batchstats` using pip:

``` console
pip install batchstats
```
The package is also available on conda-forge:
``` console
conda install -c conda-forge batchstats
```

``` console
mamba install batchstats
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
# (True, (100, 50))
```

`batchstats` is also flexible in terms of input shapes. By default, statistics are applied along the first axis: the first dimension representing the samples and the remaining dimensions representing the features:

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
# (True, (80, 90))
```

However, similar to the associated functions in `numpy`, users can specify the reduction axis or axes:

```python
import numpy as np
from batchstats import BatchMean

data = [np.random.randn(24, 7, 128) for _ in range(100)]

batchmean = BatchMean(axis=(0, 2))
for batch in data:
    batchmean.update_batch(batch)
batchmean().shape
# (7,)

batchmean = BatchMean(axis=2)
for batch in data:
    batchmean.update_batch(batch)
batchmean().shape
# (24, 7)
```

## Merging Two Objects

In some cases, it is useful to process two different `BatchStats` objects from asynchronous I/O functions and then merge the statistics of both objects at the end. The `batchstats` library supports this functionality by allowing the simple addition of two objects. Under the hood, the necessary computations are performed to produce a resulting statistic that reflects the data from both input datasets, even imbalanced:

```python
import numpy as np
from batchstats import BatchCov

data = np.random.randn(25_000, 50)
data1 = data[:10_000]
data2 = data[10_000:]

cov = BatchCov().update_batch(data)
cov1 = BatchCov().update_batch(data1)
cov2 = BatchCov().update_batch(data2)

cov_merged = cov1 + cov2
np.allclose(cov(), cov_merged())
# True
```

The `__add__` method has been specifically overloaded to facilitate the merging of statistical objects in `batchstats`, including `BatchCov`, `BatchMax`, `BatchMean`, `BatchMin`, `BatchPeakToPeak`, `BatchStd`, `BatchSum`, and `BatchVar`.

## Performance

In addition to result accuracy, much attention has been given to computation times and memory usage. Fun fact, calculating the variance using `batchstats` consumes little RAM while being faster than `numpy.var`:

```python
%load_ext memory_profiler
import numpy as np
from batchstats import BatchVar

data = np.random.randn(100_000, 1000)
print(data.nbytes/2**20)
# 762.939453125

%memit a = np.var(data, axis=0)
# peak memory: 1604.63 MiB, increment: 763.35 MiB

%memit b = BatchVar().update_batch(data)()    
# peak memory: 842.62 MiB, increment: 0.91 MiB

np.allclose(a, b)
# True

%timeit a = np.var(data, axis=0)
# 510 ms ± 111 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit b = BatchVar().update_batch(data)()    
# 306 ms ± 5.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## NaN handling possibility

While the previous `Batch*` classes exclude every sample containing at least one NaN from the computations, the `BatchNan*` classes adopt a more flexible approach to handling NaN values, similar to `np.nansum`, `np.nanmean`, etc. Consequently, the outputted statistics can be computed from various numbers of samples for each feature:

```python
import numpy as np
from batchstats import BatchNanSum

m, n = 1_000_000, 50
nan_ratio = 0.05
n_batches = 17

data = np.random.randn(m, n)
num_nans = int(m * n * nan_ratio)
nan_indices = np.random.choice(range(m * n), num_nans, replace=False)
data.ravel()[nan_indices] = np.nan

batchsum = BatchNanSum()
for batch_data in np.array_split(data, n_batches):
    batchsum.update_batch(batch=batch_data)
np.allclose(np.nansum(data, axis=0), batchsum())
# True
```

## Available Classes/Stats

- `BatchCov`: Compute the covariance matrix of two datasets (not necessarily square)
- `BatchMax`: Compute the maximum value (associated to `np.max`)
- `BatchMean`: Compute the mean (associated to `np.mean`)
- `BatchMin`: Compute the minimum value (associated to `np.min`)
- `BatchPeakToPeak`: Compute maximum - minimum value (associated to `np.ptp`)
- `BatchStd`: Compute the standard deviation (associated to `np.std`)
- `BatchSum`: Compute the sum (associated to `np.sum`)
- `BatchVar`: Compute the variance (associated to `np.var`)

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
# True
```

## Machine Learning Application

Fitting a simple linear regression on chunked or streaming data can be done using `BatchCov`, for example:

```python
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from batchstats import BatchCov

class IncrementalLinearRegression(RegressorMixin, BaseEstimator):
    """
    IncrementalLinearRegression performs linear regression in an incremental way
    using batches of data. It uses BatchCov to accumulate covariance and mean
    information for incremental updates.
    """
    def __init__(self):
        self.cov_ = BatchCov()

    def partial_fit(self, X, y):
        self.cov_.update_batch(np.c_[X, y])
        return self

    def _compute_parameters(self):
        means = self.cov_.mean1()
        cov_matrix = self.cov_()
        # Calculate the coefficients
        coef_ = np.linalg.inv(cov_matrix[:-1, :-1]) @ cov_matrix[-1][:-1]
        # Calculate the intercept
        intercept_ = means[-1] - coef_ @ means[:-1]
        return coef_, intercept_

    def fit(self, X, y):
        return self.partial_fit(X, y)

    @property
    def coef_(self):
        coef_, _ = self._compute_parameters()
        return coef_

    @property
    def intercept_(self):
        _, intercept_ = self._compute_parameters()
        return intercept_

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# Generate a synthetic regression dataset
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100_000, n_features=50, n_informative=35, bias=8)
X[:, 8] += 5  # Adding a shift to feature 8 for testing purposes

model = IncrementalLinearRegression()

# Simulate updating the model in batches (e.g., 17 batches)
n_batches = 17
for index in np.array_split(np.arange(len(X)), n_batches):
    model.partial_fit(X[index], y[index])

# Compare with sklearn's LinearRegression model (using full data)
from sklearn.linear_model import LinearRegression

linear = LinearRegression().fit(X, y)

# Check if the results match (coefficients and intercept)
np.allclose(linear.coef_, model.coef_), np.allclose(linear.intercept_, model.intercept_)
# (True, True)
```

## Documentation

The documentation is available [here](https://batchstats.readthedocs.io).

## Requesting Additional Statistics

If you require additional statistics that are not currently implemented in `batchstats`, feel free to open an issue on the GitHub repository or submit a pull request with your suggested feature. We welcome contributions and feedback from the community to improve `batchstats` and make it more versatile for various data analysis tasks.
