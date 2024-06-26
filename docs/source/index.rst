BatchStats's documentation
==========================

.. image:: https://img.shields.io/pypi/v/batchstats.svg
    :target: https://pypi.org/project/batchstats/

.. image:: https://anaconda.org/conda-forge/batchstats/badges/version.svg
    :target: https://anaconda.org/conda-forge/batchstats

.. toctree::
   :hidden:
   :maxdepth: 1

   api_reference
   future

Introduction
------------

``batchstats`` is a Python package designed for computing various statistics of data that arrive batch by batch, making it suitable for streaming input or handling data too large to fit in memory. The package relies solely on numpy, ensuring efficient computation and compatibility with numpy arrays.

Installation
------------

You can install ``batchstats`` using pip:

.. code-block:: bash

    pip install batchstats

Usage Example
-------------

Here's an example of how to use ``batchstats`` to compute batch mean and variance:

.. code-block:: python

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

``batchstats`` relies on two main methods: ``update_batch`` for processing a batch of data, and ``__call__`` for computing the statistic that represents all the data previously fed to the class via ``update_batch``.

``batchstats`` is also flexible in terms of input shapes. By default, statistics are applied along the first axis: the first dimension representing the samples and the remaining dimensions representing the features:

.. code-block:: python

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

However, similar to the associated functions in ``numpy``, users can specify the reduction axis or axes:

.. code-block:: python

    import numpy as np
    from batchstats import BatchMean

    data = [np.random.randn(24, 7, 128) for _ in range(100)]

    batchmean = BatchMean(axis=(0, 2))
    for batch in data:
        batchmean.update_batch(batch)
    batchmean().shape
    >>> (7,)

    batchmean = BatchMean(axis=2)
    for batch in data:
        batchmean.update_batch(batch)
    batchmean().shape
    >>> (24, 7)

Merging Two Objects
-------------------

In some cases, it is useful to process two different ``BatchStats`` objects from asynchronous I/O functions and then merge the statistics of both objects at the end. The ``batchstats`` library supports this functionality by allowing the simple addition of two objects. Under the hood, the necessary computations are performed to produce a resulting statistic that reflects the data from both input datasets, even imbalanced:

.. code-block:: python

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
    >>> True

The ``__add__`` method has been specifically overloaded to facilitate the merging of statistical objects in ``batchstats``, including ``BatchCov``, ``BatchMax``, ``BatchMean``, ``BatchMin``, ``BatchPeakToPeak``, ``BatchStd``, ``BatchSum``, and ``BatchVar``.

Performance
-----------

In addition to result accuracy, much attention has been given to computation times and memory usage. For example, calculating the variance using ``batchstats`` consumes less RAM and is faster than ``numpy.var``:

.. code-block:: python

    %load_ext memory_profiler
    import numpy as np
    from batchstats import BatchVar

    data = np.random.randn(100_000, 1000)
    print(data.nbytes/2**20)
    >>> 762.939453125

    %memit a = np.var(data, axis=0)
    >>> peak memory: 1604.63 MiB, increment: 763.35 MiB

    %memit b = BatchVar().update_batch(data)()    
    >>> peak memory: 842.62 MiB, increment: 0.91 MiB

    np.allclose(a, b)
    >>> True

    %timeit a = np.var(data, axis=0)
    >>> 510 ms ± 111 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    %timeit b = BatchVar().update_batch(data)()    
    >>> 306 ms ± 5.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

NaN handling possibility
------------------------

While the previous ``Batch*`` classes exclude every sample containing at least one NaN from the computations, the ``BatchNan*`` classes adopt a more flexible approach to handling NaN values, similar to `np.nansum`, `np.nanmean`, etc. Consequently, the outputted statistics can be computed from various numbers of samples for each feature:

.. code-block:: python

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
    >>> True

Available Classes/Stats
-----------------------

- ``BatchCov``: Compute the covariance matrix of two datasets (not necessarily square).
- ``BatchMax``: Compute the maximum value.
- ``BatchMean``: Compute the mean.
- ``BatchMin``: Compute the minimum value.
- ``BatchPeakToPeak``: Compute the difference between maximum and minimum.
- ``BatchSum``: Compute the sum.
- ``BatchStd``: Compute the standard deviation.
- ``BatchVar``: Compute the variance.