BatchStats's documentation
==========================

.. image:: https://img.shields.io/pypi/v/batchstats.svg
    :target: https://pypi.org/project/batchstats/

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

Available Classes/Stats
-----------------------

- ``BatchCov``: Compute the covariance matrix of two datasets (not necessarily square).
- ``BatchMax``: Compute the maximum value.
- ``BatchMean``: Compute the mean.
- ``BatchMin``: Compute the minimum value.
- ``BatchSum``: Compute the sum.
- ``BatchVar``: Compute the variance.

For more details on usage and available classes, please refer to the documentation.

Performance
-----------

In addition to result accuracy, much attention has been given to computation times and memory usage. For example, calculating the variance using `batchstats` consumes less RAM and is faster than `numpy.var`:

.. code-block:: python

    %load_ext memory_profiler
    import numpy as np
    from batchstats import BatchVar

    data = np.random.randn(100_000, 1000)

    %memit a = np.var(data, axis=0)
    %memit b = BatchVar().update_batch(data)()
    np.allclose(a, b)

    %timeit a = np.var(data, axis=0)
    %timeit b = BatchVar().update_batch(data)()