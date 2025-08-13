=======================
Incremental Computing
=======================

``batchstats`` is built for scenarios where data arrives in batches or streams. This is often called incremental or online computing. Instead of loading the entire dataset into memory, which can be inefficient or impossible for very large datasets, ``batchstats`` processes data chunk by chunk.

This approach has several advantages:

- **Memory Efficiency**: It uses a small, constant amount of memory, regardless of the total dataset size.
- **Real-time Processing**: It's well-suited for real-time data streams where statistics need to be updated as new data arrives.
- **Numerical Stability**: For variance and covariance, `batchstats` uses Welford's online algorithm, which is more numerically stable than a naive two-pass approach.

Welford's Online Algorithm
--------------------------

Welford's online algorithm is a method for computing the variance (and by extension, standard deviation and covariance) in a single pass. It is less prone to catastrophic cancellation, which can be an issue with the naive two-pass algorithm when the variance is small compared to the mean.

The algorithm updates the mean and the sum of squared differences from the mean with each new data point (or batch of data points). This is the core of how ``BatchVar``, ``BatchStd``, and ``BatchCov`` work.

N-dimensional Arrays and Multiple Axes
--------------------------------------

All ``batchstats`` classes support n-dimensional ``numpy.ndarray`` inputs. You can specify the axis (or axes) to perform the reduction on, just like in ``numpy``.

For example, you can compute a statistic over a single axis:

.. code-block:: python

   from batchstats import BatchMean
   import numpy as np

   # 3D data
   data = np.random.rand(10, 5, 8)
   # mean over the second axis (axis=1)
   mean = BatchMean(axis=1).update_batch(data)()


Or over multiple axes by providing a tuple to the ``axis`` parameter:

.. code-block:: python

   # mean over the last two axes
   mean_multiple_axes = BatchMean(axis=(1, 2)).update_batch(data)()


Usage Examples
--------------

Here's a simple example of how to use `batchstats` to compute the mean and variance of a dataset that is processed in batches:

.. code-block:: python

    import numpy as np
    from batchstats import BatchMean, BatchVar

    # Imagine this is a stream of data
    data_stream = (np.random.randn(100, 10) for _ in range(10))

    batch_mean = BatchMean()
    batch_var = BatchVar()

    for batch in data_stream:
        batch_mean.update_batch(batch)
        batch_var.update_batch(batch)

    # Get the final statistics
    mean = batch_mean()
    variance = batch_var()

    print("Mean shape:", mean.shape)
    print("Variance shape:", variance.shape)

Merging Statistics
------------------

Sometimes, you might process different parts of your data in parallel and need to combine the results. `batchstats` allows you to merge two statistical objects using the ``+`` operator:

.. code-block:: python

    import numpy as np
    from batchstats import BatchCov

    data = np.random.randn(25_000, 50)
    data1 = data[:10_000]
    data2 = data[10_000:]

    # Process the whole dataset at once
    cov_total = BatchCov().update_batch(data)

    # Process in two separate parts
    cov1 = BatchCov().update_batch(data1)
    cov2 = BatchCov().update_batch(data2)

    # Merge the two parts
    cov_merged = cov1 + cov2

    # The results should be very close
    assert np.allclose(cov_total(), cov_merged())
