BatchStats
==========

BatchStats computes statistics on data that arrives in batches, so you can stream or process large datasets without loading everything into memory. Feed batches with ``update_batch``, then call the object to get the final result.

Installation
------------

.. code-block:: console

   pip install batchstats

Or with ``conda``/``mamba``:

.. code-block:: console

   conda install -c conda-forge batchstats

Quick Start
-----------

.. code-block:: python

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

Available Statistics
--------------------

* ``BatchSum`` / ``BatchNanSum``
* ``BatchWeightedSum``
* ``BatchMean`` / ``BatchNanMean``
* ``BatchWeightedMean``
* ``BatchMin`` / ``BatchNanMin``
* ``BatchMax`` / ``BatchNanMax``
* ``BatchPeakToPeak`` / ``BatchNanPeakToPeak``
* ``BatchVar``
* ``BatchStd``
* ``BatchCov``
* ``BatchCorr``

Docs: https://batchstats.readthedocs.io

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   api
   incremental_computing
