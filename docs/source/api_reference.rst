.. BatchStats API Reference

API Reference
=============

The class ``BatchStat`` is the parent class from which other classes inherit. It allows for the factorization of the ``_process_batch`` method, which removes samples containing at least one NaN and keeps track of the number of samples seen by the class up to date.

.. autoclass:: batchstats.BatchStat

The following classes inherit from ``BatchStat``, and enable the user to compute various statistics over batch-accessed data:

.. automodule:: batchstats
    :members: BatchCov, BatchMax, BatchMean, BatchMin, BatchSum, BatchVar, BatchStd, BatchPeakToPeak
    :undoc-members:
    :show-inheritance:


The class ``BatchNanStat`` is the parent class from which other classes that can treat NaNs inherit. It allows for the factorization of the ``_process_batch`` method, which keeps track of the number of NaNs per feature.

.. autoclass:: batchstats.BatchNanStat

The following classes inherit from ``BatchNanStat``:

.. automodule:: batchstats
    :members: BatchNanMean, BatchNanSum
    :undoc-members:
    :show-inheritance:
    :no-index: