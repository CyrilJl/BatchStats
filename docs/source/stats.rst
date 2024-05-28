BatchStat classes
=================

The class ``BatchStat`` is the parent class from which other classes inherit. It allows for the factorization of the ``_process_batch`` method, which removes samples containing at least one NaN and keeps track of the number of samples seen by the class up to date.

.. autoclass:: batchstats.BatchStat

The following classes inherit from ``BatchStat``, and enable the user to compute various statistics over batch-accessed data:

.. automodule:: batchstats
    :members: BatchCov, BatchMax, BatchMean, BatchMin, BatchSum, BatchVar, BatchStd, BatchPeakToPeak
    :undoc-members:
    :show-inheritance: