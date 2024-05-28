BatchNaNStat classes
====================

The class ``BatchNanStat`` is the parent class from which other classes that can treat NaNs inherit. It allows for the factorization of the ``_process_batch`` method, which keeps track of the number of NaNs per feature.

.. autoclass:: batchstats.BatchNanStat

The following classes inherit from ``BatchNanStat``:

.. automodule:: batchstats
    :members: BatchNanMean, BatchNanSum
    :undoc-members:
    :show-inheritance: