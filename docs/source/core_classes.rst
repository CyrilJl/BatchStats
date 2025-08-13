=============
Core Classes
=============

These are the standard classes for computing statistics on datasets. If your data contains ``NaN`` values, they will be handled by removing the entire sample (row) that contains them.

.. autoclass:: batchstats.stats.BatchSum
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchMean
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchVar
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchStd
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchCov
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchCorr
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchMin
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchMax
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchPeakToPeak
   :members: __init__, update_batch, __call__

Weighted Statistics
===================

These classes are used for computing weighted statistics on datasets.

.. autoclass:: batchstats.stats.BatchWeightedSum
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.stats.BatchWeightedMean
   :members: __init__, update_batch, __call__
