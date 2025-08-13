=============
Core Classes
=============

These are the standard classes for computing statistics on datasets. If your data contains ``NaN`` values, they will be handled by removing the entire sample (row) that contains them.

.. autoclass:: batchstats.BatchSum
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchMean
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchVar
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchStd
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchCov
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchMin
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchMax
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.BatchPeakToPeak
   :members: __init__, update_batch, __call__
