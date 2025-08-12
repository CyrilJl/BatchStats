=============
Core Classes
=============

These are the standard classes for computing statistics on datasets. If your data contains ``NaN`` values, they will be handled by removing the entire sample (row) that contains them.

.. autoclass:: batchstats.BatchSum
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchMean
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchVar
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchStd
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchCov
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchMin
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchMax
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchPeakToPeak
   :members: update_batch, __call__
