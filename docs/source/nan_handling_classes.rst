======================
NaN-handling Classes
======================

These classes are designed to handle datasets that contain ``NaN`` values, similar to ``numpy``'s ``nan*`` functions. They compute statistics by ignoring ``NaN`` values.

.. autoclass:: batchstats.nanstats.BatchNanSum
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.nanstats.BatchNanMean
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.nanstats.BatchNanMin
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.nanstats.BatchNanMax
   :members: __init__, update_batch, __call__

.. autoclass:: batchstats.nanstats.BatchNanPeakToPeak
   :members: __init__, update_batch, __call__
