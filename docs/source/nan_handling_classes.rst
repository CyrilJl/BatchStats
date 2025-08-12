======================
NaN-handling Classes
======================

These classes are designed to handle datasets that contain ``NaN`` values, similar to ``numpy``'s ``nan*`` functions. They compute statistics by ignoring ``NaN`` values.

.. autoclass:: batchstats.BatchNanSum
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchNanMean
   :members: update_batch, __call__
