=============
API Reference
=============

This page provides a reference for the classes available in the ``batchstats`` library.

Core Classes
------------

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


NaN-handling Classes
--------------------

These classes are designed to handle datasets that contain ``NaN`` values, similar to ``numpy``'s ``nan*`` functions. They compute statistics by ignoring ``NaN`` values.

.. autoclass:: batchstats.BatchNanSum
   :members: update_batch, __call__

.. autoclass:: batchstats.BatchNanMean
   :members: update_batch, __call__
