=============
Core Classes
=============

These are the standard classes for computing statistics on datasets. If your data contains ``NaN`` values, they will be handled by removing the entire sample (row) that contains them.

Non-Weighted Statistics
=======================

.. admonition:: BatchSum
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchSum
      :members: __init__, update_batch, __call__

.. admonition:: BatchMean
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchMean
      :members: __init__, update_batch, __call__

.. admonition:: BatchVar
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchVar
      :members: __init__, update_batch, __call__

.. admonition:: BatchStd
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchStd
      :members: __init__, update_batch, __call__

.. admonition:: BatchCov
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchCov
      :members: __init__, update_batch, __call__

.. admonition:: BatchCorr
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchCorr
      :members: __init__, update_batch, __call__

.. admonition:: BatchMin
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchMin
      :members: __init__, update_batch, __call__

.. admonition:: BatchMax
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchMax
      :members: __init__, update_batch, __call__

.. admonition:: BatchPeakToPeak
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchPeakToPeak
      :members: __init__, update_batch, __call__

Weighted Statistics
===================

These classes are used for computing weighted statistics on datasets.

.. admonition:: BatchWeightedSum
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchWeightedSum
      :members: __init__, update_batch, __call__

.. admonition:: BatchWeightedMean
   :class: dropdown

   .. autoclass:: batchstats.stats.BatchWeightedMean
      :members: __init__, update_batch, __call__
