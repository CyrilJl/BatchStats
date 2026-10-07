Extreme ranks, quantiles and checkpoints
=========================================

``BatchTopK(k, axis=0, largest=True)`` retains exact extrema with multiplicity.
``BatchNanTopK`` ignores NaNs independently at each position; ``BatchTopK``
rejects a batch containing NaNs before changing its state. Infinities are valid
observations. Complex, object and masked inputs are unsupported: decode masks
explicitly to NaNs first.

Inputs follow the existing ``np.atleast_2d`` convention: a one-dimensional array
becomes one row, so use ``data[:, None]`` for a single time series. ``axis`` may
be an integer, a tuple (including negative axes), or ``None`` for all axes.
Remaining dimensions keep their original order. Every batch must represent
the same positions. Reduced dimensions may vary in length.

Calling the accumulator returns an independent masked array with shape
``(k, *remaining_shape)``. Rank 1 is the largest value, or smallest when
``largest=False``. Missing ranks are masked, including all ranks at an empty
position. Values retain their NumPy dtype; no float32 conversion or rounding is
introduced. ``n_samples`` is an integer array counting valid observations per
position. An accumulator never updated raises ``NoValidSamplesError`` on read;
an empty batch initializes shapes and produces masked ranks and NaN quantiles.

Read ``rank(r)`` with one-based ranks up to k. ``quantile(q, method="linear")``
accepts a scalar q in [0, 1] and returns float64 results using the actual count
at every position. Other methods are rejected. If any nonempty position needs
a discarded rank, it raises ``ValueError`` rather than approximating. Empty
positions yield NaN. For N=1 or an integer quantile position the rank value is
returned directly (including infinities); fractional interpolation follows IEEE
arithmetic and may yield NaN for infinities. Thus infinity endpoints deliberately
avoid NumPy versions' NaN results caused by interpolation at integer positions.

``required_k(q, max_samples, largest=True)`` dimensions the retained tail:
for the upper tail it returns ``N - floor((N - 1) * q)``; for the lower tail,
``ceil((N - 1) * q) + 1``. P99 needs five upper values for 365 or 366 samples;
P99.8 needs nineteen for 8760 or 8784. The helper does not enforce an observation
limit: reads check actual counts and refuse insufficient capacity. Increasing
capacity after discarding values cannot recover them.

Streaming a spatial tile
------------------------

.. code-block:: python

   import numpy as np
   from batchstats import BatchNanMean, BatchNanTopK, required_k

   def summarize(blocks):
       mean = BatchNanMean(axis=0)
       extremes = BatchNanTopK(required_k(.998, 8784), axis=0)
       for block in blocks:  # each block has shape (time, y, x)
           if np.ma.isMaskedArray(block):
               block = block.astype(np.float64).filled(np.nan)
           mean.update_batch(block)
           extremes.update_batch(block)
       extremes.save("tile-extremes.npz")
       return mean(), extremes.quantile(.998)

   rng = np.random.default_rng(0)
   blocks = (rng.normal(size=(24, 8, 10)) for _ in range(365))
   annual_mean, annual_p998 = summarize(blocks)

For NetCDF, read bounded temporal blocks and spatial tiles, decode missing values
and masks, and preserve dimension order and coordinate correspondence. Finalize
or checkpoint each tile separately. Never feed another spatial tile as new time
samples into the same state. Calendar rules, units, coverage, provenance, duplicate
days, checksums and retention policies belong to the consumer. No xarray or NetCDF
dependency is required in BatchStats.

Fusion and precision
--------------------

``a + b`` returns independent arrays, including when either operand is
uninitialized. The type, axis parameter and remaining shape must match;
top-k also requires equal capacity, direction and input dimensionality.
Axis parameters are compared literally, as in the existing package: ``0`` and
``-3`` or differently ordered tuples are not interchangeable in a merge.
Top-k partitions contiguous rank vectors before combining at most 2k retained
values, then sorts the retained tail. Fully populated local tails do not need
an intermediate sort. For k=1, a min/max reduction avoids the partition working
copy. Persistent memory is O(k times positions), plus counts. Temporaries include
the transposed/reshaped batch when a copy is needed, its NaN mask for floating
inputs, a partition working copy, and the retained merge buffers. The public
state remains contiguous for efficient reads and checkpoint writes. Reading
``rank(r)`` copies only that rank, using O(positions) additional memory.

``BatchNanSum`` and ``BatchNanMean`` can also be merged; means combine sums and
valid counts rather than averaging means. Empty positions still return NaN.
Updates reuse one NaN mask for the sum and count reductions without making a
numeric copy of the batch. Complete batches use ordinary sums and derive counts
from their shape; integer and boolean inputs need no NaN mask.
Sum reductions follow NumPy's default accumulation dtype (small integers are
promoted); updates and merges use NumPy result-type promotion. Integer overflow
is possible, and mixing large integers with floating dtypes can lose precision.
Choose input dtypes accordingly. Floating addition order can change the last
bits, so parallel sums and means require numerical tolerances.
For a fixed common input dtype, selected values are identical across batch
splits and merge trees. No merge is idempotent: merging a day twice counts it twice.

Checkpoint contract
-------------------

Only ``BatchNanSum``, ``BatchNanMean``, ``BatchTopK`` and ``BatchNanTopK`` expose
``to_state()``, ``from_state(state)``, ``save(path)`` and ``load(path)``.
The state is a dictionary containing ``metadata`` and ``arrays``. Metadata is
JSON-compatible and includes schema version 1, statistic type, parameters,
array shapes and dtype strings, plus input dimensionality for top-k. Arrays
contain valid counts and either sums or selected values; uninitialized states
have no arrays. Both export and restoration copy arrays. Restoration rejects
unknown types/versions, incompatible shapes/dtypes, invalid counts and unordered
retained values. This checks structure, not authenticity or provenance.

``save`` stores JSON metadata as a Unicode scalar in an NPZ archive alongside
numeric arrays; ``load`` always uses ``allow_pickle=False``. Consumers may also
store ``state['metadata']`` in a separate JSON file and ``state['arrays']`` in
NPZ. File paths follow ``np.savez`` semantics (the .npz extension is appended
when absent). Atomic writes, integrity checks and overlap detection remain the
consumer's responsibility.
Saving writes the live arrays synchronously without first copying the whole
state; do not update an accumulator concurrently with saving it. ``to_state()``
and ``from_state()`` still return independent arrays.

.. code-block:: python

   from batchstats import BatchNanTopK

   resumed = BatchNanTopK.load("tile-extremes.npz")
   resumed.update_batch(next_block)
   combined = resumed + another_tile_state  # same spatial positions only

Benchmarks
----------

Run ``python benchmarks/extremes.py`` to compare incremental selection, complete
sort, complete partition, fusion, saving and single-rank reads in isolated processes. It emits JSON
with elapsed time, process peak RSS, persistent state bytes and tracked peak
allocations (including NumPy temporaries). Use ``--samples``, ``--positions``,
``--batch`` and ``--k`` to vary sizes; it covers float32/64 and missing data.
``--repeat`` controls repetitions (three by default), and ``--operations`` selects
operations. Timings are medians after warm-up, with allocation tracking disabled;
peak allocations are measured in a separate run. Peak RSS includes input generation,
warm-up and interpreter overhead. Results depend on hardware and
input sizes. Large cases stay outside CI.
Nineteen float32 values across 5.2 million cells occupy about 395 MB before counts
and temporaries; spatial tiling remains necessary.

Performance comparison on Windows / Python 3.14.6 / NumPy 2.5.1, 7 October 2026,
against commit ``4760457``, using the same benchmark harness for both versions:

.. code-block:: console

   python benchmarks/extremes.py --positions 1024 --repeat 5
   python benchmarks/reductions.py
   python benchmarks/reductions.py --axis 1
   python benchmarks/extremes.py --samples 128 --positions 65536 --k 64 --operations save rank

The top-k workload has 8760 samples, 1024 positions, 168-sample batches and k=19.
The reductions workload has 4096 samples, 1024 positions and 256-sample batches;
the table uses axis=0. Both use five timing repetitions. Missing data replaces
every seventh sample at every third position with NaN. MB denotes decimal MB.

.. list-table:: Time and peak tracked allocations, before to after
   :header-rows: 1

   * - Operation
     - Dtype / data
     - Time (ms)
     - Peak (MB)
   * - Incremental top-k
     - float32 / complete
     - 141.0 to 69.1
     - 1.036 to 0.862
   * - Incremental top-k
     - float64 / missing
     - 158.1 to 126.7
     - 1.880 to 1.878
   * - Merge top-k
     - float64 / complete
     - 0.762 to 0.396
     - 0.821 to 0.470
   * - NaN sum
     - float32 / complete
     - 11.21 to 1.80
     - 1.329 to 0.292
   * - NaN sum
     - float32 / missing
     - 13.46 to 9.17
     - 1.329 to 0.354
   * - NaN sum
     - float64 / missing
     - 18.89 to 11.05
     - 2.386 to 0.362
   * - NaN mean
     - float64 / missing
     - 20.55 to 11.53
     - 2.386 to 0.363

Persistent state sizes are unchanged. Incremental top-k with missing data is
faster, but its peak temporary memory is nearly unchanged in this workload.
For the separate large-state case (k=64, 65536 positions, float64, complete data,
three repetitions), saving improved from 31.28 to 23.48 ms and from 50.99 to
16.91 MB of tracked allocations. Reading one rank dropped from 37.75 to 0.59 MB;
it no longer allocates all k ranks and their masks. These are local measurements,
not performance guarantees for other machines, layouts or data distributions.

API
---

.. autoclass:: batchstats.BatchTopK
   :members:

.. autoclass:: batchstats.BatchNanTopK
   :members:

.. autofunction:: batchstats.required_k
