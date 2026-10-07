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
Top-k uses local partition before combining at most 2k retained values, then
sorts only the retained tail. Persistent memory is O(k times positions), plus
counts. Temporaries include the transposed/reshaped batch when a copy is needed,
its validity mask, a partition working copy, and the retained merge buffers.

``BatchNanSum`` and ``BatchNanMean`` can also be merged; means combine sums and
valid counts rather than averaging means. Empty positions still return NaN.
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

.. code-block:: python

   from batchstats import BatchNanTopK

   resumed = BatchNanTopK.load("tile-extremes.npz")
   resumed.update_batch(next_block)
   combined = resumed + another_tile_state  # same spatial positions only

Benchmarks
----------

Run ``python benchmarks/extremes.py`` to compare incremental selection, complete
sort, complete partition, fusion and saving in isolated processes. It emits JSON
with elapsed time, process peak RSS, persistent state bytes and tracked peak
allocations (including NumPy temporaries). Use ``--samples``, ``--positions``,
``--batch`` and ``--k`` to vary sizes; it covers float32/64 and missing data.
Peak RSS includes input generation and interpreter overhead; tracked allocations
start immediately before the measured operation. Results depend on hardware and
input sizes. Large cases stay outside CI.
Nineteen float32 values across 5.2 million cells occupy about 395 MB before counts
and temporaries; spatial tiling remains necessary.

An illustrative Windows / Python 3.14 / NumPy 2.5.1 run on 7 October 2026 used
8760 samples, 256 positions, 168-sample batches and k=19 (complete data):

.. list-table:: Measured time and peak tracked allocations
   :header-rows: 1

   * - Operation
     - float32 time / peak
     - float64 time / peak
   * - Incremental
     - 0.078 s / 1.32 MB
     - 0.064 s / 1.53 MB
   * - Complete sort
     - 0.035 s / 17.98 MB
     - 0.055 s / 35.95 MB
   * - Complete partition
     - 0.024 s / 11.21 MB
     - 0.042 s / 20.18 MB
   * - Merge
     - 0.00030 s / 0.129 MB
     - 0.00034 s / 0.208 MB
   * - Save
     - 0.0038 s / 0.189 MB
     - 0.0038 s / 0.227 MB

Persistent states including counts were 21,504 and 40,960 bytes respectively.
With missing values every seventh sample at every third position, incremental
times were 0.058 / 0.064 s with essentially identical allocation peaks. These
small cases demonstrate the memory/time tradeoff, not a general speed advantage.

API
---

.. autoclass:: batchstats.BatchTopK
   :members:

.. autoclass:: batchstats.BatchNanTopK
   :members:

.. autofunction:: batchstats.required_k
