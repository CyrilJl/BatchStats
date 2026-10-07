Review of optional xarray support — 2026-10-07
==============================================

Initial review scope: ``main...0f30f58`` on ``feat/optional-xarray``. The findings
and initial measurements below describe that revision, before corrections.
Its existing suite passed 631 tests; two additional reproductions exposed
correctness gaps. Both bugs have since been fixed, along with the copy
optimizations described in the follow-up section at the end of this report.

Findings
--------

P1 — A coordinate snapshot can still depend on a mutable input file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Location: ``batchstats/xarray.py``, ``_Layout.from_array``, line 67.

``coords.copy(deep=True)`` does not materialize NetCDF-backed auxiliary
coordinates. The resulting layout can retain a lazy reference to the first
input file. Closing the Dataset does not detach that reference.

Reproduced with an auxiliary ``aux(x)`` coordinate:

* After a successful update, closing and removing the input makes
  ``stat().aux.values`` raise ``FileNotFoundError``. Saving the checkpoint
  also needs those coordinate values and therefore the original file.
* More seriously, replacing the first file with a different ``aux`` coordinate
  before the next update makes the stored layout read the replacement values.
  Updating with that replacement is accepted, although the non-reduced
  coordinate changed from ``[10, 20]`` to ``[100, 200]``. The accumulator now
  contains both batches under the new labels. The coordinate check has been
  bypassed silently.

Minimal setup for the second case::

    first = xr.DataArray(
        np.arange(6.).reshape(3, 2), dims=("time", "x"), name="a",
        coords={"aux": ("x", [10., 20.])},
    )
    first.to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4") as batch:
        stat = BatchMean("time").update_batch(batch)
    path.unlink()
    first.assign_coords(aux=("x", [100., 200.])).to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4") as batch:
        stat.update_batch(batch)  # Should reject the changed coordinate.
    print(stat().aux.values)     # [100., 200.]; first batch labels were lost.

Suggested correction: materialize and independently own retained coordinate
values once, when establishing a layout. Do not load the entire input Dataset
just to snapshot coordinates. Subsequent batches only need coordinate
validation. Add tests for input deletion/replacement and checkpoint saving
after the source is removed, using an auxiliary rather than an index coordinate.

P2 — Covariance/correlation reject dimensions without coordinate variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Location: ``batchstats/xarray.py``, ``BatchCov._matrix_layout``, lines 501–504.

The rename mapping includes every remaining dimension, but the coordinate-only
Dataset may contain neither that dimension nor its coordinate. Renaming a name
absent from that Dataset raises before any data can be accumulated::

    a = xr.DataArray(np.arange(6.).reshape(3, 2), dims=("time", "x"))
    BatchCov("time").update_batch(a)
    # ValueError: cannot rename 'x' because it is not a variable or dimension
    # in this dataset

``BatchCorr`` has the same failure. Explicit ``x`` labels are optional in
xarray. Suggested correction: filter the mapping passed to the coordinate
Dataset's ``rename`` to existing coordinate variables/dimensions, while
retaining the full output-dimension mapping. Cover paired/unpaired inputs,
DataArrays/Datasets and checkpoint round trips without explicit labels.

Copies and memory
-----------------

There are several avoidable allocations, but not every defensive copy should
be removed:

* ``_prepare`` (lines 151–160) calls ``_Layout.from_array`` for every variable
  of every batch. It deep-copies retained coordinates and attributes, checks
  them, and discards the candidate layout in favour of the original. Validation
  can inspect input coordinates without taking a new ownership snapshot.
* Shared Dataset coordinates are separately snapshotted for each variable.
  With three variables and two float64 1024×1024 auxiliary coordinates, the
  input has 16 MiB of unique coordinate arrays but the stored layouts have
  48 MiB. A shared internal coordinate snapshot could avoid this multiplication.
* ``__add__`` (lines 264–267) deep-copies the complete left accumulator, then
  discards its copied kernels and deep-copies the merged kernels again. Build
  the output shell/layout snapshots separately. The NumPy kernels sometimes
  return an operand when the other is empty: those aliasing cases must still
  be copied.
* Checkpoints serialize the same ``BatchNanMean`` count array twice: once on
  the parent and once on its ``BatchNanSum`` child. A tiny float64 example with
  three output cells contains 48 bytes of unique kernel arrays but exports
  72 bytes. Restoration also loses the original count-array alias. Array
  identity memoization or an explicit composite-state schema would avoid it.
* ``load`` eagerly reads all NPZ arrays, then ``from_state`` copies the decoded
  arrays again. Copying caller-owned arrays is appropriate for ``from_state``;
  a private consuming path for arrays freshly read by ``load`` could lower its
  peak memory without changing public ownership guarantees.
* ``transpose(...).to_numpy().reshape(...)`` is a view for an in-memory,
  contiguous ``(time, y, x)`` array reduced over ``time``. Reducing ``(time, x)``
  instead was confirmed to allocate a complete batch copy during reshape.
  Avoiding that generally requires passing the original reduction axes to the
  NumPy kernels, with attention to their NaN filtering conventions.

Keep the independent coordinates/attributes on returned results and the
transactional update guarantee. The kernel copy in ``_update`` protects the
previous state if a later Dataset variable fails; replacing it needs an
equivalent commit/rollback design, not just a shallow copy.

Isolated validation experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an already initialized three-variable Dataset, shape ``(1, 1024, 1024)``,
with the two 2D coordinates above, twenty repetitions of ``_prepare`` gave:

=========================== =================== =======================
Validation implementation   Median time         Peak traced allocation
=========================== =================== =======================
Current branch              34.10 ms             20.01 MiB
No new coordinate snapshot  21.53 ms              4.01 MiB
=========================== =================== =======================

The second row used a temporary, in-process replacement of the candidate
layout constructor; no production code was changed. It validates the same
coordinate values for this workload, but is not a complete proposed patch.
This is a measurement of validation only, not end-to-end speedup. Allocation
measurement uses tracemalloc separately from timing; it is distinct from the
OS RSS measurements below. The disposable reproduction script and raw output
are in ``build/xarray_review_probes.py`` and
``build/xarray-benchmark/review-probes.json``.

NetCDF benchmark protocol
-------------------------

Reproduce from the repository root::

    python -m pip install -e ".[xarray]" "dask[array]" netCDF4 psutil
    python benchmarks/xarray_netcdf.py --output build/xarray-benchmark
    # Reuse generated inputs for another run:
    python benchmarks/xarray_netcdf.py --output build/xarray-benchmark --reuse

The script creates synthetic data only, in its output directory. By default
it refuses to overwrite an existing input directory; use ``--reuse`` or choose
a new output directory. The benchmark dependencies are not added to the
library's mandatory dependencies.

* Ten uncompressed, contiguous NetCDF4 files. Each has one float32 variable
  of shape ``(time=32, y=1024, x=1024)``, shared 1D spatial indexes and two
  static 2D float32 coordinates. There are 1.25 GiB of variable data and
  1,426,231,540 bytes on disk including coordinates/headers.
* Approximately 1.03% of values are NaN, plus one all-NaN spatial cell.
  Batchstats uses ``BatchNanMean("time")``; xarray uses
  ``dataset.mean("time", skipna=True, keep_attrs=False).compute()``. Ordinary
  ``BatchMean`` removes whole samples containing NaNs and is not equivalent.
* Batchstats opens/closes each file separately with ``chunks=None, cache=False``.
  The sliced variant feeds four time steps at a time. The final result is
  loaded, including its coordinates, inside the timed region.
* ``open_mfdataset`` uses ``engine="netcdf4"`` and default combination by
  coordinates. Its default chunks in this experiment are one 32-step chunk
  per file. Compare the default threaded scheduler, a single-threaded
  scheduler and explicit four-step chunks with the default threaded scheduler.
* Each measurement runs in a fresh subprocess. Imports and backend discovery
  are outside timing; opening files, reading, coordinate checks, computing and
  closing are included. No output NetCDF is written. The OS process peak RSS
  includes Python, imports, NumPy/native allocations and threads, but does not
  charge the system-wide filesystem cache to the process. It is not the Python
  heap or the size of the input Dataset.
* One unmeasured warm-up per method, then three measurements per method with
  deterministically shuffled order. OS filesystem cache is warm; this is not
  a cold-disk benchmark. Generation/reference construction are not timed.
* Each result is checked against a separately accumulated float64 reference,
  including matching NaN locations. Batchstats returns float64 for these
  float32 inputs; xarray returns float32. All sums still use the respective
  implementations' default accumulation rules, so comparisons use explicit
  tolerances rather than bitwise equality.

Machine: Windows 11, Python 3.14.6, 8 physical / 16 logical CPUs, 63.9 GiB RAM.
NumPy 2.5.1, xarray 2026.9.0, Dask 2026.8.0, pandas 3.0.6, netCDF4 1.7.4
(netCDF 4.9.3 / HDF5 1.14.6). The installed editable package metadata still
reports batchstats 0.6; the code actually measured is the repository's xarray
module at commit ``0f30f58``. Its path and SHA-256 are recorded in the JSON.

NetCDF benchmark results
------------------------

Medians of three runs; RSS is the median of each process's peak. The baseline
after imports/backend discovery is approximately 102 MiB for every method.

.. list-table:: Median timings and peak resident memory
   :header-rows: 1

   * - Method
     - Time (s)
     - Peak RSS (MiB)
     - Time range (s)
   * - Batchstats, one file per batch
     - 3.118
     - 306.3
     - 3.100–3.169
   * - Batchstats, four-step batches
     - 4.810
     - 166.1
     - 4.786–4.870
   * - open_mfdataset, default threads
     - 2.659
     - 962.2
     - 2.604–2.832
   * - open_mfdataset, single thread
     - 3.565
     - 464.0
     - 3.562–3.665
   * - open_mfdataset, four-step chunks
     - 2.874
     - 369.1
     - 2.842–2.885

On this workload, file-wise batchstats uses 68% less peak RSS than the default
threaded open_mfdataset calculation, with 17% more elapsed time. Dask with small
chunks closes much of the memory gap: 369 MiB versus batchstats' 306 MiB, while
still finishing slightly earlier. Single-threaded Dask uses more memory and
time than file-wise batchstats in this run. Batchstats' smaller batches achieve
the lowest RSS but pay for more updates and coordinate validations.

All 15 measured outputs pass the reference comparison. The maximum absolute
error is 3.29e-7 or less on values between zero and one. These results describe
this particular uncompressed, warm-cache, one-variable workload; compressed
files, storage latency, variable count, chunk layout and scheduling can change
the ranking. open_mfdataset does not inherently require loading all ten files
simultaneously.

Initial timings, per-process RSS, numerical errors, software versions and
actual Dask chunks are kept at
``benchmarks/results/xarray_netcdf_2026-10-07.json``. The latest run also writes
``build/xarray-benchmark/results.json``. The input files
remain under the ignored ``build`` directory and are not part of the branch.

Reference documentation:
`open_mfdataset <https://docs.xarray.dev/en/stable/generated/xarray.open_mfdataset.html>`_
and `Dataset.mean <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.mean.html>`_.

Follow-up: implemented corrections and measurements
---------------------------------------------------

Both correctness findings are fixed. Retained coordinates are materialized and
owned once, shared internally between Dataset variable layouts, and validated
without fresh snapshots on later updates. Covariance/correlation accept
dimensions without explicit labels. Unpaired matrix statistics reuse the same
layout for both sides. Merge copies metadata separately and only detaches
kernel results that alias an operand (including nested weighted/composite
kernels). Transactional updates and independently owned public results remain
unchanged.

The checkpoint codec writes shared numeric arrays once and restores their
sharing. ``load`` consumes its freshly read arrays directly; ``from_state``
still owns a copy of caller arrays. Version 1 remains unchanged and legacy
checkpoints with duplicated count arrays are supported. General transpose /
reshape copies for non-contiguous multi-axis reductions were not changed.

The end-to-end benchmark was rerun on the same ten input files, with the same
script and three fresh processes per method. Source SHA-256 and all per-run
measurements are recorded in
``benchmarks/results/xarray_netcdf_2026-10-07_optimized.json``.

.. list-table:: End-to-end comparison after corrections
   :header-rows: 1

   * - Method
     - Before time / peak RSS
     - After time / peak RSS
   * - Batchstats, one file per batch
     - 3.118 s / 306.3 MiB
     - 3.077 s / 313.0 MiB
   * - Batchstats, four-step batches
     - 4.810 s / 166.1 MiB
     - 4.425 s / 173.2 MiB
   * - open_mfdataset, default threads
     - 2.659 s / 962.2 MiB
     - 2.739 s / 997.0 MiB
   * - open_mfdataset, single thread
     - 3.565 s / 464.0 MiB
     - 3.674 s / 464.0 MiB
   * - open_mfdataset, four-step chunks
     - 2.874 s / 369.1 MiB
     - 3.019 s / 367.3 MiB

The sliced workload is about 8% faster. The file-wise timing change is small
(about 1%); the unchanged Dask timings vary between runs too. Peak RSS grows
by roughly 7 MiB for batchstats because the first file's retained coordinates
now really reside in memory rather than retaining a lazy file reference.
This correctness cost should not be mistaken for an optimization failure.
All numerical comparisons pass as before.

The new ``benchmarks/xarray_allocations.py`` measures coordinate storage,
validation, merge and checkpoints on the three-variable in-memory workload
described above. For a direct comparison, the same script was run in separate
processes with the two optional modules from ``0f30f58`` and with the corrected
modules. Raw results are in
``benchmarks/results/xarray_allocations_2026-10-07.json``.

.. list-table:: Same allocation probe, before and after
   :header-rows: 1

   * - Measurement
     - Before
     - After
   * - Unique retained coordinate buffers
     - 48 MiB
     - 16 MiB
   * - Validation median (20 calls)
     - 34.17 ms
     - 22.47 ms
   * - Validation peak traced allocations
     - 20.01 MiB
     - 4.01 MiB
   * - Arrays stored in a checkpoint
     - 108 MiB
     - 52 MiB
   * - to_state peak traced allocations
     - 108.02 MiB
     - 52.02 MiB
   * - load peak traced allocations
     - 220.13 MiB
     - 53.11 MiB
   * - Merge peak traced allocations
     - 132.02 MiB
     - 52.01 MiB

These allocation peaks come from tracemalloc, not process RSS, and include
the newly constructed result. The input state already exists before tracing.
They must not be compared directly with the NetCDF benchmark's RSS column.

Validation: 704 tests pass on the current environment, with 90.27% total
coverage. The 418 optional xarray/checkpoint tests also pass on xarray 2023.1
with Python 3.10 (the final added test was run separately). Without xarray,
286 core tests pass and the two optional test modules are skipped. Ruff checks,
format checks and the Sphinx build with warnings treated as errors pass.
The current netCDF4 backend emits six upstream NumPy 2.5 deprecation warnings
in the NetCDF regression tests; no test fails.
