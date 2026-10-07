Optional xarray support
=======================

Install the extra to work with labelled arrays:

.. code-block:: console

   pip install "batchstats[xarray]"

Import the labelled classes from ``batchstats.xarray``. The regular classes in
``batchstats`` keep their existing NumPy API, and importing them does not import
xarray. This extra is optional at both installation and import time.

DataArrays and Datasets
-----------------------

.. code-block:: python

   import numpy as np
   import xarray as xr
   from batchstats.xarray import BatchNanMean

   data = xr.DataArray(
       np.arange(24, dtype=float).reshape(6, 4),
       dims=("time", "station"),
       coords={"time": np.arange(6), "station": ["A", "B", "C", "D"]},
       name="temperature",
       attrs={"units": "K"},
   )
   mean = BatchNanMean(dim="time", keep_attrs=True)
   mean.update_batch(data.isel(time=slice(0, 2)))
   mean.update_batch(data.isel(time=slice(2, None)))
   result = mean()  # DataArray(station), with station labels and units

   dataset = xr.Dataset({"temperature": data, "pressure": data * 2})
   means = BatchNanMean(dim="time")
   means.update_batch(dataset.isel(time=slice(0, 2)))
   means.update_batch(dataset.isel(time=slice(2, None)))
   result = means()  # Dataset containing temperature and pressure means

``update_batch`` returns the accumulator for chaining. ``dim`` accepts a string,
a tuple/list of strings, or ``None`` (the default, reducing all dimensions).
One-dimensional inputs reduce to scalar DataArrays. There is no implicit
two-dimensional promotion in the labelled output.

Dataset variables may have different dimensions. Each variable reduces the
intersection of its dimensions with ``dim``. Every variable must be numeric and
have at least one selected dimension; select the desired variables first when a
Dataset also contains static fields or strings. With ``dim=None``, scalar
variables are also supported and contribute one observation per batch.

Available classes
-----------------

The module exports ``BatchSum``, ``BatchMean``, ``BatchMin``, ``BatchMax``,
``BatchPeakToPeak``, ``BatchVar``, ``BatchStd``, ``BatchWeightedSum``,
``BatchWeightedMean``, ``BatchCov``, ``BatchCorr``, ``BatchTopK``, and the existing
NaN-aware counterparts: ``BatchNanSum``, ``BatchNanMean``, ``BatchNanMin``,
``BatchNanMax``, ``BatchNanPeakToPeak``, ``BatchNanTopK``.

``BatchVar``, ``BatchStd``, ``BatchCov`` and ``BatchCorr`` accept ``ddof=0``.
All classes accept ``keep_attrs=False``. When enabled, variable and Dataset
attributes are copied from the first batch (the left operand when merging).
Attributes are not recomputed; for example, variance units require updating by
the caller. Names and non-reduced coordinates are preserved independently of
``keep_attrs``.

Coordinate and batch contracts
------------------------------

* DataArray and Dataset batches cannot be mixed. A Dataset must contain the same
  variable names across all updates.
* Input dimension order may change between updates; arrays are transposed by
  name before computation. Dimension names must remain the same.
* Reduced dimensions can change size and coordinates. Batches contribute
  observations to the running statistic; overlapping labels are counted again.
* Non-reduced dimensions must keep the same sizes and coordinates, including
  label order. Auxiliary and scalar coordinates are checked too. Coordinates
  depending on any reduced dimension are removed from the output.
* Coordinates are never automatically reindexed or joined across batches.
  A mismatch raises an error. Reindex explicitly before calling batchstats if
  that is the intended operation.
* A rejected update leaves all variable accumulators unchanged. Updates stage
  copies of the bounded accumulator state before committing the new batch.

NaN handling follows the corresponding NumPy statistic. ``BatchNan*`` classes
ignore NaNs per output cell; ``BatchNanSum`` returns NaN when a cell has no valid
observations. Ordinary reductions discard an entire sample if any of its
non-reduced cells is NaN, separately for each Dataset variable. With multiple
reduced dimensions, their Cartesian product forms the sample axis. Pass
``assume_valid=True`` to ordinary reductions to bypass this filtering.
Weighted statistics propagate NaNs; ``BatchTopK`` rejects them.

``n_samples`` returns labelled counts, or a Dataset of counts. Ordinary
statistics have scalar counts per variable; NaN-aware and top-k statistics have
counts per remaining position. Weighted statistics and uninitialized labelled
accumulators return ``None``. Reading a statistic before any update raises
``NoValidSamplesError``. Empty/all-invalid input follows the underlying NumPy
statistic's behavior.

Weights
-------

.. code-block:: python

   from batchstats.xarray import BatchWeightedMean

   weights = xr.DataArray([1., 1., 2., 2., 3., 3.], dims="time",
                          coords={"time": data.time})
   mean = BatchWeightedMean(dim="time")
   mean.update_batch(data, weights=weights)
   result = mean()

Weights may be scalars, DataArrays with a subset of the variable's dimensions,
or Datasets containing the same variables as the input Dataset. DataArray
weights are broadcast by dimension name; shared coordinates and indexes must
match exactly, including along reduced dimensions within the current batch.
Use labelled weights rather than unlabelled NumPy arrays. A zero total weight
produces NaN for the weighted mean.

Top-k and quantiles
-------------------

.. code-block:: python

   from batchstats import required_k
   from batchstats.xarray import BatchNanTopK

   extremes = BatchNanTopK(required_k(0.99, 6), dim="time")
   extremes.update_batch(data)
   largest = extremes.rank(1)
   p99 = extremes.quantile(0.99)
   ranks = extremes()  # (rank, station); one-based rank coordinate

``largest=False`` retains minima. ``rank_dim="rank"`` selects the extra output
dimension's name; choose another name if it conflicts with an input dimension,
coordinate or Dataset variable. Missing ranks become NaN in xarray, which may
promote integer data to floating point. Fully populated integer ranks retain
their dtype. Exact quantiles raise an error when retained capacity is
insufficient. Only scalar quantiles with ``method="linear"`` are supported.

Covariance and correlation
--------------------------

.. code-block:: python

   from batchstats.xarray import BatchCov

   covariance = BatchCov(dim="time", ddof=1).update_batch(data)
   matrix = covariance()  # (station, station_2)

These classes compute matrices between features, as in the NumPy API.
Non-reduced dimensions are flattened into features for computation and restored
in the output. Dimensions and coordinates belonging to the second feature axis
receive a ``_2`` suffix. Rename conflicting input coordinates first.

``update_batch(batch, batch2)`` computes cross-covariance or cross-correlation.
Both inputs must use the same reduced dimension names, sizes and sample
coordinates. Their feature dimensions can differ. Choose either paired or
single-input updates for the lifetime of an accumulator. For Datasets, the
inputs must have the same variables and each variable produces its own matrix;
this does not compute cross-variable covariance between Dataset variables.

Merging and memory
------------------

.. code-block:: python

   left = BatchNanMean("time").update_batch(data.isel(time=slice(0, 2)))
   right = BatchNanMean("time").update_batch(data.isel(time=slice(2, None)))
   merged = left + right
   xr.testing.assert_allclose(merged(), data.mean("time"))

Merges require the same class, parameters, container type, variables and
non-reduced dimensions/coordinates. The order of non-reduced dimensions must
also match between independently initialized accumulators. Neither operand is
modified, and subsequent updates do not share state with the merged result.

Only statistics and non-reduced metadata are retained. Each update eagerly
converts the supplied batch to NumPy, including Dask-backed DataArrays. Slice
large or lazy datasets into suitably sized batches before calling
``update_batch``. This adapter does not schedule automatic Dask chunk reductions
and does not add Dask as a dependency. Matrix statistics require quadratic
feature storage; top-k storage scales with ``k`` times the remaining shape.

The labelled adapters currently support in-memory merging, but do not expose
the NumPy accumulators' ``save/load`` or ``to_state/from_state`` checkpoint API.
The existing NumPy checkpoint API remains available unchanged.
