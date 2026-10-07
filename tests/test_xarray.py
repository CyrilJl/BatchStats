"""Numerical and labelled streaming contracts of the optional xarray API."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from batchstats import NoValidSamplesError  # noqa: E402
from batchstats import xarray as bx  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(20)
    return xr.DataArray(
        rng.normal(size=(11, 3, 2)),
        dims=("time", "y", "x"),
        name="temperature",
        coords={
            "time": np.arange(11),
            "y": [10, 20, 30],
            "x": [1, 2],
            "latitude": (("y", "x"), np.arange(6).reshape(3, 2)),
            "source": "sensor",
            "time_aux": ("time", np.arange(11) * 2),
        },
        attrs={"units": "K"},
    )


REDUCTIONS = [
    (bx.BatchSum, "sum"),
    (bx.BatchMean, "mean"),
    (bx.BatchMin, "min"),
    (bx.BatchMax, "max"),
    (bx.BatchVar, "var"),
    (bx.BatchStd, "std"),
    (bx.BatchNanSum, "sum"),
    (bx.BatchNanMean, "mean"),
    (bx.BatchNanMin, "min"),
    (bx.BatchNanMax, "max"),
    (bx.BatchPeakToPeak, "ptp"),
    (bx.BatchNanPeakToPeak, "ptp"),
]


def expected_reduction(data, op, dim):
    if op == "ptp":
        return data.max(dim) - data.min(dim)
    return getattr(data, op)(dim)


@pytest.mark.parametrize("cls,op", REDUCTIONS)
@pytest.mark.parametrize("dim", ["time", ["time", "x"], None])
@pytest.mark.parametrize("dataset", [False, True])
def test_reductions_streaming_and_merge(data, cls, op, dim, dataset):
    if dataset:
        data = xr.Dataset({"temperature": data, "other": data.isel(x=0, drop=True).drop_vars("latitude") * 2})
    first, second = data.isel(time=slice(0, 4)), data.isel(time=slice(4, None))
    accumulator = cls(dim=dim)
    assert accumulator.update_batch(first) is accumulator
    accumulator.update_batch(second.transpose(..., "time"))
    expected = expected_reduction(data, op, dim)
    xr.testing.assert_allclose(accumulator(), expected)
    a, b = cls(dim=dim).update_batch(first), cls(dim=dim).update_batch(second)
    merged = a + b
    xr.testing.assert_allclose(merged(), expected)
    a.update_batch(first)
    b.update_batch(second)
    xr.testing.assert_allclose(merged(), expected)


@pytest.mark.parametrize("cls,op", REDUCTIONS)
@pytest.mark.parametrize("scalar", [False, True])
def test_one_dimensional_and_scalar(cls, op, scalar):
    data = xr.DataArray(3.0, name="a") if scalar else xr.DataArray([1.0, 3.0, 8.0], dims="time", name="a")
    result = cls().update_batch(data)()
    assert result.dims == ()
    xr.testing.assert_allclose(result, expected_reduction(data, op, None))


@pytest.mark.parametrize("cls,op", [(c, o) for c, o in REDUCTIONS if "Nan" in c.__name__])
def test_nan_values_and_counts(data, cls, op):
    data.values[0, 0, 0] = np.nan
    data.values[:, 1, 0] = np.nan
    accumulator = cls(dim="time")
    accumulator.update_batch(data.isel(time=slice(0, 2)))
    accumulator.update_batch(data.isel(time=slice(2, None)))
    expected = expected_reduction(data, op, "time")
    if op == "sum":
        expected = expected.where(data.count("time") > 0)
    xr.testing.assert_allclose(accumulator(), expected)
    xr.testing.assert_equal(accumulator.n_samples, data.count("time"))


def test_default_nan_policy_drops_whole_samples(data):
    data.values[0, 0, 0] = np.nan
    result = bx.BatchMean("time").update_batch(data)
    xr.testing.assert_allclose(result(), data.isel(time=slice(1, None)).mean("time"))
    assert result.n_samples.item() == 10
    propagated = bx.BatchMean("time").update_batch(data, assume_valid=True)()
    assert np.isnan(propagated.sel(y=10, x=1))


def test_metadata_and_output_ownership(data):
    accumulator = bx.BatchMean("time", keep_attrs=True).update_batch(data)
    output = accumulator()
    xr.testing.assert_identical(output, data.mean("time", keep_attrs=True))
    assert "time_aux" not in output.coords
    output.attrs["units"] = "changed"
    output.coords["latitude"].values[0, 0] = -99
    output.values[:] = 0
    xr.testing.assert_identical(accumulator(), data.mean("time", keep_attrs=True))
    data.attrs["units"] = "changed"
    assert accumulator().attrs == {"units": "K"}
    dataset = data.to_dataset()
    dataset.attrs = {"title": "weather"}
    result = bx.BatchMean("time", keep_attrs=True).update_batch(dataset)()
    assert result.attrs == dataset.attrs
    assert bx.BatchMean("time").update_batch(dataset)().attrs == {}


@pytest.mark.parametrize("change", ["coords", "shape", "dims", "aux", "missing_coord", "new_coord"])
def test_reject_incompatible_batches_without_mutation(data, change):
    accumulator = bx.BatchMean("time").update_batch(data)
    before = accumulator()
    if change == "coords":
        bad = data.isel(y=slice(None, None, -1))
    elif change == "shape":
        bad = data.isel(y=slice(0, 2))
    elif change == "dims":
        bad = data.rename(y="z")
    elif change == "aux":
        bad = data.assign_coords(latitude=data.latitude + 1)
    elif change == "missing_coord":
        bad = data.drop_vars("latitude")
    else:
        bad = data.assign_coords(new="label")
    with pytest.raises(ValueError):
        accumulator.update_batch(bad)
    xr.testing.assert_identical(accumulator(), before)


def test_dataset_update_is_transactional(data):
    dataset = xr.Dataset({"first": data, "last": data + 1})
    accumulator = bx.BatchTopK(2, "time").update_batch(dataset)
    before = accumulator()
    bad = dataset.copy(deep=True)
    bad["last"].values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="rejects NaNs"):
        accumulator.update_batch(bad)
    xr.testing.assert_identical(accumulator(), before)
    with pytest.raises(ValueError, match="variables"):
        accumulator.update_batch(dataset.drop_vars("last"))
    with pytest.raises(TypeError, match="numeric"):
        accumulator.update_batch(dataset.assign(last=dataset["last"].astype(str)))
    xr.testing.assert_identical(accumulator(), before)


@pytest.mark.parametrize("dim", ["unknown", ["time", "unknown"]])
def test_unknown_dimensions(data, dim):
    with pytest.raises(ValueError, match="Unknown"):
        bx.BatchMean(dim).update_batch(data)


@pytest.mark.parametrize("dim", [0, [0], {"time"}])
def test_invalid_dimensions(dim):
    with pytest.raises(TypeError, match="dim"):
        bx.BatchMean(dim)


@pytest.mark.parametrize("dim", [[], ["time", "time"]])
def test_empty_or_duplicate_dimensions(dim):
    with pytest.raises(ValueError, match="dim"):
        bx.BatchMean(dim)


def test_container_validation(data):
    with pytest.raises(TypeError, match="Expected"):
        bx.BatchMean().update_batch(data.values)
    with pytest.raises(ValueError, match="at least one"):
        bx.BatchMean().update_batch(xr.Dataset())
    with pytest.raises(TypeError, match="Cannot mix"):
        bx.BatchMean().update_batch(data).update_batch(data.to_dataset())
    dataset = xr.Dataset({"a": data, "static": ("y", [1, 2, 3])})
    with pytest.raises(ValueError, match="no reduction dimension"):
        bx.BatchMean("time").update_batch(dataset)
    # Reducing all dimensions does support variables with different dimensions.
    xr.testing.assert_allclose(bx.BatchMean().update_batch(dataset)(), dataset.mean())


def test_merge_validation(data):
    a = bx.BatchMean("time").update_batch(data)
    with pytest.raises(TypeError):
        a + bx.BatchSum("time")
    with pytest.raises(ValueError):
        a + bx.BatchMean("x")
    with pytest.raises(ValueError):
        a + bx.BatchMean("time").update_batch(data.to_dataset())
    with pytest.raises(ValueError):
        a + bx.BatchMean("time").update_batch(data.isel(y=slice(None, None, -1)))
    with pytest.raises(ValueError):
        bx.BatchVar("time", ddof=0) + bx.BatchVar("time", ddof=1)
    for merged in (a + bx.BatchMean("time"), bx.BatchMean("time") + a):
        merged.update_batch(data * 2)
        xr.testing.assert_allclose(a(), data.mean("time"))


@pytest.mark.parametrize("cls", [bx.BatchVar, bx.BatchStd])
def test_ddof(data, cls):
    op = "var" if cls is bx.BatchVar else "std"
    a = cls("time", ddof=1).update_batch(data.isel(time=slice(0, 4)))
    b = cls("time", ddof=1).update_batch(data.isel(time=slice(4, None)))
    xr.testing.assert_allclose((a + b)(), getattr(data, op)("time", ddof=1))


@pytest.mark.parametrize("cls", [bx.BatchWeightedSum, bx.BatchWeightedMean])
@pytest.mark.parametrize("dim", ["time", ["time", "x"], None])
@pytest.mark.parametrize("dataset", [False, True])
def test_weighted(data, cls, dim, dataset):
    weights = xr.DataArray(np.arange(1.0, 12), dims="time", coords={"time": data.time})
    if dataset:
        data = xr.Dataset({"a": data, "b": data.isel(x=0, drop=True).drop_vars("latitude")})
    a, b = cls(dim), cls(dim)
    a.update_batch(data.isel(time=slice(0, 4)), weights.isel(time=slice(0, 4)))
    b.update_batch(data.isel(time=slice(4, None)).transpose(..., "time"), weights.isel(time=slice(4, None)))
    expected = (data * weights).sum(dim)
    if cls is bx.BatchWeightedMean:
        if dataset:
            denominator = xr.Dataset(
                {
                    name: weights.broadcast_like(var).sum(
                        [
                            d
                            for d in (var.dims if dim is None else [dim] if isinstance(dim, str) else dim)
                            if d in var.dims
                        ]
                    )
                    for name, var in data.data_vars.items()
                }
            )
        else:
            denominator = weights.broadcast_like(data).sum(dim)
        expected = expected / denominator
    xr.testing.assert_allclose((a + b)(), expected)
    assert a.n_samples is None


def test_weight_dataset_and_scalar(data):
    dataset = xr.Dataset({"a": data, "b": data * 2})
    weights = xr.Dataset({"a": xr.ones_like(data), "b": xr.ones_like(data) * 2})
    xr.testing.assert_allclose(
        bx.BatchWeightedSum("time").update_batch(dataset, weights)(), (dataset * weights).sum("time")
    )
    xr.testing.assert_allclose(bx.BatchWeightedMean("time").update_batch(data, 2)(), data.mean("time"))
    scalar = xr.DataArray([1.0, 2.0], dims="time")
    assert bx.BatchWeightedMean().update_batch(scalar, 2)().item() == 1.5


@pytest.mark.parametrize("bad", ["order", "size", "extra_dim", "unlabelled", "variables", "aux"])
def test_invalid_weights(data, bad):
    weight = xr.ones_like(data)
    if bad == "order":
        weight = weight.isel(time=slice(None, None, -1))
    elif bad == "size":
        weight = weight.isel(time=slice(0, 2))
    elif bad == "extra_dim":
        weight = weight.expand_dims(z=[1])
    elif bad == "unlabelled":
        weight = weight.values
    elif bad == "variables":
        weight = weight.to_dataset(name="other")
    else:
        weight = weight.assign_coords(latitude=weight.latitude + 1)
    with pytest.raises((TypeError, ValueError)):
        bx.BatchWeightedMean("time").update_batch(data, weight)


@pytest.mark.parametrize("cls", [bx.BatchTopK, bx.BatchNanTopK])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dataset", [True, False])
def test_topk(data, cls, largest, dataset):
    if dataset:
        data = data.to_dataset()
    a = cls(4, "time", largest).update_batch(data.isel(time=slice(0, 4)))
    b = cls(4, "time", largest).update_batch(data.isel(time=slice(4, None)))
    merged = a + b
    expected = data.max("time") if largest else data.min("time")
    xr.testing.assert_allclose(merged.rank(1), expected)
    xr.testing.assert_allclose(merged().isel(rank=0, drop=True), expected)
    q = 0.9 if largest else 0.1
    expected_quantile = data.quantile(q, "time").drop_vars("quantile").assign_coords(source=data.source)
    xr.testing.assert_allclose(merged.quantile(q), expected_quantile)
    np.testing.assert_array_equal(merged()["rank"], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="capacity"):
        merged.quantile(0.5)


def test_topk_missing_ranks_and_name_collision():
    data = xr.DataArray([3, 1], dims="time")
    result = bx.BatchTopK(4).update_batch(data)()
    np.testing.assert_allclose(result, [3, 1, np.nan, np.nan])
    assert bx.BatchTopK(1).update_batch(data)().dtype.kind == "i"
    with pytest.raises(ValueError, match="conflicts"):
        bx.BatchTopK(2).update_batch(data.rename(time="rank"))
    result = bx.BatchTopK(2, rank_dim="extreme").update_batch(data.rename(time="rank"))()
    assert result.dims == ("extreme",)
    with pytest.raises(ValueError, match="Rank dimension"):
        bx.BatchTopK(2) + bx.BatchTopK(2, rank_dim="other")
    with pytest.raises(TypeError):
        bx.BatchTopK(2, rank_dim=0)
    nan = bx.BatchNanTopK(2, "time").update_batch(xr.DataArray([np.nan, 3.0], dims="time"))
    assert nan.n_samples.item() == 1
    assert np.isnan(nan.rank(2))


@pytest.mark.parametrize("cls", [bx.BatchCov, bx.BatchCorr])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("dataset", [False, True])
def test_matrix_statistics(data, cls, paired, dataset):
    raw = data.values.reshape(11, 6)
    second = (data.isel(x=0, drop=True) ** 2).rename(y="feature")
    raw2 = second.values
    if dataset:
        data, second = data.to_dataset(), second.to_dataset()
    first = cls("time", ddof=1)
    last = cls("time", ddof=1)
    for accumulator, sl in [(first, slice(0, 4)), (last, slice(4, None))]:
        accumulator.update_batch(data.isel(time=sl), second.isel(time=sl) if paired else None)
    result = (first + last)()
    if dataset:
        result = result["temperature"]
    if cls is bx.BatchCov:
        expected = np.cov(raw.T, raw2.T, ddof=1)[:6, 6:] if paired else np.cov(raw.T, ddof=1)
    else:
        expected = np.corrcoef(raw.T, raw2.T)[:6, 6:] if paired else np.corrcoef(raw.T)
    np.testing.assert_allclose(result.values.reshape(expected.shape), expected, atol=1e-14)
    assert result.dims == (("y", "x", "feature_2") if paired else ("y", "x", "y_2", "x_2"))
    np.testing.assert_array_equal(result.y, data.y)
    assert first.n_samples.item() == 4 if not dataset else first.n_samples["temperature"].item() == 4


def test_matrix_validations(data):
    stat = bx.BatchCov("time").update_batch(data)
    with pytest.raises(ValueError, match="Cannot mix"):
        stat.update_batch(data, data)
    with pytest.raises(ValueError, match="coordinates"):
        bx.BatchCov("time").update_batch(data, data.isel(time=slice(None, None, -1)))
    with pytest.raises(ValueError, match="sizes"):
        bx.BatchCov("time").update_batch(data, data.isel(time=slice(0, 2)))
    with pytest.raises(ValueError, match="collide"):
        bx.BatchCov("time").update_batch(data.rename(x="y_2"))
    with pytest.raises(ValueError, match="modes"):
        stat + bx.BatchCov("time").update_batch(data, data)
    with pytest.raises(ValueError, match="coordinates"):
        bx.BatchCov("time").update_batch(data, data) + bx.BatchCov("time").update_batch(
            data, data.isel(y=slice(None, None, -1))
        )


@pytest.mark.parametrize("cls", [bx.BatchCov, bx.BatchCorr])
def test_matrix_series(cls):
    data = xr.DataArray([1.0, 2.0, 4.0], dims="time")
    value = cls("time").update_batch(data)()
    assert value.dims == ()
    assert value.item() == pytest.approx(data.var().item() if cls is bx.BatchCov else 1.0)


@pytest.mark.parametrize("name", bx.__all__)
def test_uninitialized(name):
    cls = getattr(bx, name)
    accumulator = cls(2) if "TopK" in name else cls()
    assert accumulator.n_samples is None
    with pytest.raises(NoValidSamplesError):
        accumulator()


def test_empty_batch_can_be_followed_by_data(data):
    for cls in [bx.BatchMean, bx.BatchNanMean, bx.BatchTopK]:
        accumulator = cls(2, "time") if cls is bx.BatchTopK else cls("time")
        accumulator.update_batch(data.isel(time=slice(0, 0))).update_batch(data)
        expected = cls(2, "time") if cls is bx.BatchTopK else cls("time")
        xr.testing.assert_identical(accumulator(), expected.update_batch(data)())


def test_dask_backed_batches(data):
    pytest.importorskip("dask.array")
    lazy = data.chunk({"time": 2})
    accumulator = bx.BatchNanMean("time")
    for start in range(0, 11, 3):
        accumulator.update_batch(lazy.isel(time=slice(start, start + 3)))
    xr.testing.assert_allclose(accumulator(), data.mean("time"))
    assert isinstance(accumulator().data, np.ndarray)


def test_multiindex_coordinate_preserved(data):
    stacked = data.stack(cell=("y", "x"))
    result = bx.BatchMean("time", keep_attrs=True).update_batch(stacked)()
    xr.testing.assert_identical(result, stacked.mean("time", keep_attrs=True))


def test_transposed_full_weights(data):
    weights = xr.ones_like(data).transpose("x", "time", "y")
    result = bx.BatchWeightedMean("time").update_batch(data, weights)()
    xr.testing.assert_allclose(result, data.mean("time"))


def test_matrix_dataset_name_collision(data):
    dataset = xr.Dataset({"x_2": data})
    with pytest.raises(ValueError, match="collide"):
        bx.BatchCov("time").update_batch(dataset)


@pytest.mark.parametrize("cls", [bx.BatchCov, bx.BatchCorr])
def test_matrix_multiple_reduced_dimensions(data, cls):
    expected_data = data.transpose("time", "y", "x").values.reshape(-1, 2)
    accumulator = cls(["time", "y"])
    accumulator.update_batch(data.isel(time=slice(0, 3)))
    accumulator.update_batch(data.isel(time=slice(3, None)).transpose("x", "y", "time"))
    expected = np.cov(expected_data.T, ddof=0) if cls is bx.BatchCov else np.corrcoef(expected_data.T)
    np.testing.assert_allclose(accumulator(), expected)


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("replace_file", [False, True])
def test_netcdf_coordinates_detached_from_source(tmp_path, dataset, replace_file):
    pytest.importorskip("netCDF4")
    first = xr.DataArray(
        np.arange(6.0).reshape(3, 2), dims=("time", "x"), name="a", coords={"aux": ("x", [10.0, 20.0])}
    )
    path = tmp_path / "source.nc"
    first.to_netcdf(path, engine="netcdf4")
    with xr.open_dataset(path, engine="netcdf4", cache=False) as batch:
        accumulator = bx.BatchMean("time").update_batch(batch if dataset else batch.a)
    path.unlink()
    if replace_file:
        first.assign_coords(aux=("x", [100.0, 200.0])).to_netcdf(path, engine="netcdf4")
        with xr.open_dataset(path, engine="netcdf4", cache=False) as batch:
            with pytest.raises(ValueError, match="coordinates"):
                accumulator.update_batch(batch if dataset else batch.a)
    expected = first.mean("time")
    if dataset:
        expected = expected.to_dataset()
    xr.testing.assert_identical(accumulator(), expected)
    checkpoint = tmp_path / "checkpoint.npz"
    accumulator.save(checkpoint)
    xr.testing.assert_identical(bx.BatchMean.load(checkpoint)(), expected)


@pytest.mark.parametrize("cls", [bx.BatchCov, bx.BatchCorr])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("dataset", [False, True])
def test_matrix_without_dimension_coordinates(tmp_path, cls, paired, dataset):
    raw = np.random.default_rng(9).normal(size=(7, 3))
    first = xr.DataArray(raw, dims=("time", "x"), name="a")
    second = xr.DataArray(raw[:, :2] ** 2, dims=("time", "feature"), name="a") if paired else None
    if dataset:
        first = first.to_dataset()
        second = second.to_dataset() if paired else None
    accumulator = cls("time").update_batch(first, second)
    path = tmp_path / "matrix.npz"
    accumulator.save(path)
    restored = cls.load(path)
    result = restored()["a"] if dataset else restored()
    op = np.cov if cls is bx.BatchCov else np.corrcoef
    kwargs = {"ddof": 0} if cls is bx.BatchCov else {}
    expected = op(raw.T, (raw[:, :2] ** 2).T, **kwargs)[:3, 3:] if paired else op(raw.T, **kwargs)
    np.testing.assert_allclose(result, expected)
    assert result.dims == ("x", "feature_2" if paired else "x_2")
    restored.update_batch(first, second)
    xr.testing.assert_allclose(restored(), (accumulator + accumulator)())


def test_dataset_coordinates_shared_internally_but_owned(data):
    dataset = xr.Dataset({"a": data, "b": data * 2, "c": data.isel(x=0, drop=True).drop_vars("latitude")})
    accumulator = bx.BatchMean("time").update_batch(dataset)
    expected = accumulator()
    a, b = accumulator._layouts["a"].coords, accumulator._layouts["b"].coords
    assert np.shares_memory(a.latitude.values, b.latitude.values)
    assert not np.shares_memory(a.latitude.values, dataset.latitude.values)
    assert "latitude" not in accumulator._layouts["c"].coords
    dataset.latitude.values[:] = -1
    output = accumulator()
    output.latitude.values[:] = -2
    xr.testing.assert_identical(accumulator(), expected)
    merged = accumulator + accumulator
    assert np.shares_memory(merged._layouts["a"].coords.latitude, merged._layouts["b"].coords.latitude)
    assert not np.shares_memory(merged._layouts["a"].coords.latitude, a.latitude)


def test_subsequent_batches_do_not_snapshot_metadata(data):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("Metadata of later batches must not be copied")

    accumulator = bx.BatchMean("time", keep_attrs=True).update_batch(data)
    expected = accumulator()
    next_batch = data.copy(deep=True)
    next_batch.attrs["new"] = CannotCopy()
    next_batch.latitude.attrs["new"] = CannotCopy()
    accumulator.update_batch(next_batch)
    xr.testing.assert_identical(accumulator(), expected)


def test_coordinate_snapshot_loads_only_retained_coordinates():
    da = pytest.importorskip("dask.array")
    from dask import delayed

    @delayed
    def forbidden():
        raise AssertionError("Reduced coordinates must not be computed")

    values = np.arange(6.0).reshape(3, 2)
    data = xr.DataArray(
        da.from_array(values, chunks=(1, 2)),
        dims=("time", "x"),
        coords={
            "aux": ("x", da.from_array(np.array([10.0, 20.0]))),
            "discarded": ("time", da.from_delayed(forbidden(), shape=(3,), dtype=float)),
        },
    )
    accumulator = bx.BatchMean("time").update_batch(data)
    assert isinstance(data.aux.data, da.Array)  # The input was not loaded in place.
    assert isinstance(accumulator._layouts[None].coords.aux.data, np.ndarray)
    expected = xr.DataArray(values.mean(axis=0), dims="x", coords={"aux": ("x", [10.0, 20.0])}, name=data.name)
    xr.testing.assert_identical(accumulator(), expected)


@pytest.mark.parametrize("name", bx.__all__)
@pytest.mark.parametrize("empty_side", [None, "left", "right"])
def test_merge_kernels_never_alias_operands(data, name, empty_side):
    def make(empty):
        cls = getattr(bx, name)
        accumulator = cls(2, "time") if "TopK" in name else cls("time")
        batch = data.isel(time=slice(0, 0)) if empty else data
        if isinstance(accumulator, bx.BatchWeightedSum):
            return accumulator.update_batch(batch, weights=1)
        return accumulator.update_batch(batch)

    def buffers(kernel):
        for value in vars(kernel).values():
            if isinstance(value, np.ndarray):
                yield value
            elif hasattr(value, "update_batch"):
                yield from buffers(value)

    left, right = make(empty_side == "left"), make(empty_side == "right")
    merged = left + right
    expected = merged()
    for array in buffers(merged._accumulators[None]):
        for source in (*buffers(left._accumulators[None]), *buffers(right._accumulators[None])):
            assert not np.shares_memory(array, source)
    for source in (*buffers(left._accumulators[None]), *buffers(right._accumulators[None])):
        source[...] = 0
    xr.testing.assert_identical(merged(), expected)
