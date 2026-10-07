"""Checkpoint round trips, continuation, metadata and malformed state rejection."""

import datetime as dt
import json
from copy import deepcopy

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")

from batchstats import NoValidSamplesError  # noqa: E402
from batchstats import xarray as bx  # noqa: E402


@pytest.fixture
def data():
    return xr.DataArray(
        np.random.default_rng(42).normal(size=(9, 3)),
        dims=("time", "station"),
        coords={
            "time": np.arange(9),
            "station": ["A", "B", "C"],
            "latitude": ("station", [40.0, 50.0, 60.0]),
            "source": "sensor",
        },
        name="temperature",
        attrs={"units": "K", "history": ["first"]},
    )


def create(name, dim="time"):
    cls = getattr(bx, name)
    options = {"dim": dim, "keep_attrs": True}
    if "TopK" in name:
        options.update(k=4, largest=False, rank_dim="extreme")
    if name in ("BatchVar", "BatchStd", "BatchCov", "BatchCorr"):
        options["ddof"] = 1
    return cls(**options)


def update(accumulator, data):
    if isinstance(accumulator, bx.BatchWeightedSum):
        return accumulator.update_batch(data, weights=data.time + 1)
    if isinstance(accumulator, bx.BatchCov):
        return accumulator.update_batch(data, data**2)
    return accumulator.update_batch(data)


def assert_counts(a, b):
    if a.n_samples is None:
        assert b.n_samples is None
    else:
        xr.testing.assert_identical(a.n_samples, b.n_samples)


@pytest.mark.parametrize("name", bx.__all__)
@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("disk", [False, True])
def test_resume_every_statistic(tmp_path, data, name, dataset, disk):
    if dataset:
        data = xr.Dataset({"first": data, "second": data * 2}, attrs={"title": "weather"})
    accumulator = update(create(name), data.isel(time=slice(0, 4)))
    before = accumulator()
    if disk:
        path = tmp_path / "checkpoint.npz"
        accumulator.save(path)
        restored = type(accumulator).load(path)
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                assert not archive[key].dtype.hasobject
    else:
        state = accumulator.to_state()
        # Metadata is valid strict JSON, independent of large numeric arrays.
        state["metadata"] = json.loads(json.dumps(state["metadata"], allow_nan=False))
        restored = type(accumulator).from_state(state)
    xr.testing.assert_identical(restored(), before)
    assert_counts(accumulator, restored)
    tail = data.isel(time=slice(4, None))
    update(restored, tail)
    xr.testing.assert_identical(accumulator(), before)
    update(accumulator, tail)
    xr.testing.assert_identical(restored(), accumulator())
    assert_counts(accumulator, restored)
    merged = restored + update(create(name), data)
    xr.testing.assert_allclose(merged(), (accumulator + update(create(name), data))())
    if "TopK" in name:
        assert restored.rank_dim == "extreme"
        xr.testing.assert_identical(restored.quantile(0.1), accumulator.quantile(0.1))


@pytest.mark.parametrize("name", bx.__all__)
@pytest.mark.parametrize("empty_batch", [False, True])
def test_uninitialized_and_empty_checkpoint(tmp_path, data, name, empty_batch):
    accumulator = create(name)
    if empty_batch:
        update(accumulator, data.isel(time=slice(0, 0)))
    path = tmp_path / "checkpoint"  # Exact filename, no suffix appended.
    accumulator.save(str(path))
    assert path.is_file()
    restored = type(accumulator).load(path)
    if not empty_batch:
        with pytest.raises(NoValidSamplesError):
            restored()
    update(restored, data)
    xr.testing.assert_identical(restored(), update(create(name), data)())


@pytest.mark.parametrize("name", [n for n in bx.__all__ if "Nan" in n])
def test_all_nan_then_resume(data, name):
    accumulator = create(name).update_batch(xr.full_like(data, np.nan))
    restored = type(accumulator).from_state(accumulator.to_state())
    assert_counts(accumulator, restored)
    accumulator.update_batch(data)
    restored.update_batch(data)
    xr.testing.assert_identical(restored(), accumulator())


@pytest.mark.parametrize("name", ["BatchMean", "BatchNanMean", "BatchNanTopK", "BatchCov", "BatchCorr"])
@pytest.mark.parametrize("dim", [None, ["time", "station"]])
def test_scalar_output_roundtrip(data, name, dim):
    accumulator = update(create(name, dim), data)
    restored = type(accumulator).from_state(accumulator.to_state())
    xr.testing.assert_identical(restored(), accumulator())
    update(restored, data)
    update(accumulator, data)
    xr.testing.assert_identical(restored(), accumulator())


@pytest.mark.parametrize("cls", [bx.BatchCov, bx.BatchCorr])
def test_unpaired_matrix_checkpoint(data, cls):
    original = cls("time").update_batch(data)
    restored = cls.from_state(original.to_state())
    xr.testing.assert_identical(restored(), original())
    restored.update_batch(data)
    original.update_batch(data)
    xr.testing.assert_identical(restored(), original())
    with pytest.raises(ValueError, match="mix"):
        restored.update_batch(data, data)


def test_metadata_types_and_ownership(tmp_path, data):
    attributes = {
        "units": "°C",
        "tuple": (1, "two"),
        "list": [None, True, b"bytes"],
        "nested": {3: {"value": np.float32(4)}},
        "array": np.array([1, 2], dtype="int16"),
        "object_array": np.array(["A", 2, None, (1, 2)], dtype=object),
        "nan": float("nan"),
        "infinity": float("inf"),
        "complex": 1 + 2j,
        "datetime64": np.datetime64("2026-01-01", "D"),
        "timedelta64": np.timedelta64(2, "h"),
        "date": dt.date(2026, 1, 2),
        "datetime": dt.datetime(2026, 1, 2, 10, tzinfo=dt.timezone.utc),
        "timedelta": dt.timedelta(days=2, microseconds=3),
        "timestamp": pd.Timestamp("2026-01-01"),
        "pd_timedelta": pd.Timedelta("2 days"),
        "nat": pd.NaT,
        "na": pd.NA,
        "dtype": np.dtype("float64"),
    }
    data.attrs = attributes
    data.latitude.attrs = {"units": "degrees_north"}
    data.latitude.encoding = {"dtype": np.dtype("float32"), "_FillValue": -999.0}
    accumulator = bx.BatchNanMean("time", keep_attrs=True).update_batch(data)
    path = tmp_path / "metadata.npz"
    accumulator.save(path)
    restored = bx.BatchNanMean.load(path)
    # Compare encoded states too: pandas.NA cannot participate in dict equality.
    state1, state2 = accumulator.to_state(), restored.to_state()
    assert state1["metadata"] == state2["metadata"]
    for key in state1["arrays"]:
        np.testing.assert_array_equal(state1["arrays"][key], state2["arrays"][key])
    attrs = restored().attrs
    assert isinstance(attrs["tuple"], tuple)
    assert attrs["array"].dtype == np.dtype("int16")
    assert isinstance(attrs["nested"][3]["value"], np.float32)
    assert restored().latitude.attrs == data.latitude.attrs
    assert restored().latitude.encoding == data.latitude.encoding
    attrs["array"][0] = 100
    assert restored().attrs["array"][0] == 1


@pytest.mark.parametrize("kind", ["object_strings", "datetime", "timedelta", "multiindex", "no_index", "aux_index"])
def test_coordinate_roundtrip(data, kind):
    if kind == "object_strings":
        data = data.assign_coords(station=np.array(["été", "hiver", "a"], dtype=object))
    elif kind == "datetime":
        data = data.assign_coords(station=np.array(["2025-01-01", "NaT", "2025-03-01"], dtype="datetime64[ns]"))
    elif kind == "timedelta":
        data = data.assign_coords(station=np.array([1, 2, 3], dtype="timedelta64[ns]"))
    elif kind == "multiindex":
        data = data.expand_dims(level=[100, 200]).stack(cell=("station", "level"))
        data.cell.attrs = {"description": "stacked"}
    elif kind == "no_index":
        data = data.drop_indexes("station")
    else:
        data = data.set_xindex("latitude")
    accumulator = bx.BatchMean("time", keep_attrs=True).update_batch(data)
    restored = bx.BatchMean.from_state(accumulator.to_state())
    xr.testing.assert_identical(restored(), accumulator())
    assert set(restored().xindexes) == set(accumulator().xindexes)
    restored.update_batch(data)
    accumulator.update_batch(data)
    xr.testing.assert_identical(restored(), accumulator())


def test_state_arrays_do_not_alias(data):
    accumulator = bx.BatchNanMean("time").update_batch(data)
    state = accumulator.to_state()
    restored = bx.BatchNanMean.from_state(state)
    before = restored()
    for array in state["arrays"].values():
        if array.dtype.kind in "iuf":
            array[...] = 0
    xr.testing.assert_identical(restored(), before)
    xr.testing.assert_identical(accumulator(), before)


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "type",
        "format",
        "array_shape",
        "array_dtype",
        "object_dtype",
        "negative_count",
        "kernel_field",
        "kernel_type",
        "axis",
        "shape",
        "duplicate",
        "extra_array",
        "kind",
        "paired",
        "tag",
        "dimensions",
        "missing_count",
    ],
)
def test_malformed_state_rejected(data, change):
    state = bx.BatchMean("time").update_batch(data).to_state()
    meta = state["metadata"]
    variable = meta["variables"][0]
    kernel = variable["kernel"]
    if change in ("version", "type", "format", "kind"):
        meta[change] = "unknown"
    elif change.startswith("array_") or change == "object_dtype":
        key = kernel["fields"]["mean"]["key"]
        array = state["arrays"][key]
        if change == "array_shape":
            state["arrays"][key] = array.ravel()
        else:
            state["arrays"][key] = array.astype(object if change == "object_dtype" else "float32")
    elif change == "negative_count":
        kernel["fields"]["n_samples"] = -1
    elif change == "missing_count":
        kernel["fields"]["n_samples"] = None
    elif change == "kernel_field":
        kernel["fields"]["surprise"] = 1
    elif change == "kernel_type":
        kernel["type"] = "BatchSum"
    elif change == "axis":
        kernel["fields"]["axis"] = 1
    elif change == "shape":
        variable["layout"]["shape"] = [100]
    elif change == "duplicate":
        meta["variables"].append(deepcopy(variable))
    elif change == "extra_array":
        state["arrays"]["extra"] = np.zeros(1)
    elif change == "paired":
        meta["paired"] = True
    elif change == "tag":
        meta["attrs"]["tag"] = "execute_python"
    else:
        variable["layout"]["remaining"]["items"].append("station")
    with pytest.raises(ValueError):
        bx.BatchMean.from_state(state)


def test_wrong_class_and_broken_file(tmp_path, data):
    path = tmp_path / "checkpoint.npz"
    bx.BatchMean("time").update_batch(data).save(path)
    with pytest.raises(ValueError, match="statistic type"):
        bx.BatchSum.load(path)
    path.write_bytes(b"broken checkpoint")
    with pytest.raises(ValueError):
        bx.BatchMean.load(path)
    np.savez(path, metadata=np.array({"malicious": "object"}, dtype=object))
    with pytest.raises(ValueError):
        bx.BatchMean.load(path)
    with pytest.raises(FileNotFoundError):
        bx.BatchMean.load(tmp_path / "missing")


def test_atomic_save_failure(tmp_path, data, monkeypatch):
    from batchstats import _xarray_state

    path = tmp_path / "checkpoint.npz"
    original = bx.BatchMean("time").update_batch(data)
    original.save(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(_xarray_state.np, "savez", fail)
    with pytest.raises(OSError, match="simulated"):
        original.save(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_unsupported_metadata_does_not_overwrite(tmp_path, data):
    path = tmp_path / "checkpoint.npz"
    bx.BatchMean("time").update_batch(data).save(path)
    before = path.read_bytes()
    data.attrs["custom"] = object()
    bad = bx.BatchMean("time", keep_attrs=True).update_batch(data)
    with pytest.raises(TypeError, match="Unsupported checkpoint metadata"):
        bad.save(path)
    assert path.read_bytes() == before


def test_keep_attrs_policy_after_load(data):
    for keep in (False, True):
        accumulator = bx.BatchMean("time", keep_attrs=keep).update_batch(data)
        restored = bx.BatchMean.from_state(accumulator.to_state())
        changed = data.copy(deep=True)
        changed.attrs = {"units": "changed"}
        restored.update_batch(changed)
        assert restored().attrs == (data.attrs if keep else {})
        with pytest.raises(ValueError, match="coordinates"):
            restored.update_batch(data.isel(station=slice(None, None, -1)))


def test_multiindex_missing_codes_and_unused_levels(data):
    index = pd.MultiIndex(
        levels=[["A", "B", "unused"], [1, 2, 3]], codes=[[0, 1, -1], [0, 1, 2]], names=["label", "level"]
    )
    if hasattr(xr, "Coordinates"):
        data = data.drop_vars("station").assign_coords(xr.Coordinates.from_pandas_multiindex(index, "station"))
    else:
        data = data.drop_vars("station").assign_coords(station=index)
    accumulator = bx.BatchMean("time").update_batch(data)
    restored = bx.BatchMean.from_state(accumulator.to_state())
    xr.testing.assert_identical(restored(), accumulator())
    assert restored().indexes["station"].equal_levels(index)


def test_multiindex_level_dtypes_preserved(data):
    stacked = data.expand_dims(level=np.array([1, 2], dtype="int16")).stack(cell=("station", "level"))
    original = bx.BatchMean("time").update_batch(stacked)
    restored = bx.BatchMean.from_state(original.to_state())
    assert restored().station.dtype == original().station.dtype
    assert restored().level.dtype == original().level.dtype


@pytest.mark.parametrize(
    "index", [pd.CategoricalIndex(["A", "B", "C"]), pd.date_range("2020-01-01", periods=3, tz="Europe/Paris")]
)
def test_unsupported_index_fails_explicitly(data, index):
    data = data.assign_coords(station=index)
    with pytest.raises(TypeError, match="Checkpoint indexes"):
        bx.BatchMean("time").update_batch(data).to_state()


def test_shared_arrays_saved_once_and_restored_independently(tmp_path, data):
    dataset = xr.Dataset({"a": data, "b": data * 2})
    accumulator = bx.BatchNanMean("time").update_batch(dataset)
    state = accumulator.to_state()
    variables = state["metadata"]["variables"]
    for variable in variables:
        fields = variable["kernel"]["fields"]
        assert fields["n_samples"]["key"] == fields["sum"]["fields"]["n_samples"]["key"]
    latitude_keys = [
        next(c["data"]["key"] for c in v["layout"]["coords"]["variables"] if c["name"] == "latitude") for v in variables
    ]
    assert latitude_keys[0] == latitude_keys[1]
    path = tmp_path / "shared.npz"
    accumulator.save(path)
    for restored in (bx.BatchNanMean.from_state(state), bx.BatchNanMean.load(path)):
        for kernel in restored._accumulators.values():
            assert kernel.n_samples is kernel.sum.n_samples
        coords = [layout.coords.latitude.values for layout in restored._layouts.values()]
        assert np.shares_memory(*coords)
        assert not np.shares_memory(coords[0], dataset.latitude.values)
        expected = accumulator()
        for array in state["arrays"].values():
            if array.dtype.kind in "iuf":
                array[...] = 0
        xr.testing.assert_identical(restored(), expected)


def test_old_v1_duplicate_count_arrays_still_load(data):
    original = bx.BatchNanMean("time").update_batch(data)
    state = original.to_state()
    fields = state["metadata"]["variables"][0]["kernel"]["fields"]
    old_key = fields["n_samples"]["key"]
    duplicate_key = "old_duplicate_count"
    state["arrays"][duplicate_key] = state["arrays"][old_key].copy()
    fields["n_samples"]["key"] = duplicate_key
    restored = bx.BatchNanMean.from_state(state)
    kernel = restored._accumulators[None]
    assert kernel.n_samples is kernel.sum.n_samples
    xr.testing.assert_identical(restored(), original())
    restored.update_batch(data)
    xr.testing.assert_identical(restored(), original.update_batch(data)())


def test_load_owns_npz_arrays_without_an_extra_copy(tmp_path, data, monkeypatch):
    original = bx.BatchMean("time").update_batch(data)
    path = tmp_path / "checkpoint.npz"
    original.save(path)
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    getitem = archive_type.__getitem__
    read_arrays = {}

    def record(archive, key):
        array = getitem(archive, key)
        read_arrays[key] = array
        return array

    monkeypatch.setattr(archive_type, "__getitem__", record)
    loaded = bx.BatchMean.load(path)
    metadata = json.loads(str(read_arrays["metadata"]))
    key = metadata["variables"][0]["kernel"]["fields"]["mean"]["key"]
    assert loaded._accumulators[None].mean is read_arrays[key]
    output = loaded()
    output.values[:] = 0
    xr.testing.assert_identical(loaded(), original())


def test_shared_array_descriptors_are_each_validated(data):
    state = bx.BatchNanMean("time").update_batch(data).to_state()
    fields = state["metadata"]["variables"][0]["kernel"]["fields"]
    fields["sum"]["fields"]["n_samples"]["shape"] = [999]
    with pytest.raises(ValueError, match="shape"):
        bx.BatchNanMean.from_state(state)
