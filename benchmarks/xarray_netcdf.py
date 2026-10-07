"""Compare streaming and open_mfdataset means in fresh processes (outside CI).

Run from the repository root, after installing netCDF4, psutil, and dask[array]:
    python benchmarks/xarray_netcdf.py --output build/xarray-benchmark

The default workload is ten uncompressed NetCDF4 files, one float32 variable,
32 x 1024 x 1024 samples/file, static 2D coordinates, and about 1% missing data.
Generation, imports, reference checking, and output serialization are not timed.
RSS is the OS process high-water mark, including imports and native allocations.
Files are warmed before measurements; this does NOT measure cold-storage speed.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import dask
import netCDF4
import numpy as np
import psutil
import xarray as xr
from extremes import peak_rss

from batchstats import xarray as bx
from batchstats.xarray import BatchNanMean

METHODS = ("batchstats-file", "batchstats-slice", "mfdataset-single", "mfdataset-default", "mfdataset-chunked")


def generate(config):
    directory = config.output / "inputs"
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261007)
    shape = (config.time, config.side, config.side)
    total = np.zeros(shape[1:], dtype=np.float64)
    count = np.zeros(shape[1:], dtype=np.int64)
    y = np.linspace(-60, 60, config.side, dtype=np.float32)
    x = np.linspace(-180, 180, config.side, dtype=np.float32)
    latitude = np.broadcast_to(y[:, None], shape[1:]).copy()
    longitude = np.broadcast_to(x[None, :], shape[1:]).copy()
    for index in range(config.files):
        values = rng.random(shape, dtype=np.float32)
        values.reshape(-1)[index::97] = np.nan
        values[:, 0, 0] = np.nan
        total += np.nansum(values, axis=0, dtype=np.float64)
        count += np.count_nonzero(~np.isnan(values), axis=0)
        dataset = xr.Dataset(
            {"temperature": (("time", "y", "x"), values, {"units": "K"})},
            coords={
                "time": np.arange(index * config.time, (index + 1) * config.time),
                "y": y,
                "x": x,
                "latitude": (("y", "x"), latitude),
                "longitude": (("y", "x"), longitude),
            },
        )
        dataset.to_netcdf(
            directory / f"part-{index:02d}.nc",
            engine="netcdf4",
            encoding={"temperature": {"contiguous": True, "zlib": False}},
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        reference = total / count
    np.save(config.output / "reference.npy", reference)


def worker(config):
    paths = sorted((config.output / "inputs").glob("*.nc"))
    # Initialize backend discovery equally, outside the timed region.
    xr.backends.list_engines()
    gc.collect()
    baseline = psutil.Process().memory_info().rss
    start = time.perf_counter()
    if config.worker.startswith("batchstats"):
        accumulator = BatchNanMean("time")
        for path in paths:
            # Disable the xarray cache so discarded time slices can be released.
            with xr.open_dataset(path, engine="netcdf4", chunks=None, cache=False) as dataset:
                if config.worker == "batchstats-file":
                    accumulator.update_batch(dataset)
                else:
                    for offset in range(0, dataset.sizes["time"], config.chunk):
                        accumulator.update_batch(dataset.isel(time=slice(offset, offset + config.chunk)))
        result = accumulator().load()  # Include materialization of output coordinates, too.
        chunk_description = None
    else:
        chunks = {"time": config.chunk} if config.worker == "mfdataset-chunked" else None
        with xr.open_mfdataset(paths, engine="netcdf4", chunks=chunks) as dataset:
            chunk_description = dataset["temperature"].chunks
            kwargs = {"scheduler": "single-threaded"} if config.worker == "mfdataset-single" else {}
            result = dataset.mean("time", skipna=True, keep_attrs=False).compute(**kwargs)
    seconds = time.perf_counter() - start
    rss = peak_rss()  # Read before reference loading or comparison allocations.
    reference = np.load(config.output / "reference.npy")
    actual = result["temperature"].values
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-7, equal_nan=True)
    record = {
        "method": config.worker,
        "seconds": seconds,
        "baseline_rss_bytes": baseline,
        "peak_rss_bytes": rss,
        "peak_minus_baseline_bytes": rss - baseline,
        "max_abs_error": float(np.nanmax(np.abs(actual - reference))),
        "output_dtype": str(actual.dtype),
        "chunks": chunk_description,
    }
    print(json.dumps(record), flush=True)


def run_worker(config, method):
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(config.output),
            "--chunk",
            str(config.chunk),
            "--worker",
            method,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/xarray-benchmark"))
    parser.add_argument("--files", type=int, default=10)
    parser.add_argument("--time", type=int, default=32)
    parser.add_argument("--side", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reuse", action="store_true", help="Reuse this script's existing inputs and reference")
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    config = parser.parse_args()
    if min(config.files, config.time, config.side, config.chunk, config.repeat) < 1:
        parser.error("All sizes and repeat must be positive")
    config.output = config.output.resolve()
    if config.worker:
        worker(config)
        return
    config.output.mkdir(parents=True, exist_ok=True)
    if not config.reuse:
        generate(config)
    paths = sorted((config.output / "inputs").glob("*.nc"))
    with xr.open_dataset(paths[0], engine="netcdf4") as first:
        input_shape = dict(first.sizes)
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("batchstats", "numpy", "xarray", "dask", "pandas", "netCDF4", "psutil")
        },
        "batchstats_source": str(Path(bx.__file__).resolve()),
        "batchstats_xarray_sha256": hashlib.sha256(Path(bx.__file__).read_bytes()).hexdigest(),
        "netcdf_library": netCDF4.__netcdf4libversion__,
        "hdf5_library": netCDF4.__hdf5libversion__,
        "dask_workers": dask.config.get("num_workers", default=None),
        "files": len(paths),
        "shape_per_file": input_shape,
        "disk_bytes": sum(path.stat().st_size for path in paths),
        "chunk_time": config.chunk,
        "repeat": config.repeat,
        "cache": "warm OS cache; one untimed run per method; no cache eviction",
    }
    print(json.dumps(environment), flush=True)
    for method in METHODS:
        run_worker(config, method)
        print(f"Warmup complete: {method}", flush=True)
    records = []
    rng = random.Random(20261007)
    for repeat in range(config.repeat):
        methods = list(METHODS)
        rng.shuffle(methods)
        for method in methods:
            record = {"repeat": repeat + 1, **run_worker(config, method)}
            records.append(record)
            print(json.dumps(record), flush=True)
            (config.output / "results.json").write_text(
                json.dumps({"environment": environment, "runs": records}, indent=2), encoding="utf-8"
            )
    summary = []
    for method in METHODS:
        runs = [record for record in records if record["method"] == method]
        summary.append(
            {
                "method": method,
                "median_seconds": statistics.median(record["seconds"] for record in runs),
                "min_seconds": min(record["seconds"] for record in runs),
                "max_seconds": max(record["seconds"] for record in runs),
                "median_peak_mib": statistics.median(record["peak_rss_bytes"] for record in runs) / 2**20,
                "max_peak_mib": max(record["peak_rss_bytes"] for record in runs) / 2**20,
            }
        )
    (config.output / "results.json").write_text(
        json.dumps({"environment": environment, "runs": records, "summary": summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
