"""Measure coordinate ownership and checkpoint allocations, outside CI.

Run from the repository root: python benchmarks/xarray_allocations.py
Traced peaks count Python/NumPy allocations, not OS resident memory.
"""

import gc
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import xarray as xr

from batchstats.xarray import BatchNanMean


def peak_allocation(operation):
    gc.collect()
    tracemalloc.start()
    try:
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / 2**20


def main():
    side = 1024
    array = xr.DataArray(
        np.zeros((1, side, side), dtype=np.float32),
        dims=("time", "y", "x"),
        coords={
            "lat": (("y", "x"), np.ones((side, side))),
            "lon": (("y", "x"), np.ones((side, side))),
        },
    )
    dataset = xr.Dataset({name: array for name in ("a", "b", "c")})
    accumulator = BatchNanMean("time").update_batch(dataset)
    unique_coordinates = []
    for layout in accumulator._layouts.values():
        for coord in layout.coords.coords.values():
            if not any(np.shares_memory(coord.values, previous) for previous in unique_coordinates):
                unique_coordinates.append(coord.values)
    accumulator._prepare(dataset)
    times = []
    for _ in range(20):
        start = time.perf_counter()
        accumulator._prepare(dataset)
        times.append(time.perf_counter() - start)
    _, validation_peak = peak_allocation(lambda: accumulator._prepare(dataset))
    # Initialize the checkpoint module before allocation measurement.
    accumulator.to_state()
    state, export_peak = peak_allocation(accumulator.to_state)
    checkpoint_mib = sum(value.nbytes for value in state["arrays"].values()) / 2**20
    del state
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.npz"
        accumulator.save(path)
        restored, load_peak = peak_allocation(lambda: BatchNanMean.load(path))
        xr.testing.assert_identical(restored(), accumulator())
        del restored
    merged, merge_peak = peak_allocation(lambda: accumulator + accumulator)
    xr.testing.assert_identical(merged(), accumulator())
    print(
        json.dumps(
            {
                "variables": 3,
                "shape": [1, side, side],
                "unique_coordinate_mib": sum(value.nbytes for value in unique_coordinates) / 2**20,
                "validation_median_ms": statistics.median(times) * 1000,
                "validation_peak_traced_mib": validation_peak,
                "checkpoint_array_mib": checkpoint_mib,
                "export_peak_traced_mib": export_peak,
                "load_peak_traced_mib": load_peak,
                "merge_peak_traced_mib": merge_peak,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
