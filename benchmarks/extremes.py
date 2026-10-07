"""Isolated-process time and memory benchmark; deliberately outside CI."""

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import tempfile
import time
import tracemalloc

import numpy as np

from batchstats import BatchNanTopK


def peak_rss():
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak_working
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def worker(connection, config, operation, dtype, missing):
    try:
        rng = np.random.default_rng(3)
        data = rng.normal(size=(config.samples, config.positions)).astype(dtype)
        if missing:
            data[::7, ::3] = np.nan
        factory = lambda: BatchNanTopK(config.k)  # noqa: E731
        if operation in ("merge", "save", "rank"):
            state = factory().update_batch(data[: config.samples // 2])
            other = factory().update_batch(data[config.samples // 2 :])

        def calculate():
            if operation == "incremental":
                current = factory()
                for offset in range(0, config.samples, config.batch):
                    current.update_batch(data[offset : offset + config.batch])
                return current.values, current.n_samples.nbytes
            if operation == "sort":
                work = np.where(np.isnan(data), -np.inf, data)
                return np.sort(work, axis=0)[-config.k :], 0
            if operation == "partition":
                work = np.where(np.isnan(data), -np.inf, data)
                work.partition(config.samples - config.k, axis=0)
                return np.sort(work[-config.k :], axis=0), 0
            if operation == "merge":
                current = state + other
                return current.values, current.n_samples.nbytes
            if operation == "rank":
                return state.rank(1), 0
            with tempfile.TemporaryDirectory() as directory:
                state.save(os.path.join(directory, "state.npz"))
            return state.values, state.n_samples.nbytes

        calculate()  # warm up separately from timing and allocation tracking
        times = []
        for _ in range(config.repeat):
            start = time.perf_counter()
            calculate()
            times.append(time.perf_counter() - start)
        tracemalloc.start()
        result, count_bytes = calculate()
        _, tracked_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        connection.send(
            {
                "operation": operation,
                "dtype": dtype,
                "missing": missing,
                "seconds": float(np.median(times)),
                "peak_rss_bytes": peak_rss(),
                "tracked_peak_bytes": tracked_peak,
                "state_bytes": (
                    state.values.nbytes + state.n_samples.nbytes if operation == "rank" else result.nbytes + count_bytes
                ),
            }
        )
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("samples", 8760), ("positions", 256), ("batch", 168), ("k", 19), ("repeat", 3)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("incremental", "sort", "partition", "merge", "save", "rank"),
        default=["incremental", "sort", "partition", "merge", "save", "rank"],
    )
    config = parser.parse_args()
    if min(config.samples, config.positions, config.batch, config.k, config.repeat) < 1 or config.k > config.samples:
        parser.error("positive sizes and k <= samples are required")
    context = mp.get_context("spawn")
    for dtype in ["float32", "float64"]:
        for missing in [False, True]:
            for operation in config.operations:
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=worker, args=(writer, config, operation, dtype, missing))
                process.start()
                writer.close()
                result = reader.recv()
                process.join()
                reader.close()
                if "error" in result or process.exitcode:
                    raise RuntimeError(result)
                print(json.dumps({**vars(config), **result}), flush=True)


if __name__ == "__main__":
    main()
