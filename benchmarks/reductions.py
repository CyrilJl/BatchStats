"""Repeated NaN sum/mean benchmarks in isolated processes, outside CI."""

import argparse
import json
import multiprocessing as mp
import time
import tracemalloc

import numpy as np
from extremes import peak_rss

from batchstats import BatchNanMean, BatchNanSum


def worker(connection, config, statistic, dtype, missing):
    try:
        rng = np.random.default_rng(3)
        data = rng.normal(size=(config.samples, config.positions)).astype(dtype)
        if missing:
            data[::7, ::3] = np.nan
        if config.axis == 1:
            data = data.T.copy()
        factory = {"sum": BatchNanSum, "mean": BatchNanMean}[statistic]

        def calculate():
            state = factory(axis=config.axis)
            for offset in range(0, config.samples, config.batch):
                block = (
                    data[offset : offset + config.batch]
                    if config.axis == 0
                    else data[:, offset : offset + config.batch]
                )
                state.update_batch(block)
            return state

        calculate()  # warm up separately from timing and allocation tracking
        times = []
        for _ in range(config.repeat):
            start = time.perf_counter()
            state = calculate()
            times.append(time.perf_counter() - start)
        tracemalloc.start()
        calculate()
        _, tracked_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss = peak_rss()
        reference = np.nansum(data, axis=config.axis)
        counts = np.count_nonzero(~np.isnan(data), axis=config.axis)
        if statistic == "mean":
            reference = reference / counts
        np.testing.assert_allclose(state(), reference, rtol=2e-4, atol=2e-4)
        np.testing.assert_array_equal(state.n_samples, counts)
        connection.send(
            {
                "statistic": statistic,
                "dtype": dtype,
                "missing": missing,
                "seconds": float(np.median(times)),
                "tracked_peak_bytes": tracked_peak,
                "peak_rss_bytes": rss,
            }
        )
    except Exception as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("samples", 4096), ("positions", 1024), ("batch", 256), ("repeat", 5)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--axis", type=int, choices=(0, 1), default=0)
    config = parser.parse_args()
    if min(config.samples, config.positions, config.batch, config.repeat) < 1:
        parser.error("sizes and repeat must be positive")
    context = mp.get_context("spawn")
    for dtype in ["float32", "float64"]:
        for missing in [False, True]:
            for statistic in ["sum", "mean"]:
                reader, writer = context.Pipe(duplex=False)
                process = context.Process(target=worker, args=(writer, config, statistic, dtype, missing))
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
