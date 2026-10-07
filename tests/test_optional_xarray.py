"""Run even when xarray is not installed."""

import subprocess
import sys


def test_numpy_api_does_not_import_xarray():
    script = """
import sys
import importlib.abc

class BlockXarray(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xarray" or fullname.startswith("xarray."):
            raise ModuleNotFoundError("xarray deliberately unavailable", name="xarray")

sys.meta_path.insert(0, BlockXarray())
import batchstats
import numpy as np
assert "xarray" not in sys.modules
np.testing.assert_allclose(batchstats.BatchMean().update_batch([[1, 3], [5, 7]])(), [3, 5])
try:
    import batchstats.xarray
except ImportError as exc:
    assert "pip install 'batchstats[xarray]'" in str(exc)
else:
    raise AssertionError("Optional module should explain how to install xarray")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
