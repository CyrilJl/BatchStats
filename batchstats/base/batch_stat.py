import numpy as np

from .._misc import (
    DifferentAxisError,
    DifferentShapesError,
    DifferentStatsError,
    any_nan,
    check_params,
)


class BatchStat:
    """
    Base class for calculating statistics over batches of data.

    This is a base class and is not meant to be used directly. Instead, use one of
    the derived classes like `BatchSum`, `BatchMean`, etc.

    Attributes:
        n_samples (int): Total number of samples processed.

    .. code:: python

        import numpy as np
        from batchstats import BatchSum # inheriting from BatchStat

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[5, 6], [7, 8]])

        # create a BatchSum object
        bs = BatchSum()

        # update with the first batch
        bs.update_batch(data1)
        print(f"Number of samples after first batch: {bs.n_samples}")

        # update with the second batch
        bs.update_batch(data2)
        print(f"Number of samples after second batch: {bs.n_samples}")

    """

    def __init__(self, axis=0):
        self.axis = check_params(param=axis, types=(int, tuple))
        self.n_samples = 0

    def _complementary_axis(self, ndim):
        if isinstance(self.axis, int):
            axis = {self.axis % ndim}
        else:
            axis = {ax % ndim for ax in self.axis}
        return tuple(set(range(ndim)) - axis)

    def _get_n_samples_in_batch(self, batch):
        if self.axis is None:
            return batch.size

        axis = self.axis
        if isinstance(axis, int):
            axis = (axis,)

        shape = batch.shape
        n = 1
        for ax in axis:
            try:
                n *= shape[ax]
            except IndexError:
                pass
        return n

    def _process_batch(self, batch, assume_valid=False):
        """
        Process the input batch, handling NaN values if necessary.

        Args:
            batch (numpy.ndarray): Input batch.
            assume_valid (bool, optional): If True, assumes all elements in the batch are valid. Default is False.

        Returns:
            numpy.ndarray: Processed batch.

        """
        batch = np.atleast_2d(np.asarray(batch))
        # NaN can only occur in inexact (floating/complex) dtypes: skip the scan otherwise
        if assume_valid or not np.issubdtype(batch.dtype, np.inexact):
            self.n_samples += self._get_n_samples_in_batch(batch)
            return batch
        else:
            axis = self._complementary_axis(ndim=batch.ndim)
            nan_mask = any_nan(batch, axis=axis)
            if nan_mask.any():
                valid_batch = self._filter_valid(batch, nan_mask)
            else:
                valid_batch = batch
            self.n_samples += self._get_n_samples_in_batch(valid_batch)
            return valid_batch

    def _filter_valid(self, batch, nan_mask):
        """Drop invalid samples along the reduction axis."""
        axis = self.axis
        if isinstance(axis, tuple) and len(axis) == 1:
            axis = axis[0]
        if isinstance(axis, int):
            return np.compress(~nan_mask, batch, axis=axis % batch.ndim)
        raise NotImplementedError(
            "NaN filtering is only supported for a single reduction axis. "
            "For multi-axis reductions over data containing NaNs, use the BatchNan* "
            "classes or pass assume_valid=True after filtering the data yourself."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def merge_test(self, other, field: str):
        if type(self) != type(other):
            raise DifferentStatsError()
        if self.axis != other.axis:
            raise DifferentAxisError()
        if hasattr(self, field) and hasattr(other, field):
            if getattr(self, field).shape != getattr(other, field).shape:
                raise DifferentShapesError()
