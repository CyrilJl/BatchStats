import numpy as np

from ._misc import DifferentAxisError, DifferentShapesError, DifferentStatsError, any_nan, check_params


class BatchStat:
    """
    Base class for calculating statistics over batches of data.

    Attributes:
        n_samples (int): Total number of samples processed.

    """

    def __init__(self, axis=0):
        self.axis = check_params(param=axis, types=(int, tuple))
        self.n_samples = 0

    def _complementary_axis(self, ndim):
        if isinstance(self.axis, int):
            return tuple(set(range(ndim)) - set((self.axis,)))
        else:
            return tuple(set(range(ndim)) - set(self.axis))

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
        if assume_valid:
            self.n_samples += len(batch)
            return batch
        else:
            axis = self._complementary_axis(ndim=batch.ndim)
            nan_mask = any_nan(batch, axis=axis)
            if nan_mask.any():
                valid_batch = batch[~nan_mask]
            else:
                valid_batch = batch
            self.n_samples += len(valid_batch)
            return valid_batch

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


class BatchNanStat:
    """
    Base class for calculating statistics over batches of data that can contain NaN values.

    Attributes:
        n_samples (numpy.ndarray): Total number of samples processed, accounting for NaN values.

    """

    def __init__(self, axis=0):
        """
        Initialize the BatchNanStat object.
        """
        self.n_samples = None
        self.axis = axis

    def _process_batch(self, batch):
        """
        Process the input batch, counting NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            numpy.ndarray: Processed batch.

        """
        batch = np.atleast_2d(np.asarray(batch))
        if self.n_samples is None:
            self.n_samples = np.isfinite(batch).sum(axis=self.axis)
        else:
            self.n_samples += np.isfinite(batch).sum(axis=self.axis)
        return batch
