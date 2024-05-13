import numpy as np

from ._misc import NoValidSamplesError


class BatchNanStat:
    """
    Base class for calculating statistics over batches of data that can contain NaN values.

    Attributes:
        n_samples (numpy.ndarray): Total number of samples processed, accounting for NaN values.

    """

    def __init__(self):
        """
        Initialize the BatchNanStat object.
        """
        self.n_samples = None

    def _process_batch(self, batch):
        """
        Process the input batch, counting NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            numpy.ndarray: Processed batch.

        """
        batch = np.atleast_2d(np.asarray(batch))
        axis = tuple(range(1, batch.ndim))
        if self.n_samples is None:
            self.n_samples = np.isfinite(batch).sum(axis=0)
        else:
            self.n_samples += np.isfinite(batch).sum(axis=0)
        return batch


class BatchNanSum(BatchNanStat):
    """
    Class for calculating the sum of batches of data that can contain NaN values.

    """

    def __init__(self):
        """
        Initialize the BatchNanSum object.
        """
        super().__init__()
        self.sum = None

    def update_batch(self, batch):
        """
        Update the sum with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanSum: Updated BatchNanSum object.

        """
        batch = self._process_batch(batch)
        axis = tuple(range(1, batch.ndim))
        if self.sum is None:
            self.sum = np.nansum(batch, axis=0)
        else:
            self.sum += np.nansum(batch, axis=0)
        return self

    def __call__(self):
        """
        Calculate the sum of the batches that can contain NaN values.

        Returns:
            numpy.ndarray: Sum of the batches.

        Raises:
            NoValidSamplesError: If no valid samples are available.

        """
        if self.sum is None:
            raise NoValidSamplesError()
        else:
            return np.where(self.n_samples > 0, self.sum, np.nan)


class BatchNanMean(BatchNanStat):
    """
    Class for calculating the mean of batches of data that can contain NaN values.

    """

    def __init__(self):
        """
        Initialize the BatchNanMean object.
        """
        super().__init__()
        self.sum = BatchNanSum()

    def update_batch(self, batch):
        """
        Update the mean with a new batch of data that can contain NaN values.

        Args:
            batch (numpy.ndarray): Input batch.

        Returns:
            BatchNanMean: Updated BatchNanMean object.

        """
        self.sum.update_batch(batch)
        return self

    def __call__(self):
        """
        Calculate the mean of the batches that can contain NaN values.

        Returns:
            numpy.ndarray: Mean of the batches.

        """
        return self.sum() / self.sum.n_samples
