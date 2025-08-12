import numpy as np


class BatchNanStat:
    """
    Base class for calculating statistics over batches of data that can contain NaN values.

    This is a base class and is not meant to be used directly. Instead, use one of
    the derived classes like `BatchNanSum`, `BatchNanMean`, etc.

    Attributes:
        n_samples (numpy.ndarray): Total number of samples processed, accounting for NaN values.

    .. code:: python

        import numpy as np
        from batchstats import BatchNanSum # inheriting from BatchNanStat

        # create some data
        data1 = np.array([[1, 2], [3, np.nan]])
        data2 = np.array([[5, 6], [np.nan, 8]])

        # create a BatchNanSum object
        bns = BatchNanSum()

        # update with the first batch
        bns.update_batch(data1)
        print(f"Number of samples after first batch: {bns.n_samples}")

        # update with the second batch
        bns.update_batch(data2)
        print(f"Number of samples after second batch: {bns.n_samples}")

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
