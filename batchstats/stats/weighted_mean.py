import numpy as np

from .._misc import NoValidSamplesError
from ..base.batch_stat import BatchStat
from .weighted_sum import BatchWeightedSum


class BatchWeightedMean(BatchStat):
    """
    Class for calculating the weighted mean of batches of data.
    It computes `sum(w*x) / sum(w)` by using two `BatchWeightedSum` instances.
    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        # For calculating sum(w*x)
        self.weighted_sum = BatchWeightedSum(axis=axis)
        # For calculating sum(w)
        self.sum_of_weights = BatchWeightedSum(axis=axis)
        self.n_samples = None  # n_samples is not used in weighted stats

    def update_batch(self, batch, weights):
        """
        Update the weighted mean with a new batch of data.
        """
        # Update sum(w*x)
        self.weighted_sum.update_batch(batch, weights=weights)

        # Update sum(w) by calculating sum(broadcasted_w * 1)
        broadcasted_weights = np.broadcast_to(weights, batch.shape)
        self.sum_of_weights.update_batch(batch=broadcasted_weights, weights=1)

        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the weighted mean.
        """
        try:
            # Calculate sum(w*x) and sum(w)
            w_sum = self.weighted_sum()
            sow = self.sum_of_weights()

            # To prevent division by zero, where sow is zero, we should return NaN or raise an error.
            # Using np.errstate to handle division by zero gracefully.
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_ = w_sum / sow
                if hasattr(mean_, "dtype") and np.issubdtype(mean_.dtype, np.floating):
                    mean_[sow == 0] = np.nan  # Set to NaN where sum of weights is zero

        except NoValidSamplesError:
            # Re-raise with a more specific message if no samples were processed
            raise NoValidSamplesError("No valid samples for calculating weighted mean.")

        return mean_

    def __add__(self, other):
        """
        Merge two BatchWeightedMean objects.
        """
        # Basic checks
        if type(self) != type(other):
            from .._misc import DifferentStatsError

            raise DifferentStatsError()
        if self.axis != other.axis:
            from .._misc import DifferentAxisError

            raise DifferentAxisError()

        # Create a new object and merge the internal calculators
        ret = BatchWeightedMean(axis=self.axis)
        ret.weighted_sum = self.weighted_sum + other.weighted_sum
        ret.sum_of_weights = self.sum_of_weights + other.sum_of_weights

        return ret
