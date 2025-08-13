import numpy as np

from .._misc import NoValidSamplesError
from ..base.batch_stat import BatchStat


class BatchWeightedSum(BatchStat):
    """
    Class for calculating the weighted sum of batches of data.

    The algorithm used is a simple cumulative sum. Each time a new batch is added,
    the weighted sum of the new batch is added to the existing sum.

    .. code:: python

        import numpy as np
        from batchstats.stats.weighted_sum import BatchWeightedSum

        # create some data
        data1 = np.array([[1, 2], [3, 4]])
        weights1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        data2 = np.array([[5, 6], [7, 8]])
        weights2 = np.array([[0.5, 0.6], [0.7, 0.8]])

        # create a BatchWeightedSum object
        bws = BatchWeightedSum()

        # update with the first batch
        bws.update_batch(data1, weights=weights1)

        # update with the second batch
        bws.update_batch(data2, weights=weights2)

        # get the weighted sum
        total_weighted_sum = bws()

        # verify the result
        expected_sum = np.array([8.4, 12.0])
        np.testing.assert_allclose(total_weighted_sum, expected_sum)
    """

    def __init__(self, axis=0):
        super().__init__(axis=axis)
        self.sum = None
        self.n_samples = None
        self._weights_pattern = None

        axis_tuple = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        # If batch axis 0 is not summed, we need to collect results in a list
        self._is_list_mode = 0 not in axis_tuple
        if self._is_list_mode:
            self.sum = []

    def update_batch(self, batch, weights):
        """
        Update the weighted sum with a new batch of data.
        """
        weights = np.asarray(weights)

        axis_tuple = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis_tuple = tuple(ax if ax >= 0 else ax + batch.ndim for ax in axis_tuple)

        # Check for consistent weight shapes
        # The pattern ignores the batch axis (0) and any summed axes
        axes_to_ignore = set(axis_tuple)
        axes_to_ignore.add(0)  # always ignore batch axis
        current_pattern = tuple(s for i, s in enumerate(weights.shape) if i not in axes_to_ignore)
        if self._weights_pattern is None:
            self._weights_pattern = current_pattern
        elif self._weights_pattern != current_pattern:
            raise ValueError(
                f"Inconsistent weights shape pattern. "
                f"Expected pattern for non-summed axes: {self._weights_pattern}, "
                f"but got {current_pattern}."
            )

        batch_sum = np.sum(a=batch * weights, axis=self.axis, keepdims=True)

        if self._is_list_mode:
            self.sum.append(batch_sum)
        else:
            if self.sum is None:
                self.sum = batch_sum
            else:
                self.sum += batch_sum
        return self

    def __call__(self) -> np.ndarray:
        """
        Calculate the weighted sum.
        """
        if self.sum is None or (self._is_list_mode and not self.sum):
            raise NoValidSamplesError("No valid samples for calculating weighted sum.")

        if self._is_list_mode:
            sum_ = np.concatenate(self.sum, axis=0)
        else:
            sum_ = self.sum.copy()

        if self.axis is not None:
            return sum_.squeeze(axis=self.axis)
        return sum_

    def __add__(self, other):
        # Basic checks from merge_test
        if type(self) != type(other):
            from .._misc import DifferentStatsError

            raise DifferentStatsError()
        if self.axis != other.axis:
            from .._misc import DifferentAxisError

            raise DifferentAxisError()

        if self._is_list_mode:
            if not self.sum:
                return other
            if not other.sum:
                return self
            ret = BatchWeightedSum(axis=self.axis)
            ret.sum = self.sum + other.sum
            ret._weights_pattern = self._weights_pattern
            return ret
        else:
            self.merge_test(other, field="sum")  # ok to call here
            if self.sum is None:
                return other
            elif other.sum is None:
                return self
            ret = BatchWeightedSum(axis=self.axis)
            ret.sum = self.sum + other.sum
            ret._weights_pattern = self._weights_pattern
            return ret
