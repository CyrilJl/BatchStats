import numpy as np

from ._misc import NoValidSamplesError


class BatchNanStat:
    def __init__(self):
        self.n_samples = None

    def _process_batch(self, batch):
        batch = np.atleast_2d(np.asarray(batch))
        axis = tuple(range(1, batch.ndim))
        if self.n_samples is None:
            self.n_samples = np.isfinite(batch).sum(axis=0)
        else:
            self.n_samples += np.isfinite(batch).sum(axis=0)
        return batch


class BatchNanSum(BatchNanStat):
    def __init__(self):
        super().__init__()
        self.sum = None

    def update_batch(self, batch):
        batch = self._process_batch(batch)
        axis = tuple(range(1, batch.ndim))
        if self.sum is None:
            self.sum = np.nansum(batch, axis=0)
        else:
            self.sum += np.nansum(batch, axis=0)
        return self

    def __call__(self):
        if self.sum is None:
            raise NoValidSamplesError()
        else:
            return np.where(self.n_samples > 0, self.sum, np.nan)


class BatchNanMean(BatchNanStat):
    def __init__(self):
        super().__init__()
        self.sum = BatchNanSum()

    def update_batch(self, batch):
        self.sum.update_batch(batch)
        return self

    def __call__(self):
        return self.sum()/self.sum.n_samples
