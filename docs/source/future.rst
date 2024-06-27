.. Future Development

What's New ?
============

Version 0.4.3 (June 27, 2024)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Overload of the ``__add__`` operator of the ``BatchStats*`` classes to merge the statistics of two objects computed on two different datasets, even imbalanced.

Version 0.4.1 (June 9, 2024)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Improved numerical stability of the ``BatchMean``'s algorithm, based on `Numerically Stable Parallel Computation of (Co-)Variance by Schubert and Gertz <https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf>`_


Future Plans
============

Here are some potential areas for future development of the ``BatchStats`` package:

- Adding additional statistics (``BatchOver``, ``BatchHist`` ?)
- Implementing ``xarray`` compatibility
- Improving memory consumption in ``BatchCov``
- Implementing shape checks
- Adding a ``dtype`` argument for better control of the underlying arrays

If you have any suggestions or would like to contribute to the development of BatchStats, please feel free to reach out to the maintainers or submit a pull request on the GitHub repository.
