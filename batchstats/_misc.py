import warnings

import numpy as np

# Customize the warning format
warnings.formatwarning = lambda msg, *args, **kwargs: str(msg) + '\n'


class NoValidSamplesError(ValueError):
    """
    Error raised when there are no valid samples for calculation.
    """
    pass


class UnequalSamplesNumber(ValueError):
    """
    Error raised when two batches have unequal lengths.
    """
    pass


class DifferentAxisError(ValueError):
    """
    Error raised when two BatchStats objects are merged but have different `axis`.
    """
    pass


class DifferentShapesError(ValueError):
    """
    Error raised when two BatchStats objects are merged but have different shapes.
    """
    pass


class DifferentStatsError(ValueError):
    """
    Error raised when two BatchStats objects are merged but hare not of the same type.
    """
    pass


def any_nan(x, axis=None):
    """
    Check if there are any NaN values in the input array.

    Args:
        x (numpy.ndarray): Input array.
        axis (int or tuple of ints, optional): Axis or axes along which to operate. Default is None.

    Returns:
        numpy.ndarray: Boolean array indicating NaN presence.

    """
    return np.isnan(np.add.reduce(array=x, axis=axis))


def check_params(param, params=None, types=None):
    # Check if the parameter's type matches the accepted types
    if (types is not None) and (not isinstance(param, types)):
        if isinstance(types, type):
            accepted = f'{types}'
        else:
            accepted = f"{', '.join([str(t) for t in types])}"
        # Raise a TypeError with a customized message
        msg = f"`{param}` is not of an accepted type, it can only be of type {accepted}!"
        raise TypeError(msg)

    # Check if the parameter is among the recognized parameters
    if (params is not None) and (param not in params):
        # Raise a ValueError with a customized message
        msg = f"`{param}` is not a recognized argument, it can only be one of {', '.join(sorted(params))}!"
        raise ValueError(msg)

    # Return the parameter if it passes the checks
    return param


def warning(msg):
    # Trigger a warning with the provided message
    return warnings.warn(msg)
