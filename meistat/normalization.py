import numpy as np
from .core import stMean, stMin, stMax, stSum
from .descriptive import stStd


def stZScore(x, axis=None):
    mean = stMean(x, axis=axis)
    std = stStd(x, ddof=0, axis=axis)
    
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
        std = np.expand_dims(std, axis=axis)
    
    return (x - mean) / std


def stCumSum(x, axis=None):
    return np.cumsum(x, axis=axis)


def stCumProd(x, axis=None):
    return np.cumprod(x, axis=axis)


def stProd(x, axis=None):
    return np.prod(x, axis=axis)


def stGeomMean(x, axis=None):
    if np.any(x <= 0):
        raise ValueError("All values must be positive for geometric mean")
    
    n = x.shape[axis] if axis is not None else x.size
    prod = stProd(x, axis=axis)
    
    return prod ** (1 / n)


def stHarmMean(x, axis=None):
    if np.any(x == 0):
        raise ValueError("Values cannot be zero for harmonic mean")
    
    n = x.shape[axis] if axis is not None else x.size
    reciprocal_sum = stSum(1 / x, axis=axis)
    
    return n / reciprocal_sum


def stTrimMean(x, proportiontocut, axis=None):
    if not 0 <= proportiontocut < 0.5:
        raise ValueError("proportiontocut must be between 0 and 0.5")
    
    if axis is None:
        sorted_x = np.sort(x.flatten())
        n = len(sorted_x)
        cut = int(n * proportiontocut)
        
        if cut == 0:
            return stMean(x)
        
        trimmed = sorted_x[cut:-cut]
        return stMean(trimmed)
    else:
        return np.apply_along_axis(
            lambda a: stTrimMean(a, proportiontocut, axis=None), 
            axis, 
            x
        )