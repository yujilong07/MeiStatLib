import numpy as np
from .core import stMean


def stDisp(x, ddof=0, axis=None):
    mean = stMean(x, axis=axis)
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
    
    centered = x - mean
    squared = centered ** 2
    varian = stMean(squared, axis=axis)
    
    n = x.shape[axis] if axis is not None else x.size
    
    return varian * n / (n - ddof)


def stStd(x, ddof=0, axis=None):
    return np.sqrt(stDisp(x, ddof, axis=axis))


def stMed(x, axis=None):
    if axis is None:
        sorted_x = sorted(x.flat)
        n = len(sorted_x)
        if n % 2 == 1:
            return sorted_x[n//2]
        else:
            return (sorted_x[n//2 - 1] + sorted_x[n//2]) / 2
    return np.median(x, axis=axis)


def stMode(x):
    freq_dict = {}
    for val in x.flat:
        if val in freq_dict:
            freq_dict[val] += 1
        else:
            freq_dict[val] = 1
    
    max_count = 0      
    mode_val = None
    for val, count in freq_dict.items():
        if count > max_count:
            max_count = count
            mode_val = val
    return mode_val


def stQuantile(x, q):
    sorted_x = sorted(x.flat)
    n = len(sorted_x)
    pos = q * (n - 1)
    i = int(pos)
    f = pos - i
    
    return sorted_x[i] + f * (sorted_x[i+1] - sorted_x[i])