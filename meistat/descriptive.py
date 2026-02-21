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
    if i >= n - 1:
        return sorted_x[-1]
    
    return sorted_x[i] + f * (sorted_x[i+1] - sorted_x[i])


def stRange(x, axis=None):
    if axis is None:
        max_e, min_e = x.flat[0], x.flat[0]  
        for val in x.flat:                    
            if val > max_e: max_e = val       
            if val < min_e: min_e = val
        return max_e - min_e
    else:
        return np.max(x, axis=axis) - np.min(x, axis=axis)


def stIQR(x, axis=None):
    if axis is None:
        q1 = stQuantile(x, 0.25)
        q3 = stQuantile(x, 0.75)
        return q3 - q1
    else:
        return np.percentile(x, 75, axis=axis) - np.percentile(x, 25, axis=axis)


def stCV(x, axis=None):
    mean = stMean(x, axis=axis)
    std = stStd(x, ddof=1, axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.abs(std / mean) * 100
    
    return cv


def stSEM(x, ddof=0, axis=None):
    tetta = stStd(x, ddof, axis=axis)  
    n = x.shape[axis] if axis is not None else x.size  
    return tetta / np.sqrt(n)


def stPercentile(x, p, axis=None):
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    
    if axis is None:
        return stQuantile(x, p / 100)
    else:
        return np.percentile(x, p, axis=axis)


def stQuartiles(x, axis=None):
    if axis is None:
        q1 = stQuantile(x, 0.25)
        q2 = stQuantile(x, 0.50)
        q3 = stQuantile(x, 0.75)
        return q1, q2, q3
    else:
        q1 = np.percentile(x, 25, axis=axis)
        q2 = np.percentile(x, 50, axis=axis)
        q3 = np.percentile(x, 75, axis=axis)
        return q1, q2, q3


def stMAD(x, axis=None):
    median = stMed(x, axis=axis)
    
    if axis is not None:
        median = np.expand_dims(median, axis=axis)
    
    abs_dev = np.abs(x - median)
    return stMed(abs_dev, axis=axis)


def stSkew(x, axis=None):
    mean = stMean(x, axis=axis)
    
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
    
    centered = x - mean
    std = stStd(x, ddof=0, axis=axis)
    
    if axis is not None:
        std = np.expand_dims(std, axis=axis)
    
    m3 = np.mean((centered / std) ** 3, axis=axis)
    
    return m3


def stKurt(x, axis=None):
    mean = stMean(x, axis=axis)
    
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
    
    centered = x - mean
    std = stStd(x, ddof=0, axis=axis)
    
    if axis is not None:
        std = np.expand_dims(std, axis=axis)
    
    m4 = np.mean((centered / std) ** 4, axis=axis)
    
    return m4 - 3