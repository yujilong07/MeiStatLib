import numpy as np
from .core import stMean
from .descriptive import stDisp


def stCov(x, y, ddof=0):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    
    mean_x = stMean(x)
    mean_y = stMean(y)
    centered_x = x - mean_x
    centered_y = y - mean_y
    product = centered_x * centered_y
    n = x.size
    
    return np.sum(product) / (n - ddof)


def stCovPirs(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    var_x = stDisp(x, ddof=0)
    var_y = stDisp(y, ddof=0)
    cov = stCov(x, y, ddof=0)

    denominator = np.sqrt(var_x) * np.sqrt(var_y)
    if denominator == 0:
        return np.nan
    
    return cov / denominator


def stCorMatr(x):
    p = x.shape[1]
    result = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            result[i, j] = stCovPirs(x[:, i], x[:, j])
    return result