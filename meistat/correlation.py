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

def stCovMatrix(x, ddof=0):
    if x.ndim != 2:
        raise ValueError("Input must be 2-dimensional")
    
    p = x.shape[1]
    result = np.zeros((p, p))
    
    for i in range(p):
        for j in range(p):
            result[i, j] = stCov(x[:, i], x[:, j], ddof=ddof)
    
    return result

def stCovMatrix(x, ddof=0):
    if x.ndim != 2:
        raise ValueError("Input must be 2-dimensional")
    
    p = x.shape[1]
    result = np.zeros((p, p))
    
    for i in range(p):
        for j in range(p):
            result[i, j] = stCov(x[:, i], x[:, j], ddof=ddof)
    
    return result


def stSpearman(x, y):

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    
    def rank(arr):
        sorted_indices = np.argsort(arr)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(arr))
        return ranks + 1 
    
    rank_x = rank(x)
    rank_y = rank(y)
    
    return stCovPirs(rank_x, rank_y)


def stKendall(x, y):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            sign_x = np.sign(x[j] - x[i])
            sign_y = np.sign(y[j] - y[i])
            
            if sign_x * sign_y > 0:
                concordant += 1
            elif sign_x * sign_y < 0:
                discordant += 1
    
    tau = (concordant - discordant) / (n * (n - 1) / 2)
    
    return tau