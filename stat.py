import numpy as np

def stMin(x, axis=None):
    if x.size == 0:
        raise ValueError("Empty array")
    
    if axis is None:
        min_val = x.flat[0]
        for val in x.flat[1:]:  
            if val < min_val:
                min_val = val
        return min_val
    return np.min(x, axis=axis)  

def stMax(x, axis=None):
    if x.size == 0:
        raise ValueError("Empty array")

    if axis is None:
        max_val = x.flat[0]
        for val in x.flat[1:] : 
            if val > max_val:
                max_val = val
        return max_val
    return np.max(x, axis=axis)

def stMean(x, axis=None):
    if axis is None:
        return np.sum(x) / x.size
    
    n = x.shape[axis] 
    return np.sum(x, axis=axis) / n


def stDisp (x, ddof = 0, axis = 0):
    mean = stMean(x, axis)
    centered = x - mean
    squared = centered ** 2
    varian = stMean(squared,axis) 
    n = x.shape[axis] if axis else x.size
    return varian * n  / (n - ddof)

def stCov(x, y, ddof=0):
    mean_x = stMean(x)
    mean_y = stMean(y)
    centered_x = x - mean_x
    centered_y = y - mean_y
    product = centered_x * centered_y
    n = x.size
    return stMean(product) * n / (n - ddof)

def stCovPirs(x, y):
    var_x = stDisp(x)
    var_y = stDisp(y)
    cov = stCov(x,y)
    return cov / (np.sqrt(var_x) * np.sqrt(var_y))
