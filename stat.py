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
        for val in x.flat[1:]:
            if val > max_val:
                max_val = val
        return max_val
    return np.max(x, axis=axis)


def stMean(x, axis=None):
    if axis is None:
        return np.sum(x) / x.size
    
    n = x.shape[axis] 
    return np.sum(x, axis=axis) / n


def stSum(x, axis=None):
    if x.size == 0:
        raise ValueError("Empty array")
    if axis is None:
        total = 0.0
        for val in x.flat:
            total += val
        return total
    return np.sum(x, axis=axis)


def stDisp(x, ddof=0, axis=None):
    mean = stMean(x, axis=axis)
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
    
    centered = x - mean
    squared = centered ** 2
    varian = stMean(squared, axis=axis) 
    
    n = x.shape[axis] if axis is not None else x.size
    
    return varian * n / (n - ddof)


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


def stQuantile(x, q):
    sorted_x = sorted(x.flat)
    n = len(sorted_x)
    pos = q * (n - 1)
    i = int(pos)
    f = pos - i
    return sorted_x[i] + f * (sorted_x[i+1] - sorted_x[i])


def stMode(x):
    dir = {}
    for val in x.flat:
        if val in dir:
            dir[val] += 1
        else:
            dir[val] = 1
    max_count = 0      
    mode_val = None
    for val, count in dir.items():
        if count > max_count:
            max_count = count
            mode_val = val
    return mode_val


def stCorMatr(x):
    p = x.shape[1]
    result = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            result[i, j] = stCovPirs(x[:, i], x[:, j])
    return result
