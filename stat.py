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

def stSum (x, axis=None):
    if x.size == 0:
        raise ValueError("Empty array")
    if axis is None:
        total = 0.0
        for val in x.flat:
            total += val
        return total
    return np.sum(x, axis=axis)


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

def stStd(x,ddof=0):
    return np.sqrt(stDisp(x,ddof))

def stMed(x, axis=None):
    if axis is None:
        sorted_x = sorted(x.flat)
        n = len(sorted_x)
        if n % 2 == 1:
            return sorted_x[n//2]
        else:
            return (sorted_x[n//2 - 1]) + sorted_x[n//2] / 2

    return np.median(x, axis=axis)

def stQuantile(x, q):
    sorted_x = sorted(x.flat)
    n = len(sorted_x)
    pos = q * (n - 1)
    i = int(pos // 2)
    f = pos - i
    return sorted_x[i] + f * (sorted_x[i+1] - sorted_x[i])
print(stQuantile(np.array([1,3,5,7]), 0.5))  # 4.0
