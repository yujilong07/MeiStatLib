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


def stSum(x, axis=None):
    if x.size == 0:
        raise ValueError("Empty array")
    
    if axis is None:
        total = 0.0
        for val in x.flat:
            total += val
        return total
    return np.sum(x, axis=axis)


def stMean(x, axis=None):
    if axis is None:
        return np.sum(x) / x.size
    
    n = x.shape[axis] 
    return np.sum(x, axis=axis) / n