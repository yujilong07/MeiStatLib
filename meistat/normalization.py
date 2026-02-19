import numpy as np
from .core import stMean, stMin, stMax, stSum
from .descriptive import stStd

def stZscore(x,axis=None):
	mean = stMean(x, axis=axis)
    std = stStd(x, ddof=0, axis=axis)
    
    if axis is not None:
        mean = np.expand_dims(mean, axis=axis)
        std = np.expand_dims(std, axis=axis)
    
    return (x - mean) / std