from .core import stMin, stMax, stSum, stMean

# Descriptive statistics
from .descriptive import (
    stDisp, stStd, stMed, stMode, stQuantile,
    stRange, stIQR, stPercentile, stQuartiles,
    stCV, stSEM, stMAD, stSkew, stKurt
)

# Correlation
from .correlation import (
    stCov, stCovPirs, stCorMatr,
    stCovMatrix, stSpearman, stKendall
)

# Normalization and transformations
from .normalization import (
    stZScore, stCumSum, stCumProd, stProd,
    stGeomMean, stHarmMean, stTrimMean
)

__all__ = [
    # Core
    'stMin', 'stMax', 'stSum', 'stMean',
    
    # Descriptive
    'stDisp', 'stStd', 'stMed', 'stMode', 'stQuantile',
    'stRange', 'stIQR', 'stPercentile', 'stQuartiles',
    'stCV', 'stSEM', 'stMAD', 'stSkew', 'stKurt',
    
    # Correlation
    'stCov', 'stCovPirs', 'stCorMatr',
    'stCovMatrix', 'stSpearman', 'stKendall',
    
    # Normalization
    'stZScore', 'stCumSum', 'stCumProd', 'stProd',
    'stGeomMean', 'stHarmMean', 'stTrimMean',
]

__version__ = '2.0.0'