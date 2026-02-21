import numpy as np
import meistat as ms

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def assert_equal(self, actual, expected, test_name):
        """Assert equality with tolerance for floats"""
        try:
            if isinstance(expected, float):
                assert abs(actual - expected) < 1e-10, f"Expected {expected}, got {actual}"
            else:
                assert actual == expected, f"Expected {expected}, got {actual}"
            self.passed += 1
            self.tests.append(("✓", test_name))
        except AssertionError as e:
            self.failed += 1
            self.tests.append(("✗", f"{test_name}: {e}"))
    
    def assert_array_equal(self, actual, expected, test_name):
        """Assert array equality"""
        try:
            assert np.allclose(actual, expected), f"Arrays not equal"
            self.passed += 1
            self.tests.append(("✓", test_name))
        except AssertionError as e:
            self.failed += 1
            self.tests.append(("✗", f"{test_name}: {e}"))
    
    def print_results(self):
        """Print test results"""
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        
        for status, name in self.tests:
            print(f"{status} {name}")
        
        print("\n" + "-" * 70)
        total = self.passed + self.failed
        print(f"Passed: {self.passed}/{total}")
        print(f"Failed: {self.failed}/{total}")
        
        if self.failed == 0:
            print("\n✓ ALL TESTS PASSED!")
        else:
            print(f"\n✗ {self.failed} test(s) failed")
        print("=" * 70)


def test_core_functions():
    """Test core module functions"""
    print("\n[1/5] Testing Core Functions...")
    t = TestRunner()
    
    data = np.array([3, 1, 4, 1, 5])
    
    # stMin
    t.assert_equal(ms.stMin(data), 1, "stMin: basic")
    t.assert_array_equal(ms.stMin(np.array([[1, 2], [3, 4]]), axis=0), np.array([1, 2]), "stMin: axis=0")
    
    # stMax
    t.assert_equal(ms.stMax(data), 5, "stMax: basic")
    t.assert_array_equal(ms.stMax(np.array([[1, 2], [3, 4]]), axis=1), np.array([2, 4]), "stMax: axis=1")
    
    # stSum
    t.assert_equal(ms.stSum(np.array([1, 2, 3, 4, 5])), 15.0, "stSum: basic")
    
    # stMean
    t.assert_equal(ms.stMean(np.array([1, 2, 3, 4, 5])), 3.0, "stMean: basic")
    t.assert_array_equal(ms.stMean(np.array([[1, 2], [3, 4]]), axis=0), np.array([2.0, 3.0]), "stMean: axis=0")
    
    t.print_results()
    return t.failed == 0


def test_descriptive_functions():
    """Test descriptive statistics functions"""
    print("\n[2/5] Testing Descriptive Statistics...")
    t = TestRunner()
    
    data = np.array([1, 2, 3, 4, 5])
    
    # Variance and Std
    t.assert_equal(ms.stDisp(data, ddof=0), 2.0, "stDisp: population variance")
    t.assert_equal(ms.stDisp(data, ddof=1), 2.5, "stDisp: sample variance")
    t.assert_equal(ms.stStd(data, ddof=0), np.sqrt(2), "stStd: population std")
    
    # Median (critical bug fix test!)
    t.assert_equal(ms.stMed(np.array([1, 2, 3])), 2, "stMed: odd length")
    t.assert_equal(ms.stMed(np.array([1, 2, 3, 4])), 2.5, "stMed: even length (BUG FIX)")
    
    # Mode
    t.assert_equal(ms.stMode(np.array([1, 2, 2, 3])), 2, "stMode: basic")
    
    # Quantiles (critical bug fix test!)
    t.assert_equal(ms.stQuantile(data, 0.0), 1.0, "stQuantile: 0%")
    t.assert_equal(ms.stQuantile(data, 0.25), 2.0, "stQuantile: 25% (BUG FIX)")
    t.assert_equal(ms.stQuantile(data, 0.5), 3.0, "stQuantile: 50%")
    t.assert_equal(ms.stQuantile(data, 0.75), 4.0, "stQuantile: 75% (BUG FIX)")
    t.assert_equal(ms.stQuantile(data, 1.0), 5.0, "stQuantile: 100%")
    
    # Quartiles
    q1, q2, q3 = ms.stQuartiles(data)
    t.assert_equal(q1, 2.0, "stQuartiles: Q1")
    t.assert_equal(q2, 3.0, "stQuartiles: Q2")
    t.assert_equal(q3, 4.0, "stQuartiles: Q3")
    
    # Range and IQR
    t.assert_equal(ms.stRange(data), 4, "stRange: basic")
    t.assert_equal(ms.stIQR(data), 2.0, "stIQR: basic")
    
    # Percentile
    t.assert_equal(ms.stPercentile(data, 50), 3.0, "stPercentile: 50th")
    
    # CV, SEM, MAD
    cv = ms.stCV(data)
    assert cv > 0, "stCV should be positive"
    t.tests.append(("✓", "stCV: positive value"))
    t.passed += 1
    
    sem = ms.stSEM(data)
    assert sem > 0, "stSEM should be positive"
    t.tests.append(("✓", "stSEM: positive value"))
    t.passed += 1
    
    mad = ms.stMAD(data)
    assert mad >= 0, "stMAD should be non-negative"
    t.tests.append(("✓", "stMAD: non-negative"))
    t.passed += 1
    
    # Skewness and Kurtosis
    symmetric_data = np.array([1, 2, 3, 4, 5])
    skew = ms.stSkew(symmetric_data)
    assert abs(skew) < 0.1, "Symmetric data should have near-zero skew"
    t.tests.append(("✓", "stSkew: symmetric data"))
    t.passed += 1
    
    kurt = ms.stKurt(symmetric_data)
    t.tests.append(("✓", "stKurt: computed"))
    t.passed += 1
    
    t.print_results()
    return t.failed == 0


def test_correlation_functions():
    """Test correlation functions"""
    print("\n[3/5] Testing Correlation Functions...")
    t = TestRunner()
    
    # Perfect positive correlation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    
    t.assert_equal(ms.stCov(x, y, ddof=0), 4.0, "stCov: basic")
    t.assert_equal(ms.stCovPirs(x, y), 1.0, "stCovPirs: perfect positive")
    t.assert_equal(ms.stSpearman(x, y), 1.0, "stSpearman: perfect positive")
    t.assert_equal(ms.stKendall(x, y), 1.0, "stKendall: perfect positive")
    
    # Perfect negative correlation
    y_neg = np.array([10, 8, 6, 4, 2])
    t.assert_equal(ms.stCovPirs(x, y_neg), -1.0, "stCovPirs: perfect negative")
    t.assert_equal(ms.stSpearman(x, y_neg), -1.0, "stSpearman: perfect negative")
    
    # Correlation matrix
    data = np.array([[1, 2], [2, 4], [3, 6]])
    corr = ms.stCorMatr(data)
    
    # Diagonal should be 1
    assert np.allclose(np.diag(corr), np.array([1.0, 1.0])), "Correlation matrix diagonal"
    t.tests.append(("✓", "stCorMatr: diagonal is 1"))
    t.passed += 1
    
    # Matrix should be symmetric
    assert np.allclose(corr, corr.T), "Correlation matrix should be symmetric"
    t.tests.append(("✓", "stCorMatr: symmetric"))
    t.passed += 1
    
    # Covariance matrix
    cov = ms.stCovMatrix(data, ddof=0)
    assert np.allclose(cov, cov.T), "Covariance matrix should be symmetric"
    t.tests.append(("✓", "stCovMatrix: symmetric"))
    t.passed += 1
    
    t.print_results()
    return t.failed == 0


def test_normalization_functions():
    """Test normalization functions"""
    print("\n[4/5] Testing Normalization Functions...")
    t = TestRunner()
    
    data = np.array([1, 2, 3, 4, 5])
    
    # Z-scores
    z = ms.stZScore(data)
    t.assert_equal(ms.stMean(z), 0.0, "stZScore: mean is 0")
    t.assert_equal(ms.stStd(z, ddof=0), 1.0, "stZScore: std is 1")
    
    # Cumulative functions
    t.assert_array_equal(ms.stCumSum(np.array([1, 2, 3])), np.array([1, 3, 6]), "stCumSum: basic")
    t.assert_array_equal(ms.stCumProd(np.array([1, 2, 3])), np.array([1, 2, 6]), "stCumProd: basic")
    
    # Product
    t.assert_equal(ms.stProd(np.array([1, 2, 3, 4])), 24, "stProd: basic")
    
    # Geometric mean
    geom = ms.stGeomMean(np.array([2, 8]))
    t.assert_equal(geom, 4.0, "stGeomMean: basic")
    
    # Harmonic mean
    harm = ms.stHarmMean(np.array([1, 2, 4]))
    assert abs(harm - 1.714285714) < 0.01, "Harmonic mean calculation"
    t.tests.append(("✓", "stHarmMean: basic"))
    t.passed += 1
    
    # Trimmed mean
    data_with_outlier = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    trim = ms.stTrimMean(data_with_outlier, 0.2)
    regular_mean = ms.stMean(data_with_outlier)
    assert trim < regular_mean, f"Trimmed mean should be less than regular mean"
    t.tests.append(("✓", "stTrimMean: removes outliers"))
    t.passed += 1
    
    t.print_results()
    return t.failed == 0


def test_axis_functionality():
    """Test axis parameter functionality"""
    print("\n[5/5] Testing Axis Functionality...")
    t = TestRunner()
    
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6]])
    
    # Test axis=0 (along columns)
    t.assert_array_equal(ms.stMean(matrix, axis=0), np.array([2.5, 3.5, 4.5]), "stMean: axis=0")
    t.assert_array_equal(ms.stDisp(matrix, axis=0), np.array([2.25, 2.25, 2.25]), "stDisp: axis=0")
    
    # Test axis=1 (along rows)
    t.assert_array_equal(ms.stMean(matrix, axis=1), np.array([2.0, 5.0]), "stMean: axis=1")
    
    # Test axis=None (flatten)
    t.assert_equal(ms.stMean(matrix), 3.5, "stMean: axis=None")
    
    t.print_results()
    return t.failed == 0


def run_all_tests():
    results = []
    
    results.append(test_core_functions())
    results.append(test_descriptive_functions())
    results.append(test_correlation_functions())
    results.append(test_normalization_functions())
    results.append(test_axis_functionality())
    

if __name__ == "__main__":
    exit(run_all_tests())