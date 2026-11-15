import pytest
from scripts.vector_variance import VectorVariance


def test_constant_fractional_vectors_have_exact_zero_variance():
    s=VectorVariance()
    for _ in range(128):s.add({'root':[.9061472612503986,.09385273874960148]})
    assert s.trace()==0.


def test_large_constant_coordinate_cannot_hide_small_coordinate_variance():
    s=VectorVariance()
    for x in (1.,2.,3.):s.add({'root':[1e18,x]})
    assert s.trace()==pytest.approx(1.)


def test_sparse_missing_coordinates_are_zero_observations():
    s=VectorVariance()
    for row in ({'a':[1.]},{},{'a':[3.]},{}):s.add(row)
    assert s.trace()==pytest.approx(2.)


def test_width_or_nonfinite_errors_do_not_count_invalid_vector():
    s=VectorVariance();s.add({'a':[1.]})
    for row in ({'a':[1.,2.]},{'a':[float('inf')]}):
        with pytest.raises(ValueError):s.add(row)
    assert s.count==1
    with pytest.raises(ValueError,match='two'):s.trace()
