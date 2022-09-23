"""
Tests for metrics.py module.
"""
from metrics import is_permutation, nondecreasing


def test_permutation():
    a, b = [1, 2, 3, 4], [1, 2, 3, 4, 5]
    assert not is_permutation(a, b)
    a, b = [1, 2, 3, 4], [1, 2, 3, 4]
    assert is_permutation(a, b)
    a, b = [4, 3, 2, 1