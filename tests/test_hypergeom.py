# MIT License
#
# Copyright (c) 2020 Yuxin Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""This module tests both the JIT'ed version and the original python version of each function."""
import numpy as np
from scipy.stats import hypergeom as reference
from scipy.special import comb
from numpy.ma.testutils import assert_almost_equal
import pytest
import statdp._hypergeom as hypergeom


def test_precision():
    M, n, N = 2500, 50, 500
    for pmf in (hypergeom.pmf, hypergeom.pmf.py_func):
        value = pmf(2, M, n, N)
        assert_almost_equal(value, 0.0010114963068932233, 11)


def test_ln_binomial():
    for ln_binomial in (hypergeom._ln_binomial, hypergeom._ln_binomial.py_func):
        assert_almost_equal(np.log(comb(200, 100)), ln_binomial(200, 100), 11)
        assert_almost_equal(np.log(comb(5, 3)), ln_binomial(5, 3), 11)
        assert_almost_equal(np.log(comb(67, 32)), ln_binomial(67, 32), 11)
        assert ln_binomial(100, 0) == 0
        assert ln_binomial(100, 100) == 0
        with pytest.raises(ValueError):
            ln_binomial(200, 300)


def test_pmf():
    for pmf in (hypergeom.pmf, hypergeom.pmf.py_func):
        assert_almost_equal(pmf(0, 2, 1, 0), 1.0, 11)
        assert_almost_equal(pmf(1, 2, 1, 0), 0.0, 11)
        assert_almost_equal(pmf(0, 2, 0, 2), 1.0, 11)
        assert_almost_equal(pmf(1, 2, 1, 0), 0.0, 11)
        for M in range(1000, 10000, 500):
            for n in range(1000, M, 500):
                for N in range(10, 1000, 50):
                    for k in range(10, N, 30):
                        assert_almost_equal(pmf(k, M, n, N), reference.pmf(k, M, n, N), 9)
        with pytest.raises(ValueError):
            # number of draws is greater than the total number of objects
            pmf(1, 100, 20, 300)


def test_sf():
    for sf in (hypergeom.sf, hypergeom.sf.py_func):
        # we test the values from our hypergeom module with the one from scipy
        for M in range(1000, 10000, 500):
            for n in range(1000, M, 500):
                for N in range(10, 1000, 50):
                    for k in range(10, N, 30):
                        assert_almost_equal(sf(k, M, n, N), reference.sf(k, M, n, N), 9)

        # k >= min(n, N)
        assert sf(20, 100, 5, 10) == 0
        # k < 0
        assert sf(-1, 100, 5, 10) == 1
        with pytest.raises(ValueError):
            # number of draws is greater than the total number of objects
            sf(1, 100, 20, 300)
