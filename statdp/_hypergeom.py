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
"""This module implements the sf (survival function) of hypergeometric distribution. Note that sf(x) = 1 - cdf(x)
where cdf is the cumulative density function, but implementation-wise it gives a better precision than (1 - cdf).
We try to mimic the interfaces of the counterparts in scipy (scipy.stats.hypergeom.sf and scipy.stats.hypergeom.pmf).
However, our implementation is much more efficient thanks to the JIT compiler numba, without loss of too
much precision. (difference with scipy is < 10^-9, see tests/test_hypergeom.py)

References:
https://en.wikipedia.org/wiki/Hypergeometric_distribution
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html
https://github.com/distributions-io/hypergeometric-cdf/blob/bf59133188731fe0631f1f03a8ac641ad35470bf/lib/number.js
https://stackoverflow.com/a/47725299/5148356
Wu, Trong. "An accurate computation of the hypergeometric distribution function."
ACM Transactions on Mathematical Software (TOMS) 19.1 (1993): 33-43.
"""

import math
import logging
import numba

logger = logging.getLogger(__name__)


@numba.njit(numba.float64(numba.int_, numba.int_))
def _ln_binomial(n, k):
    """log of binomial coefficient function (n k), i.e., n choose k"""
    if k > n:
        raise ValueError
    if k == n or k == 0:
        return 0
    if k * 2 > n:
        k = n - k
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@numba.njit(numba.float64(numba.int_, numba.int_, numba.int_, numba.int_))
def pmf(k, M, n, N):
    """returns the pmf of hypergeometric distribution for given parameters. This interface mimics scipy's hypergeom.pmf
    :param k: input value
    :param M: the total number of objects
    :param n: the total number of Type 1 objects
    :param N: the number of draws
    :return: the probability mass function for given parameter
    """
    if N > M:
        raise ValueError
    if k > n or k > N:
        return 0
    elif N > M - n and k + M - n < N:
        return 0
    return math.exp(_ln_binomial(n, k) + _ln_binomial(M - n, N - k) - _ln_binomial(M, N))


@numba.njit(numba.float64(numba.int_, numba.int_, numba.int_, numba.int_))
def sf(k, M, n, N):
    """returns the survival function of hypergeometric distribution for given parameters. This equals (1 - cdf) but we
    try to be more precise than (1 - cdf). This interface mimics scipy.stats.hypergeom.sf.
    :param k: input value
    :param M: the total number of objects
    :param n: the total number of Type 1 objects
    :param N: the number of draws
    :return: the cumulative density for given parameter
    """
    if N > M:
        raise ValueError('The number of draws (N) is larger than the total number of objects (M)')
    if k >= min(n, N):
        return 0
    elif k < 0:
        return 1
    # calculating the pmf is expensive, use the following recursive definition for performance:
    # P(X=i) = (i / (n - i + 1)) * ((M - n + i - N) / (N - i + 1)) * P(X=i+1)
    # P(X=i) = ((n - i) / (i + 1)) * ((N - i) / (M - n + i + 1 - N)) * P(X=i-1)

    # the hypergeometric distribution is centered around N * (n / M),
    # i.e. pmf has the largest value when k = N * (n / M)
    # therefore, for fewer iterations, we use forward recursive definition to calculate P(X > k) for k > N * (n / M)
    # otherwise we use backward recursive definition to calculate P(X <= k) and return 1 - P(x <= k)
    # this also gives use more precise result when pmf(k) ~= 0 since the error will be significant and propagated
    # through the recursion
    if k > N * n / M:
        pmf_i = pmf(k + 1, M, n, N)
        result = pmf_i
        for i in range(k + 1, N):
            pmf_i *= ((n - i) / (i + 1)) * ((N - i) / (M - n + i + 1 - N))
            result += pmf_i
        return result
    else:
        pmf_i = pmf(k, M, n, N)
        result = pmf_i
        for i in range(k, 0, -1):
            pmf_i *= (i / (n - i + 1)) * ((M - n + i - N) / (N - i + 1))
            result += pmf_i
        return 1 - result
