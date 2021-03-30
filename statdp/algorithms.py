# MIT License
#
# Copyright (c) 2018 Yuxin Wang
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
from itertools import zip_longest

import numpy as np


def _hamming_distance(result1, result2):
    # implement hamming distance in pure python, faster than np.count_zeros if inputs are plain python list
    return sum(res1 != res2 for res1, res2 in zip_longest(result1, result2))


def noisy_max_v1a(prng, queries, epsilon):
    # find the largest noisy element and return its index
    return (np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).argmax()


def noisy_max_v1b(prng, queries, epsilon):
    # INCORRECT: returning maximum value instead of the index
    return (np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).max()


def noisy_max_v2a(prng, queries, epsilon):
    return (np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).argmax()


def noisy_max_v2b(prng, queries, epsilon):
    # INCORRECT: returning the maximum value instead of the index
    return (np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).max()


def histogram_eps(prng, queries, epsilon):
    # INCORRECT: using (epsilon) noise instead of (1 / epsilon)
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=epsilon, size=len(queries))
    return noisy_array[0]


def histogram(prng, queries, epsilon):
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    return noisy_array[0]


def SVT(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=4.0 * N / epsilon)
        if query + eta2 >= noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False)


def iSVT1(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    for query in queries:
        # INCORRECT: no noise added to the queries
        eta2 = 0
        if (query + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)

    true_count = int(len(queries) / 2)
    return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


def iSVT2(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    for query in queries:
        # INCORRECT: noise added to queries doesn't scale with N
        eta2 = prng.laplace(scale=2.0 / epsilon)
        if (query + eta2) >= noisy_T:
            out.append(True)
            # INCORRECT: no bounds on the True's to output
        else:
            out.append(False)

    true_count = int(len(queries) / 2)
    return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


def iSVT3(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=4.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        # INCORRECT: noise added to queries doesn't scale with N
        eta2 = prng.laplace(scale=4.0 / (3.0 * epsilon))
        if query + eta2 > noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)

    true_count = int(len(queries) / 2)
    return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


def iSVT4(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=2.0 * N / epsilon)
        if query + eta2 > noisy_T:
            # INCORRECT: Output the noisy query instead of True
            out.append(query + eta2)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False), out[-1]
