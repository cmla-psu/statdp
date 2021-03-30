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
import logging
import pytest
from flaky import flaky
from statdp.algorithms import (SVT, iSVT1, iSVT2, iSVT3, iSVT4, noisy_max_v1a,
                               noisy_max_v1b, noisy_max_v2a, noisy_max_v2b, histogram, histogram_eps)
from statdp import detect_counterexample, ALL_DIFFER, ONE_DIFFER

correct_algorithms = (
    (noisy_max_v1a, {}, 5, ALL_DIFFER),
    (noisy_max_v2a, {}, 5, ALL_DIFFER),
    (SVT, {'N': 1, 'T': 0.5}, 10, ALL_DIFFER),
    (histogram, {}, 5, ONE_DIFFER)
)
incorrect_algorithms = (
    (noisy_max_v1b, {}, 5, ALL_DIFFER),
    (noisy_max_v2b, {}, 5, ALL_DIFFER),
    (iSVT1, {'N': 1, 'T': 1}, 10, ALL_DIFFER),
    (iSVT2, {'N': 1, 'T': 1}, 10, ALL_DIFFER),
    (iSVT3, {'N': 1, 'T': 1}, 10, ALL_DIFFER),
    (iSVT4, {'N': 1, 'T': 1}, 10, ALL_DIFFER),
    (histogram_eps, {}, 5, ONE_DIFFER)
)


@pytest.mark.parametrize('algorithm', correct_algorithms,
                         ids=[algorithm[0].__name__ for algorithm in correct_algorithms])
# due to the statistical and randomized nature, use flaky to allow maximum 5 runs of failures
@flaky(max_runs=5)
def test_correct_algorithm(algorithm):
    func, kwargs, num_input, sensitivity = algorithm
    kwargs.update({'epsilon': 0.7})
    result = detect_counterexample(func, (0.6, 0.7, 0.8), kwargs,
                                   num_input=num_input, loglevel=logging.DEBUG, sensitivity=sensitivity)
    assert isinstance(result, list) and len(result) == 3
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
    epsilon, p, *extras = result[1]
    assert p >= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
    epsilon, p, *extras = result[2]
    assert p >= 0.95, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)


@pytest.mark.parametrize('algorithm', incorrect_algorithms,
                         ids=[algorithm[0].__name__ for algorithm in incorrect_algorithms])
@flaky(max_runs=5)
def test_incorrect_algorithm(algorithm):
    func, kwargs, num_input, sensitivity = algorithm
    kwargs.update({'epsilon': 0.7})
    result = detect_counterexample(func, 0.7, kwargs,
                                   num_input=num_input, loglevel=logging.DEBUG, sensitivity=sensitivity)
    assert isinstance(result, list) and len(result) == 1
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)


@flaky(max_runs=5)
def test_large_iterations():
    result = detect_counterexample(SVT, 0.7, {'T': 0.5, 'N': 1, 'epsilon': 0.7},
                                   num_input=10, loglevel=logging.DEBUG, sensitivity=ALL_DIFFER,
                                   event_iterations=int(2e6), detect_iterations=int(5e6))
    epsilon, p, *extras = result[0]
    assert p >= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
