# MIT License
#
# Copyright (c) 2018-2019 Yuxin Wang
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
import functools
import logging

import numpy as np
import tqdm

from statdp.hypotest import test_statistics
from statdp.core import run_algorithm

logger = logging.getLogger(__name__)


def _evaluate_input(input_triplet, algorithm, iterations):
    d1, d2, kwargs = input_triplet
    return run_algorithm(algorithm, d1, d2, kwargs, None, iterations)


def select_event(algorithm, input_list, epsilon, iterations, process_pool, quiet=False):
    """
    :param algorithm: The algorithm to run on.
    :param input_list: list of (d1, d2, kwargs) input pair for the algorithm to run.
    :param epsilon: Test epsilon value.
    :param iterations: The iterations to run algorithms.
    :param process_pool: The multiprocessing.Pool() to use.
    :param quiet: Do not print progress bar or messages, logs are not affected, default is False.
    :return: (d1, d2, kwargs, event) pair which has minimum p value from search space.
    """
    if not callable(algorithm):
        raise ValueError('Algorithm must be callable')

    # fill in other arguments for _evaluate_input function, leaving out `input` to be filled
    partial_evaluate_input = functools.partial(_evaluate_input, algorithm=algorithm, iterations=iterations)

    threshold = 0.001 * iterations * np.exp(epsilon)

    event_evaluator = tqdm.tqdm(process_pool.imap_unordered(partial_evaluate_input, input_list),
                                desc='Finding best inputs/events', total=len(input_list), unit='input', leave=False,
                                disable=quiet)
    # flatten the results for all input/event pairs
    counts, input_event_pairs, p_values = [], [], []
    for local_counts, local_input_event_pair in event_evaluator:
        # put the results in the list for later references
        counts.extend(local_counts)
        input_event_pairs.extend(local_input_event_pair)

        # calculate p-values based on counts
        for (cx, cy) in local_counts:
            p_values.append(test_statistics(cx, cy, epsilon, iterations) if cx + cy > threshold else float('inf'))

    # log the information for debug purposes
    for ((d1, d2, kwargs, event), (cx, cy), p) in zip(input_event_pairs, counts, p_values):
        logger.debug(f"d1: {d1} | d2: {d2} | kwargs: {kwargs} | event: {event} | p-value: {p:5.3f} | "
                     f"cx: {cx} | cy: {cy} | ratio: {float(cy) / cx if cx != 0 else float('inf'):5.3f}")

    # find an (d1, d2, kwargs, event) pair which has minimum p value from search space
    return input_event_pairs[np.asarray(p_values).argmin()]
