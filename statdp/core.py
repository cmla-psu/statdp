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
import math
import itertools
import logging
import numpy as np

logger = logging.getLogger(__name__)


def run_algorithm(algorithm, d1, d2, kwargs, event, total_iterations):
    """ Run the algorithm for :iteration: times, count and return the number of iterations in :event:,
    event search space is auto-generated if not specified.
    :param algorithm: The algorithm to run.
    :param d1: The D1 input to run.
    :param d2: The D2 input to run.
    :param kwargs: The keyword arguments for the algorithm.
    :param event: The event to test, auto generate event search space if None.
    :param total_iterations: The iterations to run.
    :return: [(cx, cy), ...], [(d1, d2, kwargs, event), ...]
    """
    if not callable(algorithm):
        raise ValueError('Algorithm must be callable')
    prng = np.random.default_rng()
    # support multiple return values, each return value is stored as a row in result_d1 / result_d2
    # e.g if an algorithm returns (1, 1), result_d1 / result_d2 would be like
    # [
    #   [x, x, x, ..., x],
    #   [x, x, x, ..., x]
    # ]

    # get return type by a sample run
    all_possible_events = None
    event_dict = {}
    sample_result = algorithm(prng, d1, **kwargs)

    # since we need to store the output in intermediate variables (`result_d1` and `result_d2`), if the total
    # iterations are very large, peak memory usage would kill the program, therefore we divide the
    if total_iterations > int(1e6):
        logger.debug('Iterations too large, divide into different pieces')
        iteration_tuple = [int(1e6) for _ in range(math.floor(total_iterations / 1e6))] + [total_iterations % int(1e6)]
    else:
        iteration_tuple = (total_iterations,)
    for iterations in iteration_tuple:
        if np.issubdtype(type(sample_result), np.number):
            result_d1 = (np.fromiter((algorithm(prng, d1, **kwargs) for _ in range(iterations)),
                                     dtype=type(sample_result), count=iterations),)
            result_d2 = (np.fromiter((algorithm(prng, d2, **kwargs) for _ in range(iterations)),
                                     dtype=type(sample_result), count=iterations),)
        elif isinstance(sample_result, (tuple, list)):
            # create a list of numpy array, each containing the output from running
            result_d1, result_d2 = [np.empty(iterations, dtype=type(sample_result[result_index])) for result_index in
                                    range(len(sample_result))], \
                                   [np.empty(iterations, dtype=type(sample_result[result_index])) for result_index in
                                    range(len(sample_result))],

            for iteration_number in range(iterations):
                out_1 = algorithm(prng, d1, **kwargs)
                out_2 = algorithm(prng, d2, **kwargs)
                for row, (value_1, value_2) in enumerate(zip(out_1, out_2)):
                    result_d1[row][iteration_number] = value_1
                    result_d2[row][iteration_number] = value_2
        else:
            raise ValueError(f'Unsupported return type: {type(sample_result)}')

        # if possible events are not determined yet
        if not all_possible_events:
            # get desired search space for each return value
            event_search_space = []
            if event is None:
                for row in range(len(result_d1)):
                    # determine the event search space based on the return type
                    combined_result = np.concatenate((result_d1[row], result_d2[row]))
                    unique = np.unique(combined_result)

                    # categorical output
                    if len(unique) < iterations * 0.002:
                        event_search_space.append(tuple(int(key) for key in unique))
                    else:
                        combined_result.sort()
                        # find the densest 70% range
                        search_range = int(0.7 * len(combined_result))
                        search_max = min(range(search_range, len(combined_result)),
                                         key=lambda x: combined_result[x] - combined_result[x - search_range])
                        search_min = search_max - search_range

                        event_search_space.append(
                            tuple((-float('inf'), float(alpha)) for alpha in
                                  np.linspace(combined_result[search_min], combined_result[search_max], num=10)))

                logger.debug(f"search space is set to {' × '.join(str(event) for event in event_search_space)}")
            else:
                # if `event` is given, it should have the corresponding events for each return value
                if len(event) != len(result_d1):
                    raise ValueError('Given event should have the same dimension as return value.')
                # here if the event is given, we carefully construct the search space in the following format:
                # [first_event] × [second_event] × [third_event] × ... × [last_event]
                # so that when the search begins, only one possible combination can happen which is the given event
                event_search_space = ((separate_event,) for separate_event in event)
            all_possible_events = tuple(itertools.product(*event_search_space))

        for event in all_possible_events:
            cx_check, cy_check = np.full(iterations, True, dtype=np.bool), np.full(iterations, True, dtype=np.bool)
            # check for all events in the return values
            for row in range(len(result_d1)):
                if np.issubdtype(type(event[row]), np.number):
                    cx_check = np.logical_and(cx_check, result_d1[row] == event[row])
                    cy_check = np.logical_and(cy_check, result_d2[row] == event[row])
                else:
                    cx_check = np.logical_and(cx_check, np.logical_and(result_d1[row] > event[row][0],
                                                                       result_d1[row] < event[row][1]))
                    cy_check = np.logical_and(cy_check, np.logical_and(result_d2[row] > event[row][0],
                                                                       result_d2[row] < event[row][1]))

            cx, cy = np.count_nonzero(cx_check), np.count_nonzero(cy_check)
            if event not in event_dict:
                event_dict[event] = (cx, cy)
            else:
                old_cx, old_cy = event_dict[event]
                event_dict[event] = cx + old_cx, cy + old_cy

    counts, input_event_pairs = [], []
    for event, (cx, cy) in event_dict.items():
        counts.append((cx, cy) if cx > cy else (cy, cx))
        input_event_pairs.append((d1, d2, kwargs, event))
    return counts, input_event_pairs
