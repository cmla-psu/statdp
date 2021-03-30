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
import math
import enum

logger = logging.getLogger(__name__)


class Sensitivity(enum.Enum):
    ALL_DIFFER = 0
    ONE_DIFFER = 1


ALL_DIFFER = Sensitivity.ALL_DIFFER
ONE_DIFFER = Sensitivity.ONE_DIFFER


def generate_arguments(algorithm, d1, d2, default_kwargs):
    """
    :param algorithm: The algorithm to test for.
    :param d1: The database 1
    :param d2: The database 2
    :param default_kwargs: The default arguments that are given or have a default value.
    :return: Extra argument needed for the algorithm besides Q and epsilon.
    """
    arguments = algorithm.__code__.co_varnames[:algorithm.__code__.co_argcount]
    if arguments[2] not in default_kwargs:
        logger.error(f'The third argument {arguments[2]} (privacy budget) is not provided!')
        return None

    return default_kwargs


def generate_databases(algorithm, num_input, default_kwargs, sensitivity=ALL_DIFFER):
    """
    :param algorithm: The algorithm to test for.
    :param num_input: The number of inputs to be generated
    :param default_kwargs: The default arguments that are given or have a default value.
    :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
    :return: List of (d1, d2, args) with length num_input
    """
    if not isinstance(sensitivity, Sensitivity):
        raise ValueError('sensitivity must be statdp.ALL_DIFFER or statdp.ONE_DIFFER')

    # assume maximum distance is 1
    d1 = [1 for _ in range(num_input)]
    candidates = [
        (d1, [0] + [1 for _ in range(num_input - 1)]),  # one below
        (d1, [2] + [1 for _ in range(num_input - 1)]),  # one above
    ]

    if sensitivity == ALL_DIFFER:
        candidates.extend([
            (d1, [2] + [0 for _ in range(num_input - 1)]),  # one above rest below
            (d1, [0] + [2 for _ in range(num_input - 1)]),  # one below rest above
            # half half
            (d1, [2 for _ in range(int(num_input / 2))] + [0 for _ in range(num_input - int(num_input / 2))]),
            (d1, [2 for _ in range(num_input)]),  # all above
            (d1, [0 for _ in range(num_input)]),  # all below
            # x shape
            ([1 for _ in range(int(math.floor(num_input / 2.0)))] + [0 for _ in range(int(math.ceil(num_input / 2.0)))],
             [0 for _ in range(int(math.floor(num_input / 2.0)))] + [1 for _ in range(int(math.ceil(num_input / 2.0)))])
        ])

    return tuple((d1, d2, generate_arguments(algorithm, d1, d2, default_kwargs)) for d1, d2 in candidates)
