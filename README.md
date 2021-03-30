# StatDP 
[![Github Actions](https://github.com/yxwangcs/statdp/workflows/build/badge.svg)](https://github.com/yxwangcs/statdp/actions?workflow=build) [![codecov](https://codecov.io/gh/yxwangcs/statdp/branch/master/graph/badge.svg)](https://codecov.io/gh/yxwangcs/statdp)

Statistical Counterexample Detector for Differential Privacy.

## Usage
We assume your algorithm implementation has the folllowing signature: `(prng, queries, epsilon, ...)` (Pseudo-random generator, list of queries, privacy budget and extra arguments).

Throughout your algorithm, any random number must be generated through the provided generator (i.e., `prng`) for better scalability with multiple cores. It is an instance of [`numpy.random.Generator`](https://numpy.org/doc/stable/reference/random/generator.html) which supports a collection of standard distributions.

Then you can simply call the detection tool with automatic database generation and event selection:
```python
from statdp import detect_counterexample

def your_algorithm(prng, queries, epsilon, ...):
    # your algorithm implementation here
    # prng must be used instead of np.random
    prng.laplace(loc=0, scale=1 / epsilon)
 
if __name__ == '__main__':
    # algorithm privacy budget argument(`epsilon`) is needed
    # otherwise detector won't work properly since it will try to generate a privacy budget
    result = detect_counterexample(your_algorithm, {'epsilon': privacy_budget}, test_epsilon)
```

The result is returned in variable `result`, which is stored as `[(epsilon, p, d1, d2, kwargs, event), (...)]`. 

The `detect_counterexample` accepts multiple extra arguments to customize the process, check the signature and notes of `detect_counterexample` method to see how to use.

```python
def detect_counterexample(algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                          event_iterations=100000, detect_iterations=500000, cores=None, sensitivity=ALL_DIFFER,
                          quiet=False, loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
    :param databases: The databases to run for detection, optional.
    :param num_input: The length of input to generate, not used if database param is specified.
    :param event_iterations: The iterations for event selector to run.
    :param detect_iterations: The iterations for detector to run.
    :param cores: The number of max processes to set for multiprocessing.Pool(), os.cpu_count() is used if None.
    :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
    :param quiet: Do not print progress bar or messages, logs are not affected.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
```

## Install
We do provide a docker container for experiment, use `docker pull cmlapsu/statdp` to pull the container with anaconda built in, then run `docker run --rm -it cmlapsu/statdp`. 

However, for the best performance we recommend installing `statdp` in a `conda` virtual environment (or `venv` if you prefer, the setup is similar):

```bash
# we use python 3.8, but 3.6 and above should work fine
conda create -n statdp anaconda python=3.8
conda activate statdp
# install dependencies from conda for best performance
conda install numpy numba matplotlib sympy tqdm coloredlogs pip
# install icc_rt compiler for best performance with numba, this requires using intel's channel
conda install -c intel icc_rt
# install the remaining non-conda dependencies and statdp 
pip install .
```
Then you can run `examples/benchmark.py` to run the experiments we conducted in the paper.


## Visualizing the results
A nice python library `matplotlib` is recommended for visualizing your result. 

There's a python code snippet at `/examples/benchmark.py`(`plot_result` method) to show an example of plotting the results.

Then you can generate a figure like the iSVT 4 in our paper.
![iSVT4](https://raw.githubusercontent.com/yxwangcs/StatDP/master/examples/iSVT4.svg?sanitize=true)

## Customizing the detection
Our tool is designed to be modular and components are fully decoupled. You can write your own `input generator`/`event selector` and apply them to `hypothesis test`.

In general the detection process is 

`test_epsilon --> generate_databases --((d1, d2, kwargs), ...), epsilon--> select_event --(d1, d2, kwargs, event), epsilon--> hypothesis_test --> (d1, d2, kwargs, event, p-value), epsilon`
 
You can checkout the definition and docstrings of the functions respectively to define your own generator/selector. Basically the `detect_counterexample` function in `statdp.core` module is just shortcut function to take care of the above process for you.

`test_statistics` function in `hypotest` module can be used universally by all algorithms (this function is to calculate p-value based on the observed statistics). However, you may need to design your own generator or selector for your own algorithm, since our input generator and event selector are designed to work with numerical queries on databases.

## Citing this work

You are encouraged to cite the following [paper](https://arxiv.org/pdf/1805.10277.pdf) if you use this tool for academic research:

```bibtex
@inproceedings{ding2018detecting,
  title={Detecting Violations of Differential Privacy},
  author={Ding, Zeyu and Wang, Yuxin and Wang, Guanhong and Zhang, Danfeng and Kifer, Daniel},
  booktitle={Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
  pages={475--489},
  year={2018},
  organization={ACM}
}
```

## License
[MIT](https://github.com/yxwangcs/statdp/blob/master/LICENSE).
