# Adaptive probabilistic ODE solvers without adaptive memory requirements

This repository contains the experiments for:

```bibtex
@article{krämer2024adaptive,
    title={Adaptive Probabilistic {ODE} Solvers Without Adaptive Memory Requirements},
    author={Nicholas Krämer},
    year={2024},
    eprint={2410.10530},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2410.10530},
}
```

[Here](https://arxiv.org/abs/2410.10530) is a link to the preprint.

## Warning
This repository contains only experiment code.
We have made the new algorithm a part of probdiffeq ([link](https://pnkraemer.github.io/probdiffeq/)), which can be installed via
```commandline
pip install probdiffeq
```
Unless you are looking for specific experiment configurations,
you are probably interested in probdiffeq, not this repository here.

But if you want to work with this repository, proceed as follows.

## Installation

We use Python 3.12 for all experiments.
Older versions might also work.
Ensure that JAX is installed.
Then, run
```commandline
pip install .
```
This command installs the source code plus all dependencies.

## Working with the source

After following the installation instructions above, the test dependencies are installed.

To run the tests, run
```commandline
make test
```
To format the code, run
```commandline
make format-and-lint
```
