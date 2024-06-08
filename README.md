# code-checkpointed-ode-solutions

**Warning:**
This is experiment code.
The algorithm itself is part of probdiffeq ([link](https://pnkraemer.github.io/probdiffeq/)), which can be installed via
```commandline
pip install probdiffeq
```
Unless you are looking for specific experiment configurations,
you are probably interested in Probdiffeq, not this repository here.

But if you want to work with this repository, proceed as follows.

## Installation

We use Python 3.12 for all experiments.
Older versions might also work.
Ensure that JAX is installed.
Then, run
```commandline
pip install .
```
which installs the source code plus all dependencies.

## Experiments

- [ ] Visualise a probabilistic ODE solution
- [ ] Display the effect of fixed versus adaptive steps on stiff van-der-Pol
- [ ] Work-precision diagram on Pleiades
- [ ] Work-precision diagram on the three-body problem
- [ ] Work-precision diagram on Lotka-Volterra
- [ ] Train a Neural ODE