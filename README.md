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

- [x] Work-precision diagram on a simple ODE problem to show some raw results
- [x] Work-precision diagram on a slightly harder ODE problem to avoid reviewers asking for "more experiments"
- [ ] Train a Neural ODE to show that we can use this algorithm
- [ ] Visualise a probabilistic ODE solution for some eye-candy
- [ ] Display the effect of fixed versus adaptive steps on stiff van-der-Pol to motivate adaptive step-size selection
- [ ] Compare fixedpoint forward-pass to filter forward-pass and smoother forward-pass to demonstrate the 5% increase in costs

## Working with the source

After following the installation instructions above, the test-dependencies are installed.
To run the tests, run
```commandline
make test
```
