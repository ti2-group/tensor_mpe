# Tensor MPE

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Build](#build)
- [Usage](#usage)

## What it does

This repository is a proof-of-concept implentation of MPE in tensor networks. It uses derivatives of a tensor contraction over the tropical (max-plus) semiring, that maximizes the probability in a discrete graphical model.
The main file to execute is `src/runtime.py`. When you execute this program, it first generates hidden markov models in the form of tensor trains, visualized here:

![Hidden Markov Model](graphics/hidden_markov_model.png)

It generates a bunch of random queries for each model that you can customize in `src/random_benchmarks.py`.
Then, for each model, every query is run 10 times and the runtimes are plotted in `graphics/runtimes.png`.

![graphics/runtime.png](graphics/runtimes.png)

### File Contents

- `src/graphical_model.py`: Graphical model class with functions for MPE and log probability.
- `src/my_bitsets.py`: Simple bitset implementation. This is useful because we want ordered, hashable sets of variables.
- `src/contractions.py`: Simple implementation of an tensor contraction along a contraction path.
- `src/_tropical_bmm.hpp`, `src/_tropical_bmm.cpp`: C++ implementation of a batch-matrix-multiplication in the tropical semiring with forward and backward mode for autograd.
- `src/topical_bmm.pyx`: Cython wrapper for the C++ implementation and interface for pairwise einsum which is used for small contractions along the contraction path.
- `src/random_benchmarks.py`: Generates random models and random queries for each model. It is automatically seeded, but you can disable that with the flag `auto_seed = False`.
- `src/runtime.py` Uses `src/random_benchmarks.py` to generate random models and benchmarks the MPE implementation on it.

## Requirements

The requirements are:

```text
cython pytorch opt_einsum tqdm matplotlib
```

You can install them using

```text
pip install -r requirements.txt
```

or

```text
conda install --file requirements.txt
```

## Build

```text
cythonize -i src/tropical_bmm.pyx
```

## Usage

```text
python src/runtime.py
```
