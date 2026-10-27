# Numerical foundations

![tests](https://github.com/amrxshariff/numerical-foundations/actions/workflows/tests.yml/badge.svg)

Core machine learning algorithms implemented from first principles in NumPy, each
verified against a reference implementation and accompanied by one measured finding.

## Projects

| Project | Finding |
| --- | --- |
| Gradient descent | |
| k-NN and distance metrics | |
| Naive Bayes | |
| PCA and eigenfaces | |
| Markov text generation | |

## Running it

    python -m venv .venv
    .venv/Scripts/activate
    pip install -e ".[dev]"
    pytest