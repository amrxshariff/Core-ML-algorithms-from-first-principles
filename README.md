# Numerical foundations

![tests](https://github.com/amrxshariff/Core-ML-algorithms-from-first-principles/actions/workflows/tests.yml/badge.svg)

Core machine learning algorithms implemented from first principles in NumPy, each
verified against a reference implementation and accompanied by one measured finding.

## Projects

| Project | Finding |
| --- | --- |
| Gradient descent | "Gradient Descent from first principles: Started by writing functions for quadratic and quadratic_grad,  Noted analytical minimum to be able to assinst it when choosing a learning rate. Proceeded to define function which returned the final point of the gradicent + full history of positions + convergence stop to applied to both quadratic and Rosenbrock. Ran both functions from a standard point (0.5,0.5), confirming it reached minimum on both. Convergence limit was set to the final point being within 1e-6 of the known minimum. Divergence threshold was then computed following the fact that learning rate must not exceed 2/max_eigenvalue computed from the hessian of both of functions. Plots for gradient descent on both functions and log-loss / step count for 5 different learning rates were then saved with the contours and divergence thresholds incorporated resepectively."          

Findings - Hessian computed for simple quadratic and more complex Rosenbrock functions which enabled being able to find the maximum eigenvalue to be able to comput the divergence threshold. Divergence threshold - learning rate / step size must be smaller than 2/ max eigen value of the function to ensure stability, otherwise gradient descent will not converge. Tests computed for learning rates slightly over and under the learning rate threshold for divergence were carried out and the expected convergence and divergence of gradient descent was observed through the figures: "loss_trajectory_[name of function]". |
| k-NN and distance metrics | |
| Naive Bayes | |
| PCA and eigenfaces | |
| Markov text generation | |

## Running it

    python -m venv .venv
    .venv/Scripts/activate
    pip install -e ".[dev]"
    pytest
