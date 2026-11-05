import numpy as np

from optimisation.quadratic import quadratic, quadratic_grad
from optimisation.rosenbrock import rosenbrock, rosenbrock_grad


def test_gradient_matches_finite_difference():
    """
    Test that the analytical gradient matches the finite difference approximation for both quadratic and rosenbrock functions.

    """

    x0 = np.array([0.5,0.5])
    atol = 1e-5

    surfaces = [
        (quadratic, quadratic_grad),
        (rosenbrock, rosenbrock_grad)
    ]

    for f,grad_f in surfaces:
        for i in range(len(x0)):
            x_plus = np.copy(x0)
            x_plus[i] += atol
            x_minus = np.copy(x0)
            x_minus[i] -= atol
            finite_diff = (f(x_plus) - f(x_minus)) / (2 * atol)
        analytical = grad_f(x0)[i]


        assert np.allclose(finite_diff, analytical, atol=atol), f"Gradient mismatch for {f.__name__} at dimension {i}"

if __name__ == "__main__":
    test_gradient_matches_finite_difference()
    print("Test passed.")






  