import numpy as np


def test_quadratic_convergence():
    from optimisation.descent import gradient_descent
    from optimisation.quadratic import quadratic, quadratic_grad

    x_min = np.array([0.0,0.0])

    x_final, _, _ = gradient_descent(quadratic, quadratic_grad, x0 = [0.5, 0.5], lr = 0.01, n_iter = 1000)

    assert np.allclose(x_final, x_min, atol=1e-6), f"Expected {x_min}, got {x_final}"

if __name__ == "__main__":
    test_quadratic_convergence()
    print("Test passed.")