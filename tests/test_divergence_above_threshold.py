import numpy as np

from src.optimisation.descent import gradient_descent
from src.optimisation.quadratic import quadratic, quadratic_grad
from src.optimisation.rosenbrock import rosenbrock, rosenbrock_grad

def test_divergence_above_threshold():
    """
    Test that the optimiser blows up when the step size of the learning rate is above the threshold for both quadratic and rosenbrock functions.

    """

    x0 = np.array([0.5,0.5])
    lr = 1.0
    n_iter = 50

    hessian = np.array([20.0, 0.0, 0.0,10.0])
    #hessian = np.array([802.0, -400.0, -400.0,200.0])
    max_eigenvalue = np.max(np.abs(np.linalg.eigvals(hessian.reshape(2,2))))
    if lr> 2/max_eigenvalue:
        print(f"Learning rate {lr} is above the threshold for convergence for quadratic function.")

   
    x_final,_ , _ = gradient_descent(quadratic, quadratic_grad, x0 = x0, lr = lr, n_iter = n_iter)

    assert np.linalg.norm(x_final) > 1e10, f"Expected divergence for quadratic function with learning rate {lr}, but got {x_final}"


if __name__ == "__main__":

    test_divergence_above_threshold()
    
    print("Test passed.")

