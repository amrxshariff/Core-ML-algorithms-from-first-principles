import numpy as np

from optimisation import quadratic
from optimisation.rosenbrock import rosenbrock, rosenbrock_grad


def gradient_descent(f, grad_f, x0, lr, n_iter):
    """"
    Gradient Descent optimisation where f is the function to be optimised, 
    grad_f is the gradient of f, 
    x0 is the initial guess, 
    lr is the learning rate 
    and n_iter will depict the start point as occupying row 0 so n_iter - 1 updates are performed.

    The shape of the history array is (n_iter, len(x0)) and contains the values of x at each iteration.

    The function hands back a tuple of the final value of x and the history array.

    History[0] is the first point which the optimiser starts from, and History[-1] is the final point which the optimiser ends at.
    """ 

    x = np.array(x0, dtype = float)
    history = np.zeros((n_iter, len(x0)), dtype = float)
    history[0] = x
    for i in range(1, n_iter):
        x = x - lr*grad_f(x)
        history[i] = x
        if np.linalg.norm(grad_f(x)) < 1e-6:
                print(f"Converged after {i} iterations.")
                return x, history, i
        
    return x, history, n_iter


if __name__ == "__main__":

    x_final, history, n_iter = gradient_descent(rosenbrock, rosenbrock_grad, x0 = [0.5, 0.5], lr = 0.001, n_iter = 100000)

    print("shape:", history.shape)
    print("first row:", history[0])
    print("last row:", history[-1])
    print("consistent:", np.allclose(history[-1], x_final))
    print("f values:", [round(rosenbrock(p), 4) for p in history[:10]])
    print(f"Algorithm stoppped after {n_iter} iterations.")

    

    