import matplotlib.pyplot as plt
import numpy as np

from optimisation.descent import gradient_descent
from optimisation.rosenbrock import rosenbrock, rosenbrock_grad


def loss_trajectories():
    """
    Plot the loss trajectories for both the quadratic and rosenbrock functions during gradient descent.

    """
    hessian = np.array([802.0, -400.0, -400.0,200.0])
    max_eigenvalue = np.max(np.abs(np.linalg.eigvals(hessian.reshape(2,2))))

    learning_rates = [0.0001, 0.025, 0.05, 0.075, 0.1]

    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        _, history, _ = gradient_descent(rosenbrock, rosenbrock_grad, x0 = [0.5, 0.5], lr = lr, n_iter = 1000)
        loss_history = np.array(history)
        losses = [rosenbrock(p) for p in loss_history]
        if lr > 1.0 / max_eigenvalue:
            plt.plot(losses, label=f'lr={lr} (diverges)', linestyle='--')
        else:
            plt.plot(losses, label=f'lr={lr} (converges)', linestyle='-')
        

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Trajectories for Rosenbrock Function')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    loss_trajectories()



