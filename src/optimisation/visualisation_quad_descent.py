import matplotlib.pyplot as plt
import numpy as np

from optimisation.descent import gradient_descent
from optimisation.quadratic import quadratic, quadratic_grad


def visualisation_descent():
    """
    Contour plot of the quadratic with the descent path drawn over it

    """

    x0 = np.linspace(-2, 2)
    x1 = np.linspace(-2, 2)
    X, Y = np.meshgrid(x0, x1)
    z = quadratic([X, Y])

    _, history, _ = gradient_descent(quadratic, quadratic_grad, x0 = [0.5, 0.5], lr = 0.01, n_iter = 100)

    history = np.array(history)
    x_vals = history[:,0]
    y_vals = history[:,1]

    plt.contour(X, Y, z, levels=20)
    plt.plot(x_vals, y_vals, marker='o', color='red', label='Descent Path')
    plt.colorbar()
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('Contour plot of the quadratic function')
    plt.show()

if __name__ == "__main__":
    visualisation_descent()