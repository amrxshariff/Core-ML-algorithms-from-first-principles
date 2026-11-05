import numpy as np

def rosenbrock(x):
    """
    f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        
    Analytical minimum at f (x,y) = [0,0] at [1.0,1.0]
    """

    x_val, y_val = x[0], x[1]
    return (1 - x_val)**2 + 100 * (y_val - x_val**2)**2


def rosenbrock_grad(x):
    """Analytical gradient of rosenbrock: -2*(1-x) - 400*x*(y-x^2), 200*(y-x^2)
    
    Returns 0 at x = [1.0,1.0]
    """
    x_val, y_val = x[0], x[1]
    df_dx = -2*(1-x_val) - 400*x_val*(y_val-x_val**2)
    df_dy = 200*(y_val-x_val**2)
    return np.array([df_dx, df_dy])