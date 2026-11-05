import numpy as np

def quadratic(x):
    """
    f(x) = 10*x[0]^2 + 5*x[1]^2
        
    Analytical minimum at f = 0, x = [0.0,0.0]
    """

    x = np.asarray(x, dtype = float)
    return 10.0*x[0]**2 + 5.0*x[1]**2


def quadratic_grad(x):
    """Analytical gradient of quadratic: 20*x0 + 10*x1
    
    Returns 0 at x = [0.0,0.0]
    """
    x = np.asarray(x, dtype=float)
    df_dx0 = 20*x[0]
    df_dx1 = 10*x[1]
    return np.array([df_dx0, df_dx1])

if __name__ == "__main__":
    print(quadratic([10.0, 5.0]))
    print(quadratic([-23.0, 15.0]))
    print(quadratic([80.0, 40.0]))
    print(quadratic_grad([0.0, 0.0]))
    print(quadratic_grad([1.0, 1.0]))