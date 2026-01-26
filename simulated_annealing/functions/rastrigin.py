import numpy as np

def rastrigin(x, d=1):
    if d == 1:
        return 10 + x**2 - 10* np.cos(2 * np.pi * x)
    else:
        pass
    return 10