import numpy as np

def rastrigin(x, d = 1):
    ubound = 5.12
    lbound = -ubound

    if abs(x) > ubound:
        x = np.random.uniform(-5.12, 5.12)

    if d == 1:
        return 10 + x**2 - 10 * np.cos(2* np.pi * x)

