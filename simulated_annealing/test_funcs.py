import numpy as np

def rastrigin(x, d = 1):
    ubound = 5.12 # put in Optimizer
    lbound = -ubound

    if abs(x) > ubound:
        x = np.random.uniform(-5.12, 5.12) # change to bounds

    if d == 1:
        return 10 + x**2 - 10 * np.cos(2* np.pi * x)

def eggholder(x):
    x1 = np.array(x)[0]
    x2 = np.array(x)[1]

    term1 = -(x2+47)* np.sin(np.sqrt(abs(x2+(x1/2)+47)))
    term2 = x1*np.sin(np.sqrt(abs(x1-(x2+47))))
    
    return term1 - term2
