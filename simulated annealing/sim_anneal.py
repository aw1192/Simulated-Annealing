from thermalization import therm
from temp import geometric

import numpy as np

def sim_anneal(x0, T0, Tf, func, delta, total_iter, temp = geometric, therm_iter = 5):
    x = x0
    T = T0
    x_values = np.empty(total_iter)
    current_iter = 1

    for i in range(total_iter):
        x_values[i] = therm(x, func, delta, T, therm_iter)
        T, current_iter = geometric(T, Tf, current_iter, total_iter)
    
    return x_values
