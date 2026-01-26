
import numpy as np

#### Temperature Schedules

def geometric(T0, Tf, curr_iter, total_iter = 100):
    T_sched = T0 * (Tf/T0)**((curr_iter-1)/(total_iter - 1))
    curr_iter += 1

    return T_sched, curr_iter

#### Thermalization process

def therm(x, func, delta, T, iter):
    '''
    therm() takes the current input x and creates a proposal for the
    next step by random perturbation. By a decision process,
    it decides if the proposal is accepted or rejected. Repeats
    iter times to thermalize the input.

    
    :param x: input value of current iteration
    :param func: function to optimize
    :param delta: step size
    :param T: current temperature
    :param iter: number of thermalization steps (reduce autocorrelation)
    '''

    x_curr = x

    for i in range(iter):
        x_prop = x_curr + delta * (np.random.uniform(-1,1))

        eval_prop = func(x_prop)
        eval_curr = func(x_curr)

        if eval_prop >= eval_curr:
            x_curr = x_prop
        else:
            rand = np.random.uniform(0,1)
            test = np.exp((eval_prop - eval_curr)/T)
            if test > rand:
                x_curr = x_prop
            else:
                pass
    
    return x_curr
    

#### main simulated annealing algorithm
def sim_anneal(x0, T0, Tf, func, delta, total_iter, temp = geometric, therm_iter = 5):
    x = x0
    T = T0
    x_values = np.empty(total_iter)
    current_iter = 1

    for i in range(total_iter):
        x_values[i] = therm(x, func, delta, T, therm_iter)
        T, current_iter = geometric(T, Tf, current_iter, total_iter)
    
    return x_values


