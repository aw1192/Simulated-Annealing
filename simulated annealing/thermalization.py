import numpy as np

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
            rand = np.uniform(0,1)
            test = np.exp((eval_prop - eval_curr)/T)
            if test > rand:
                x_curr = x_prop
            else:
                pass
    
    return x_curr
    