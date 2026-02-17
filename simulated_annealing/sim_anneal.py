import numpy as np
from simulated_annealing.test_funcs import Func

class Optimize(Func):
    def __init__(self, name, x, T, Tf, total_iter, delta):
        super().__init__(name)
        
        self.d = np.size(x) # dimension of input

        self.x = x # data; scalar or vector
        self.T = T  # initial temp; scalar
        self.Tf = Tf # final temp; scalar
        self.total_iter = total_iter # no. of iterations; scalar
        self.delta = delta # step size; scalar

        self.formula = self.funcs[self.name][0] # formula for the function
        self.bounds = self.funcs[self.name][1] # bounds for the function


    ### Determine if correct datatype, converts to array
    def is_type(self):
        if type(self.x) in [list, tuple]:
            self.x = np.array(self.x) # convert to array
            return True
        elif type(self.x) in [float, int]:
            self.x = np.array([self.x]) # put scalar in list first, then array
            return False


    ### Different Temp Schedules

    # A simple geometric temperature schedule.
    def simple_geometric(self):
        if self.T > self.Tf:
            self.T = self.T*(1-0.01) # 0.01 is epsilon
        else:
            pass

    ### Proposal step for a single element
    def proposal(self, x):
        try:
            return x + self.delta * (np.random.uniform(-1,1))
        except:
            raise Exception('Error: proposal configuration calculation.')

    ### Optimizer for minimum
    def opt_min(self):
        self.is_type()
        y = np.apply_along_axis(self.proposal, 0, self.x) # proposal configuration
        
        # Evaluate function at proposal and current configuration.
        eval_y = self.eval(y) 
        eval_x = self.eval(self.x)
        
        # Chooses config that is less likely; else use the Metropolis acceptance probability
        if eval_y <= eval_x:
            self.x = y
        else:
            test = np.exp(-(eval_y - eval_x)/self.T)
            prob = np.random.uniform(0,1)
            if prob < test:
                self.x = y
            else:
                pass
        
        return self.x

    ### Main simulated annealing algorithm.

    def sim_ann(self, schedule):
        all_x = np.empty((self.total_iter, self.d)) # Empty array to fit all chosen configurations
        
        for i in range(self.total_iter):
            all_x[i] = np.round(self.opt_min()) # adds the new configuration to the list
            schedule() # change the temperature according to the schedule

        return all_x


        