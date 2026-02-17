import numpy as np
from simulated_annealing.test_funcs import Func

class Optimize(Func):
    def __init__(self, name, x, T, Tf, total_iter, delta):
        super().__init__(name)
        
        self.d = np.size(x) # dimension of input
        self.all_x = np.empty((total_iter, self.d)) # Empty array to fit all chosen configurations
        
        self.all_evals = np.empty(total_iter)

        self.x = x # data; scalar or vector

        self.T_init = T # initial temp and temp to start with each temperature reset; scalar

        self.T = T
        
        self.T = T
        self.Tf = Tf # final temp; scalar
        self.total_iter = total_iter # no. of iterations; scalar
        self.delta = delta # step size; scalar


        self.reset_count = 0
        self.super_reset_count = 0
        self.cool_count = 0
        self.iteration = 0

        self.formula = self.funcs[self.name][0] # formula for the function
        self.bounds = self.funcs[self.name][1] # bounds for the function
        self.ans = self.funcs[self.name][2]


    ### Determine if correct datatype, converts to array
    def is_type(self):
        if type(self.x) in [list, tuple]:
            self.x = np.array(self.x) # convert to array
            return True
        elif type(self.x) in [float, int]:
            self.x = np.array([self.x]) # put scalar in list first, then array
            return False


    ### Temp Schedules

    # A simple geometric temperature schedule.
    def simple_geometric(self, temp):
        return temp*(1-0.01) # 0.01 is epsilon
    
    # logarithmic temp schedule
    def logarithmic(self, temp):
        return temp / np.log(2+self.iteration)


    # General Cooling Schedule Algorithm
    def cool(self, sched):
        self.previous = self.all_x[-2]
        self.current = self.all_x[-1]
        
        
        if self.T > self.Tf:
            self.T = sched(self.T)
        else:
            if np.isclose(self.previous, self.current).all(): # resets temperature 
                if (self.iteration == int(self.total_iter/3)) and self.super_reset_count < 2: # two super resets
                    if self.iteration == int(2*self.total_iter/3):
                        self.x = self.x + self.delta * np.random.uniform(-0.2 * self.bounds, 0.2 * self.bounds)
                    else:
                        self.x = self.x + self.delta * np.random.uniform(-0.2 * self.bounds, 0.2 * self.bounds)
                    self.super_reset_count += 1
                
                self.T_init = sched(self.T_init)
                if self.T_init > self.Tf:
                    self.T = self.T_init
                else:
                    pass
                self.reset_count += 1
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
            self.all_evals[self.iteration] = eval_y
        else:
            test = np.exp(-(eval_y - eval_x)/self.T)
            prob = np.random.uniform(0,1)
            if prob < test:
                self.x = y
                self.all_evals[self.iteration] = eval_y
            else:
                self.all_evals[self.iteration] = eval_x
        
        return self.x

    ### Main simulated annealing algorithm.

    def sim_ann(self, schedule):
        for i in range(self.total_iter):
            self.all_x[i] = np.round(self.opt_min()) # adds the new configuration to the list
            self.cool(schedule) # change the temperature according to the schedule
            self.iteration += 1

        ans = self.eval(self.all_x[-1])
        if np.isclose(ans, self.ans):
            return self.all_x, ans, True
        else:
            return self.all_x, ans, False


        