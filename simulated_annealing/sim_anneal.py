import numpy as np
from simulated_annealing.test_funcs import Func
from scipy.stats import cauchy

class Optimize(Func):
    def __init__(self, name, x, T, Tf, total_iter, delta, local_iter):
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
        self.local_iter = local_iter
        self.delta = delta # step size; scalar

        self.gamma = 5


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
                if (self.iteration % int(self.total_iter/3) == 0): # two super resets
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
        if self.iteration % (self.total_iter* 0.2) == 0:
            self.gamma = self.logarithmic(self.gamma)
        try:
            return x + self.delta * (cauchy.rvs(scale = self.gamma))
        except:
            raise Exception('Error: proposal configuration calculation.')

    ### Optimizer for minimum
    def opt_min(self):

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


    def local_search(self): # only descend from 5 best configurations
        
        self.gamma = 5
        configs = self.closest[:,:-1]

        all_eval_local = []
        for x in configs:
            print(x)
            x_local_configs = [x]
            for i in range(self.local_iter):
                y = np.apply_along_axis(self.proposal, 0, x_local_configs[i])
                
                eval_y = self.eval(y) 
                eval_x = self.eval(x)

                if eval_y <= eval_x:
                    x = y
                    x_local_configs.append(x)
                    #print(x)
                else:
                    x_local_configs.append(x)
                    #print(x)
            
            print(x)
            all_eval_local.append(np.min(x[-1]))
        print(all_eval_local)
        return all_eval_local
            


    ### Main simulated annealing algorithm.

    def sim_ann(self, schedule):
        self.is_type()
        for i in range(self.total_iter):
            self.all_x[i] = self.opt_min() # adds the new configuration to the list
            self.cool(schedule) # change the temperature according to the schedule
            self.iteration += 1

        

        self.uniques = np.unique(np.concatenate((self.all_x, np.expand_dims(self.all_evals, 0).T), axis = 1), axis = 0)
        self.closest = []
        distance = np.array(abs(self.uniques.T[-1] - self.ans))
        self.best_5 = np.sort(distance)[:5] # 5 best evaluations

        for i in self.uniques:
            compare = abs(i[-1] - self.ans)
            if compare in self.best_5:
                self.closest.append(i) # gives the 5 best configs and their evaluation

        self.closest = np.array(self.closest)
        # perform local search on 

        ans = self.local_search()

        index = np.argmin(np.min(np.abs(ans)))
        if np.isclose(np.min(np.abs(ans)), self.ans, atol=0.01):
            return self.all_x, ans[index], True
        else:
            return self.all_x, np.min(ans), False


# record best score; save several (done)
# data on how quick it converge
# cauchy dist proposal, gamma determines temperatures (done)
# proposals on individual components
# temps on proposals (done)
# function testing parameters
