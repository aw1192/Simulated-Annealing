import numpy as np

class Optimize:
    def __init__(self, x, T, Tf, total_iter, therm_iter, delta):
        self.x = x
        self.T = T
        self.Tf = Tf
        self.total_iter = total_iter
        self.therm_iter = therm_iter
        self.delta = delta

    ### Different Temp Schedules
    def simple_geo(self):
        self.T = self.T*(1-0.01) # 0.01 is epsilon
    
    ### thermalization 

    def therm_min(self, func):
        for i in range(self.therm_iter):
            y = self.x + self.delta * (np.random.uniform(-1,1)) # perturbation
            eval_y = func(y)
            eval_x = func(self.x)

            if eval_y <= eval_x:
                self.x = y
            else:
                prob = np.random.uniform(0,1)
                test = np.exp(-(eval_y - eval_x)/self.T)
                if prob < test:
                    self.x = y
                else:
                    pass
        return self.x
    
    ### simulated annealing

    def sim_ann(self, func, schedule):
        all_x = np.empty(self.total_iter)
        for i in range(self.total_iter):
            all_x[i] = round(self.therm_min(func), 2)
            schedule()
        return all_x
