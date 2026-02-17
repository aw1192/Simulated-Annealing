import numpy as np

class Func:
    def __init__(self, name): 
        self.name = name   # name of function
        
        # dictionary of tester functions
        # first element is function formula, second is (positive) bound
        self.funcs = {'rastrigin': (lambda x: 10 * self.d + np.sum(x**2-10*np.cos(2* np.pi*x)),
                                            5.12),
                               'eggholder': (lambda x: -(x[1] + 47)* np.sin(np.sqrt(abs(x[1]+(x[0]/2) +47)))-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47)))),
                                             512),
                               'ackley': (), 
                               'shubert': ()} 
    
    def eval(self, x):
        try:
            func_info = self.funcs[self.name] # contains (formula, bounds)
        except KeyError:
            return 'Key not found.'
        
        # Keep all x values within bounds
        for i, v in enumerate(x):
            if v > func_info[1]: # check if x is beyond positive bounds
                x[i] = func_info[1]
            elif v < -func_info[1]: # check if x is beyond negative bounds
                x[i] = -func_info[1]
            else:
                pass
        
        # Evaluate function at the current configuration
        try:
            return func_info[0](x)
        except:
            raise Exception('Function evaluation failed.')