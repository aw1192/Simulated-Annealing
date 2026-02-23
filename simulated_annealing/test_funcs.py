import numpy as np

class Func:
    def __init__(self, name): 
        self.name = name   # name of function
        
        # dictionary of tester functions
        # first element is function formula, second is (positive) bound, 3rd is global minimum
        self.funcs = {'rastrigin': (lambda x: 10 * self.d + np.sum(x**2-10*np.cos(2* np.pi*x)),
                                            5.12, 0),
                               'eggholder': (lambda x: -(x[1] + 47)* np.sin(np.sqrt(abs(x[1]+(x[0]/2) +47)))-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47)))),
                                             512, -959.6407),
                               'ackley': (lambda x: -20*np.exp(-0.2 * np.sqrt((1/self.d)* np.sum((x)**2)))-np.exp((1/self.d)*np.sum(np.cos(2*np.pi*x)))+20+np.exp(1),
                                          32.768, 0), 
                               'holder table': (lambda x: -abs(np.sin(x[0])*np.cos(x[1])*np.exp(abs(1- (np.sqrt(x[0]**2+x[1]**2))/np.pi))),
                                                10, -19.2085),
                                'levy function 13': (lambda x: (np.sin(3*np.pi*x[0]))**2+(x[0]-1)**2 * (1+(np.sin(3*np.pi*x[1]))**2)+(x[1]-1)**2 * (1+(np.sin(2*np.pi*x[1]))**2),
                                                     10, 0)                
                                } 
    
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