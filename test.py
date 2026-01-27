from simulated_annealing.sim_anneal import Optimize
from simulated_annealing.test_funcs import *
test = Optimize(3,100,0.01,10000 , 5, 8)

sa = test.sim_ann(rastrigin, test.simple_geo)
print(sa)