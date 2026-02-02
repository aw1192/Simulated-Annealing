import timeit


a = '''
from simulated_annealing.sim_anneal import Optimize
from simulated_annealing.test_funcs import rastrigin


test = Optimize(5,100,0.01,10000 , 5, 10)

sa = test.sim_ann(rastrigin, test.simple_geo)
print(sa)'''

t = timeit.timeit(a,number=10)

# egg = Optimize([100, 200], 100, 0.01, 10000, 5, 10)

# sa2 = egg.sim_ann(eggholder, test.simple_geo)
#print(sa2)