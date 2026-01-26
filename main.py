import timeit
test = '''
from simulated_annealing import sim_anneal as sa
from simulated_annealing.functions.rastrigin import rastrigin as rast


sa.sim_anneal(4,10,0.01,rast, 2,300)
'''
t= timeit.timeit(test, number= 1000)
print(t)


