import time
from simulated_annealing.sim_anneal import Optimize

t1 = time.perf_counter()
f = Optimize('rastrigin', [10,2,5675,67,1], 1000, 0.01, 100000, 3)


print(f.sim_ann(f.simple_geometric))

t2 = time.perf_counter()
print(f'Time elapsed: {t2-t1}')


