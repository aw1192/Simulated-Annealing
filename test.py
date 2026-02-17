import time
from simulated_annealing.sim_anneal import Optimize

t1 = time.perf_counter()
f = Optimize('ackley', [0,0], 10000, 0.01, 100000, 500)


res = f.sim_ann(f.simple_geometric)

print(res)

t2 = time.perf_counter()
print(f'Time elapsed: {t2-t1}')

print(f.reset_count)


