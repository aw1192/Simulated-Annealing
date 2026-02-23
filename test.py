import time
from simulated_annealing.sim_anneal import Optimize
from matplotlib import pyplot as plt
import numpy as np

t1 = time.perf_counter()
f = Optimize('rastrigin', [4,5], 1000, 0.01, 10000, 500, 10)


res = f.sim_ann(f.logarithmic)

print(res)

t2 = time.perf_counter()
print(f'Time elapsed: {t2-t1}')



plt.plot(np.arange(f.iteration), f.all_evals)
plt.title('Function Optimization')
plt.xlabel('Iterations')
plt.ylabel(f'{f.name.capitalize()} Function Evaluation')
plt.show()