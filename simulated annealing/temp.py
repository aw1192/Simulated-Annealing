

def geometric(T0, Tf, curr_iter, total_iter = 100):
    T_sched = T0 * (Tf/T0)**((curr_iter-1)/(total_iter - 1))
    curr_iter += 1

    return T_sched, curr_iter
    