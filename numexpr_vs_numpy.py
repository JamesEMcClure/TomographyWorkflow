import os
import numpy as np
import numexpr as ne
from time import time
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

def np_func_1(a, b):
    t0 = time()
    _ = 2*a + 3*b
    return time() - t0
    
def ne_func_1(a, b):
    t0 = time()
    ne.evaluate("2*a + 3*b")
    return time() - t0

def np_func_2(a, b):
    t0 = time()
    _ = 2*a + b**10
    return time() - t0
    
def ne_func_2(a, b):
    t0 = time()
    ne.evaluate("2*a + b**10")
    return time() - t0

def get_mean_time(func, a, b, loops=100):
    times = [func(a, b) for _ in range(loops)]
    return np.mean(times) * 1000, np.std(times) * 1000
    

if __name__ == '__main__':
    arr_size = int(1e6)
    
    size_str = '{:.1e}'.format(arr_size).replace('0','').replace('+','').replace('.','').upper()
        
    print('Working with array size: {:.1e}'.format(arr_size))
    print('Number of CPU cores available: {}'.format(cpu_count()))
    
    # os.environ["NUMEXPR_MAX_THREADS"] = "168"
    # os.environ["OMP_NUM_THREADS"] = "8"
    
    a = np.random.rand(arr_size)
    b = np.random.rand(arr_size)
    
    tp_1 = get_mean_time(np_func_1, a, b, loops=10)
    print('Numpy: 2*a + 3*b: {:.2f} +/- {:.2f} msec'.format(*tp_1))
    
    te_1 = get_mean_time(ne_func_1, a, b, loops=100)
    print('Numexpr: 2*a + 3*b: {:.2f} +/- {:.2f} msec'.format(*te_1))
    
    tp_2 = get_mean_time(np_func_2, a, b, loops=10)
    print('Numpy: 2*a + b**10: {:.2f} +/- {:.2f} msec'.format(*tp_2))
    
    te_2 = get_mean_time(ne_func_2, a, b, loops=100)
    print('Numexpr: 2*a + b**10: {:.2f} +/- {:.2f} msec'.format(*te_2))
    
    threads_list = range(1, 64)
    mean_times = list()
    for n_threads in threads_list:
        ne.set_num_threads(n_threads)
        mean_times.append(get_mean_time(ne_func_2, a, b, loops=100)[0])
        
    print('Numexpr: 2*a + b**10: for cores: 1-64')
    print(mean_times)
    
    fig, axis = plt.subplots()
    axis.plot(threads_list, mean_times, 'o-')
    axis.set_title('numexpr - 2*a + b**10: {} points'.format(size_str))
    axis.set_xlabel('Num CPU threads')
    axis.set_ylabel('Mean execution time (msec)')
    fig.tight_layout()
    fig.savefig('numexpr_vs_threads_' + size_str + '.png', dpi=300)
    
