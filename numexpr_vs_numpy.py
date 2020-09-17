import numpy as np
import numexpr as ne
from time import time

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
    a = np.random.rand(int(1e6))
    b = np.random.rand(int(1e6))
    
    tp_1 = get_mean_time(np_func_1, a, b, loops=10)
    print('Numpy: 2*a + 3*b: {:.2f} +/- {:.2f} msec'.format(*tp_1))
    
    te_1 = get_mean_time(ne_func_1, a, b, loops=100)
    print('Numexpr: 2*a + 3*b: {:.2f} +/- {:.2f} msec'.format(*te_1))
    
    tp_2 = get_mean_time(np_func_2, a, b, loops=10)
    print('Numpy: 2*a + b**10: {:.2f} +/- {:.2f} msec'.format(*tp_2))
    
    te_2 = get_mean_time(ne_func_2, a, b, loops=100)
    print('Numexpr: 2*a + b**10: {:.2f} +/- {:.2f} msec'.format(*te_2))
    
    mean_times = list()
    for n_threads in range(1, 64):
        ne.set_num_threads(n_threads)
        mean_times.append(get_mean_time(ne_func_2, a, b, loops=100)[0])
        
    print('Numexpr: 2*a + b**10: for cores: 1-64')
    print(mean_times)
