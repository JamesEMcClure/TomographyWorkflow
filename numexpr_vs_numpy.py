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

def print_context(arr_size):
    print('Environment variables:')
    for var_name in ['OMP_NUM_THREADS', 'NUMEXPR_MAX_THREADS']:
        print('{}: {}'.format(var_name, os.environ.get(var_name)))
    
    print('Number of CPU cores available: {}'.format(cpu_count()))
    print('numexpr using {} threads by default.\n'.format(ne.nthreads))
    print('Working with array size: {:.1e}'.format(arr_size))

def bench_ne_np(arr_size=None, show_context=True, show_results=True):
    
    if arr_size is not None:
        assert isinstance(arr_size, int)
    else:
        arr_size = int(1e6)
               
    size_str = '{:.1e}'.format(arr_size).replace('0','').replace('+','').replace('.','').upper()
    
    if show_context:
        print_context(arr_size)    
    
    a = np.random.rand(arr_size)
    b = np.random.rand(arr_size)
    
    tp_1 = get_mean_time(np_func_1, a, b, loops=10)
    te_1 = get_mean_time(ne_func_1, a, b, loops=100)      
    tp_2 = get_mean_time(np_func_2, a, b, loops=10)        
    te_2 = get_mean_time(ne_func_2, a, b, loops=100)
    
    if show_context:
        print('Numpy: 2*a + 3*b: {:.2f} +/- {:.2f} msec'.format(*tp_1))
        print('Numexpr: 2*a + 3*b: {:.2f} +/- {:.2f} msec'.format(*te_1))
        print('Numpy: 2*a + b**10: {:.2f} +/- {:.2f} msec'.format(*tp_2))
        print('Numexpr: 2*a + b**10: {:.2f} +/- {:.2f} msec'.format(*te_2))
        
    return tp_1, te_1, tp_2, te_2


def vary_used_threads(arr_size, func=ne_func_2, threads_list=None, show_plots=True):
    if arr_size is not None:
        assert isinstance(arr_size, int)
    else:
        arr_size = int(1e6)
        
    if threads_list is None:
        threads_list = [1, 2, 4, 8, 16]
        
    size_str = '{:.1e}'.format(arr_size).replace('0','').replace('+','').replace('.','').upper()
    
    a = np.random.rand(arr_size)
    b = np.random.rand(arr_size)
    
    mean_times = list()
    for n_threads in threads_list:
        ne.set_num_threads(n_threads)
        te_2 = get_mean_time(func, a, b, loops=100)
        print('\tusing {:3d} threads: {:.2f} +/- {:.2f} msec'.format(ne.nthreads, *te_2))
        mean_times.append(te_2[0])
        
    # print('Numexpr: 2*a + b**10: for cores: 1-64')
    # print(mean_times)
    
    if show_plots:
        fig, axis = plt.subplots()
        axis.plot(threads_list, mean_times, 'o-')
        axis.set_title('numexpr - 2*a + b**10: {} points'.format(size_str))
        axis.set_xlabel('Num CPU threads')
        axis.set_ylabel('Mean execution time (msec)')
        fig.tight_layout()
        fig.savefig('numexpr_vs_threads_' + size_str + '.png', dpi=300)
    
    return threads_list, mean_times


def vary_omp_num_threads(thread_list, arr_size=None, show_plots=True):
    
    tp_1 = list()
    te_1 = list()
    tp_2 = list()
    te_2 = list()
    
    for this_threads in thread_list:
        os.environ['OMP_NUM_THREADS'] = str(this_threads)
        
        ret_vals = bench_ne_np(arr_size=arr_size, show_context=True, show_results=True)
        tp_1.append(ret_vals[0])
        te_1.append(ret_vals[1])
        tp_2.append(ret_vals[2])
        te_2.append(ret_vals[3])
        print('\n'*2)
    
    if show_plots:
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
        for axis, y_vals, label in zip(axes.flat, [tp_1, te_1, tp_2, te_2],
                                       ['numpy 2*a + 3*b', 'numexpr 2*a + 3*b', 'numpy 2*a + b**10', 'numexpr 2*a + b**10']):
            print(label)
            print(len(y_vals))
            print(y_vals)
            mean_vals = [item[0] for item in y_vals]
            err_vals = [item[1] for item in y_vals]
            axis.errorbar(thread_list, mean_vals, err_vals, marker='o', linestyle='-')
            axis.set_yscale('log')
            axis.set_xscale('log')
            axis.set_xlabel('OMP NUM THREADS')
            axis.set_ylabel('Execution time (msec)')
            axis.set_title(label)
            #axis.legend()
        fig.tight_layout()
        fig.savefig('numexpr_omp_num_threads.png', dpi=300)
    

if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = str(168) 
    vary_omp_num_threads([1, 2, 4 ,8, 16, 32, 64, 128], arr_size=int(1e7), show_plots=True)
    os.environ['OMP_NUM_THREADS'] = str(128)
    vary_used_threads(int(1e7), func=ne_func_2, threads_list=[1, 2, 4 ,8, 16, 32, 64, 128], show_plots=True)
