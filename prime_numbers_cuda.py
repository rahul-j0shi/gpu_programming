import numpy as np
import cupy
import math
from cupyx.profiler import benchmark

# CPU version

def all_primes_to(upper: int, prime_list: list):
    for num in range(0,upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if num % i == 0:
                prime = False
                break
        if prime:
            prime_list[num]=1
    
upper_bound = 100_000
all_primes_cpu = np.zeros(upper_bound, dtype=np.int32)


# GPU verison

check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
'''


# allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# setup the grid

all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, 'all_primes_to')
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)

# benchmark and test

import timeit

cpu_time = timeit.timeit(lambda: all_primes_to(upper_bound, all_primes_cpu), number=10)
print(f"Average CPU execution time: {cpu_time / 10:.6f} seconds")
execution_gpu = all_primes_to_gpu(grid_size, block_size, (upper_bound, all_primes_gpu), n_repeats=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"Average GPU execution time: {gpu_avg_time * 1e-6:.2f} ms")

if np.allclose(all_primes_cpu, all_primes_gpu):
    print("correct")
else:
    print("wrong")


