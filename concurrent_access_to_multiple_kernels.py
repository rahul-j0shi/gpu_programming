import numpy as np
import cupy
import math
from cupyx.profiler import benchmark

upper_bound = 100_000
histogram_size = 2**25

check_prime_gpu_code = r'''
extern "C"
__global__ void check_prime(const int *input, int *output, int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (item < size) {
        int n = input[item];
        output[item] = 1;  // Assume prime until proven otherwise
        if (n < 2) output[item] = 0;
        for (int i = 2; i <= sqrt(n); i++) {
            if (n % i == 0) {
                output[item] = 0;
                break;
            }
        }
    }
}
'''

histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int *input, int *output, int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];

    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    atomicAdd(&temp_histogram[input[item]], 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&output[threadIdx.x], temp_histogram[threadIdx.x]);
}
'''

# memery allocation
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)
input_gpu = cupy.random.randint(0, 256, size=histogram_size, dtype=cupy.int32)
output_gpu = cupy.zeros(256, dtype=cupy.int32)

#  compile and set up the grid
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size_primes = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size_primes = (1024, 1, 1)
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block_hist = 256
grid_size_hist = (int(math.ceil(histogram_size / threads_per_block_hist)), 1, 1)
block_size_hist = (threads_per_block_hist, 1, 1)

# ececute the kernels
all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))


# results
output_one = all_primes_gpu
output_two = output_gpu

# create cuda streams
stream_one = cupy.cuda.Stream()
stream_two = cupy.cuda.Stream()

stream_one.synchronize()

sync_point = cupy.cuda.Event()

with stream_one:
    all_primes_to_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))
    sync_point.record(stream = stream_one)
    all_primes_gpu(grid_size_primes, block_size_primes, (upper_bound, all_primes_gpu))

with stream_two:
    stream_two.wait_event(sync_point)
    histogram_gpu(grid_size_hist, block_size_hist, (input_gpu, output_gpu))
