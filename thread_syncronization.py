import math
import numpy as np
import cupy
from cupyx.profiler import benchmark
import timeit

def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] - output_array[item] + 1

size = 2**25

input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

histogram_cude_code = r'''
extern "C"
__global__ void histogram(const int *input, int *output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];

    // Initialize shared memory  and synchronize threads
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // compute shared memory histogram and synchronize threads
    atomicAdd(&(temp_histogram[input[item]]), 1);   
    __syncthreads();

    // update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and set up the cuda code
histogram_gpu = cupy.RawKernel(histogram_cude_code, 'histogram')
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)


# check correctness
histogram(input_cpu, output_cpu)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))

if np.allclose(output_cpu, output_gpu):
    print("correct results")
else:
    print("incorrect results")

# measure performance
cpu_time = timeit.timeit(lambda: histogram(input_cpu, output_cpu), number=1)
print(f"{cpu_time:.6f} s")
execution_gpu = benchmark(histogram_gpu, (grid_size, block_size, (input_gpu, output_gpu)), n_repeat = 10)
gpu_avg_time = np.average(execution_gpu.times)
print(f"{gpu_avg_time:.6f} s")