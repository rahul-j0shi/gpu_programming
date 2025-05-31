import numpy as np
import cupy
import math

size = 2048

a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

args = (a_gpu, b_gpu, c_gpu, size)

cuda_code = r'''
extern "C" {

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (item < size) {
        // Use the constant memory factor for multiplication
        C[item] = (A[item] + B[item]) * factors[blockIdx.x];
    }
}
}
'''

module = cupy.RawModule(code=cuda_code)
sum_and_multiply_gpu = module.get_function('sum_and_multiply')

factors_ptr = module.get_global('factors')
factors_gpu = cupy.ndarray(2,cupy.float32, factors_ptr)
factors_gpu[...] = cupy.random.random(2, dtype=cupy.float32)

sum_and_multiply_gpu((2,1,1), (size//2,1,1), args)

