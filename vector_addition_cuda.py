
""" 
Summing two vectors using in python

def vector_add(A,B,C,size):
    for item in range(0,size):
        C[item]=A[item]+B[item]

Summing two vectors using CUDA
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, int size)
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
"""

# running code on gpu with cupy

import cupy

size = 1024

a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

#cude vector_add
vector_add_cuda_code = r'''
extern "C" 
__global__ void vector_add(const float *A, const float *B, float *C, int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, 'vector_add')

vector_add_gpu((1,1,1), (size,1,1), (a_gpu, b_gpu, c_gpu, size))


# to be sure that the cuda code does exactly what we want, we can execute our sequential python code and compare the results
import numpy as np

def vector_add(A,B,C,size):
    for item in range(0,size):
        C[item]=A[item]+B[item]

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if np.allclose(c_cpu, c_gpu):
    print("The GPU and CPU results match!")



