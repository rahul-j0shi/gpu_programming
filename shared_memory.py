import math
import numpy as np
import cupy

size = 2048
# gpu memory allocation
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)


# cpu memory allocation
a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu) 
c_cpu = np.zeros(size, dtype=np.float32)

# CUDA code

vector_add_cuda_code = r''' 
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  int offset = threadIdx.x * 3;
  extern __shared__ float temp[];

  if ( item < size )
  {
      temp[offset + 0] = A[item];
      temp[offset + 1] = B[item];
      temp[offset + 2] = temp[offset + 0] + temp[offset + 1];
      C[item] = temp[offset + 2];
  }
}
'''

vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, 'vector_add')
threads_per_block = 32
grid_size = (int(math.ceil(size / threads_per_block)), 1,1)
block_size = (threads_per_block, 1, 1)
vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size), shared_mem=threads_per_block * 3 * cupy.dtype(cupy.float32).itemsize)

# execution

from vector_addition_cuda import vector_add
vector_add(a_cpu, b_cpu, c_cpu, size)
np.allclose(c_cpu, c_gpu)  # This will return True if the results match

