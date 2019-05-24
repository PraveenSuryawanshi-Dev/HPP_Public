
import os


#numpy imports
import numpy as np
#


#cuda imports
import pycuda.driver as cuda_driver
import pycuda.autoinit
from pycuda.compiler import SourceModule # used for cuda kernel code compilation
#


#create random data on host
h_list_a = np.random.randn(5)

h_list_b = np.random.randn(5)

#

# as above statement create double precision data and  nvidia supports only single precious of data so convert it.
h_list_a = h_list_a.astype(np.float32)

h_list_b = h_list_b.astype(np.float32)

h_list_out = np.empty_like(h_list_a)

#

#pass this data from host to device

#step 1: alloc data on device first
d_list_a = cuda_driver.mem_alloc(h_list_a.nbytes)

d_list_b = cuda_driver.mem_alloc(h_list_b.nbytes)

d_list_out = cuda_driver.mem_alloc(h_list_b.nbytes)
#



#step 2: send data to alloced device
cuda_driver.memcpy_htod(d_list_a,h_list_a)

cuda_driver.memcpy_htod(d_list_b,h_list_b)
#

#




#write cuda kernel and compile
kernel_module = SourceModule("""
__global__ void list_add(float *a, float *b, float *out)
{
    int index = threadIdx.x;
    out[index] = a[index] + b[index];
}
""")
#

#get reference of function
list_add_kernel = kernel_module.get_function("list_add")
#

#call the kernel
list_add_kernel(d_list_a,d_list_b,d_list_out,block=(len(h_list_a),1,1))
#

#fetch data from gpu back
cuda_driver.memcpy_dtoh(h_list_out,d_list_out)
#


#display the result
print("\n A: ",h_list_a)
print("\n B: ",h_list_b)
print("\n R: ",h_list_out)