/*
 * Vector Addition Kernel - For Driver API Example
 * 
 * This kernel file is compiled separately to PTX format for loading into
 * the Driver API example. This demonstrates how kernels can be pre-compiled
 * and loaded dynamically.
 * 
 * Compile to PTX:
 *   nvcc -ptx -arch=sm_75 vectorAdd_kernel.cu -o vectorAdd.ptx
 * 
 * Or compile to .cubin (binary):
 *   nvcc -cubin -arch=sm_75 vectorAdd_kernel.cu -o vectorAdd.cubin
 */

// GPU kernel: parallel vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}











