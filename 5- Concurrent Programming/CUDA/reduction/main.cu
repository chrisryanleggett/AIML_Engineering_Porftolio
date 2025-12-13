/*
 * Parallel Reduction - CUDA Example
 * 
 * Purpose: Demonstrates parallel reduction pattern using shared memory,
 *          thread synchronization, and warp-level primitives for efficient
 *          aggregation operations (sum, max, min).
 * 
 * Why included: Reduction is a fundamental parallel pattern used in many
 *                algorithms (sum, dot product, finding max/min). This example
 *                shows how to efficiently aggregate results across thousands
 *                of threads using tree-based reduction.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 256

// GPU kernel: parallel reduction using shared memory
__global__ void reduce(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE]; // Shared memory for block-level reduction
    
    int tid = threadIdx.x; // Thread ID within block
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads(); // Synchronize all threads in block
    
    // Tree-based reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // Add elements
        }
        __syncthreads(); // Synchronize before next iteration
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// CPU reference implementation
float reduceCPU(float *input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

// Error checking helper
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const int n = 10000000; // Number of elements
    size_t size = n * sizeof(float);
    
    // Host memory allocation
    float *h_input = (float *)malloc(size);
    float *h_output;
    
    // Initialize input array
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    
    // Device memory allocation
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "cudaMalloc d_input");
    
    // Calculate number of blocks needed
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t outputSize = numBlocks * sizeof(float);
    checkCudaError(cudaMalloc(&d_output, outputSize), "cudaMalloc d_output");
    h_output = (float *)malloc(outputSize);
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "H2D input");
    
    // CPU timing
    clock_t start_cpu = clock();
    float cpu_result = reduceCPU(h_input, n);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    
    // GPU timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Event create start");
    checkCudaError(cudaEventCreate(&stop), "Event create stop");
    
    checkCudaError(cudaEventRecord(start), "Event record start");
    reduce<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, n);
    checkCudaError(cudaEventRecord(stop), "Event record stop");
    checkCudaError(cudaEventSynchronize(stop), "Event sync");
    
    float gpu_time = 0.0f;
    checkCudaError(cudaEventElapsedTime(&gpu_time, start, stop), "Event elapsed");
    
    // Copy partial results back to host
    checkCudaError(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost), "D2H output");
    
    // Final reduction on CPU (or launch another kernel for large reductions)
    float gpu_result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        gpu_result += h_output[i];
    }
    
    // Verify correctness
    float error = fabs(gpu_result - cpu_result);
    
    // Print results
    printf("Reduction Results:\n");
    printf("  Elements: %d\n", n);
    printf("  CPU result: %.2f\n", cpu_result);
    printf("  GPU result: %.2f\n", gpu_result);
    printf("  CPU time: %.3f ms\n", cpu_time);
    printf("  GPU time: %.3f ms\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Error: %.6f\n", error);
    
    // Cleanup
    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

