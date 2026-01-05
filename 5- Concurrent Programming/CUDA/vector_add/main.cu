/*
 * Vector Addition - CUDA Example
 * 
 * Purpose: Demonstrates fundamental CUDA programming concepts including kernel
 *          execution, memory management, and parallel computation patterns.
 * 
 * Why included: This is the "Hello World" of CUDA programming, essential for
 *                understanding how threads map to data elements and basic GPU
 *                memory operations. Foundation for all parallel GPU algorithms.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// GPU kernel: parallel vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (idx < n) c[idx] = a[idx] + b[idx]; // Perform addition if within bounds
}

// CPU reference implementation for comparison
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Error checking helper
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const int n = 1000000; // Number of elements
    size_t size = n * sizeof(float);
    
    // Host memory allocation
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_c_ref = (float *)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Device memory allocation
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c, size), "cudaMalloc d_c");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "H2D a");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "H2D b");
    
    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // CPU timing
    clock_t start_cpu = clock();
    vectorAddCPU(h_a, h_b, h_c_ref, n);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    
    // GPU timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Event create start");
    checkCudaError(cudaEventCreate(&stop), "Event create stop");
    
    checkCudaError(cudaEventRecord(start), "Event record start");
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    checkCudaError(cudaEventRecord(stop), "Event record stop");
    checkCudaError(cudaEventSynchronize(stop), "Event sync");
    
    float gpu_time = 0.0f;
    checkCudaError(cudaEventElapsedTime(&gpu_time, start, stop), "Event elapsed");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "D2H c");
    
    // Verify correctness
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-5) errors++;
    }
    
    // Print results
    printf("Vector Addition Results:\n");
    printf("  Elements: %d\n", n);
    printf("  CPU time: %.3f ms\n", cpu_time);
    printf("  GPU time: %.3f ms\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Errors: %d\n", errors);
    printf("  Sample result: c[0] = %.2f (expected 3.00)\n", h_c[0]);
    
    // Cleanup
    free(h_a); free(h_b); free(h_c); free(h_c_ref);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}





