/*
 * Matrix Multiplication - CUDA Example with Shared Memory
 * 
 * Purpose: Demonstrates 2D thread indexing, shared memory optimization, and
 *          tiling techniques for high-performance matrix operations.
 * 
 * Why included: Matrix multiplication is fundamental to neural networks, image
 *                processing, and scientific computing. This example shows how
 *                shared memory reduces global memory access and improves performance.
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16 // Tile size for shared memory optimization

// GPU kernel: tiled matrix multiplication using shared memory
__global__ void matrixMul(float *A, float *B, float *C, int width) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index
    
    float sum = 0.0f;
    
    // Process tiles
    for (int tile = 0; tile < (width + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile from A into shared memory
        int aRow = row;
        int aCol = tile * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = (aRow < width && aCol < width) ? A[aRow * width + aCol] : 0.0f;
        
        // Load tile from B into shared memory
        int bRow = tile * TILE_SIZE + threadIdx.y;
        int bCol = col;
        tileB[threadIdx.y][threadIdx.x] = (bRow < width && bCol < width) ? B[bRow * width + bCol] : 0.0f;
        
        __syncthreads(); // Synchronize threads before computation
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads(); // Synchronize before next tile
    }
    
    // Write result to global memory
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// CPU reference implementation
void matrixMulCPU(float *A, float *B, float *C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
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
    const int width = 512; // Matrix dimension (width x width)
    size_t size = width * width * sizeof(float);
    
    // Host memory allocation
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);
    
    // Initialize matrices
    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Device memory allocation
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, size), "cudaMalloc d_C");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "H2D A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "H2D B");
    
    // Launch configuration (2D grid and blocks)
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (width + blockSize.y - 1) / blockSize.y);
    
    // CPU timing
    clock_t start_cpu = clock();
    matrixMulCPU(h_A, h_B, h_C_ref, width);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    
    // GPU timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Event create start");
    checkCudaError(cudaEventCreate(&stop), "Event create stop");
    
    checkCudaError(cudaEventRecord(start), "Event record start");
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    checkCudaError(cudaEventRecord(stop), "Event record stop");
    checkCudaError(cudaEventSynchronize(stop), "Event sync");
    
    float gpu_time = 0.0f;
    checkCudaError(cudaEventElapsedTime(&gpu_time, start, stop), "Event elapsed");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "D2H C");
    
    // Verify correctness
    int errors = 0;
    for (int i = 0; i < width * width; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3) errors++;
    }
    
    // Print results
    printf("Matrix Multiplication Results:\n");
    printf("  Matrix size: %dx%d\n", width, width);
    printf("  CPU time: %.3f ms\n", cpu_time);
    printf("  GPU time: %.3f ms\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Errors: %d\n", errors);
    printf("  Sample result: C[0][0] = %.2f (expected %.2f)\n", h_C[0], width * 2.0f);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}





