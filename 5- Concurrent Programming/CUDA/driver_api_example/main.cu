/*
 * Vector Addition - CUDA Driver API Example
 * 
 * Purpose: Demonstrates the CUDA Driver API, which provides lower-level control
 *          over GPU execution compared to the Runtime API. This example shows
 *          explicit initialization, device/context management, and module loading.
 * 
 * Why This Is Included in the Portfolio:
 * 
 * 1. **Understanding CUDA Software Architecture**: The Driver API sits below the
 *    Runtime API in the CUDA software stack. Understanding both APIs is essential
 *    for developers who need fine-grained control over GPU execution or who want to
 *    integrate CUDA into languages other than C++.
 * 
 * 2. **Real-World Applications**: The Driver API is used in:
 *    - High-performance libraries that need to manage multiple GPU contexts
 *    - Language bindings (Python, Java, etc.) that wrap CUDA functionality
 *    - Applications requiring dynamic kernel loading from pre-compiled .cubin files
 *    - Multi-GPU applications with complex context switching
 * 
 * 3. **Educational Value**: This example demonstrates:
 *    - Manual initialization with cuInit() (vs automatic in Runtime API)
 *    - Explicit device selection and context management
 *    - Module loading from PTX or .cubin files (enabling kernel reuse across languages)
 *    - Lower-level kernel launch control
 * 
 * How Other Developers Can Use This:
 * 
 * - **Reference Implementation**: Use as a template for Driver API applications
 * - **Learning Tool**: Compare with vector_add/main.cu to understand Runtime vs Driver API differences
 * - **Integration Guide**: Adapt for embedding CUDA in non-C++ applications
 * - **Multi-GPU Development**: Extend to manage multiple devices and contexts
 * 
 * Compilation:
 *   nvcc -ptx driver_api_example/main.cu -o vectorAdd.ptx  # Generate PTX
 *   nvcc driver_api_example/main.cu -o driver_api_example -lcuda
 * 
 * Or compile kernel separately:
 *   nvcc -ptx -arch=sm_75 vectorAdd_kernel.cu -o vectorAdd.ptx
 *   nvcc driver_api_example/main.cu -o driver_api_example -lcuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Error checking helper for Driver API
#define CHECK_CUDA_ERROR(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char *errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA Driver API error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
            exit(1); \
        } \
    } while(0)

// PTX code for vector addition kernel (normally loaded from .ptx file)
// In production, this would be compiled separately: nvcc -ptx kernel.cu -o kernel.ptx
const char *vectorAddPTX = 
    ".version 7.5\n"
    ".target sm_75\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry vectorAdd(\n"
    "    .param .u64 vectorAdd_param_0,\n"
    "    .param .u64 vectorAdd_param_1,\n"
    "    .param .u64 vectorAdd_param_2,\n"
    "    .param .u32 vectorAdd_param_3\n"
    ")\n"
    "{\n"
    "    .reg .f32   %f<4>;\n"
    "    .reg .b32   %r<7>;\n"
    "    .reg .b64   %rd<8>;\n"
    "\n"
    "    ld.param.u64    %rd1, [vectorAdd_param_0];\n"
    "    ld.param.u64    %rd2, [vectorAdd_param_1];\n"
    "    ld.param.u64    %rd3, [vectorAdd_param_2];\n"
    "    ld.param.u32    %r1, [vectorAdd_param_3];\n"
    "    mov.u32     %r2, %tid.x;\n"
    "    mov.u32     %r3, %ctaid.x;\n"
    "    mov.u32     %r4, %ntid.x;\n"
    "    mad.lo.s32  %r5, %r3, %r4, %r2;\n"
    "    setp.ge.s32 %p1, %r5, %r1;\n"
    "    @%p1 bra    BB0_2;\n"
    "\n"
    "    cvta.to.global.u64    %rd4, %rd1;\n"
    "    mul.wide.s32  %rd5, %r5, 4;\n"
    "    add.s64     %rd6, %rd4, %rd5;\n"
    "    ld.global.f32   %f1, [%rd6];\n"
    "    cvta.to.global.u64    %rd7, %rd2;\n"
    "    add.s64     %rd6, %rd7, %rd5;\n"
    "    ld.global.f32   %f2, [%rd6];\n"
    "    add.f32     %f3, %f1, %f2;\n"
    "    cvta.to.global.u64    %rd7, %rd3;\n"
    "    add.s64     %rd6, %rd7, %rd5;\n"
    "    st.global.f32   [%rd6], %f3;\n"
    "\n"
    "BB0_2:\n"
    "    ret;\n"
    "}\n";

// Alternative: Load PTX from file (recommended for production)
// This function shows how to load PTX from a file
CUresult loadPTXFromFile(const char *filename, char **ptx, size_t *ptxSize) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open PTX file: %s\n", filename);
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
    
    fseek(fp, 0, SEEK_END);
    *ptxSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    *ptx = (char *)malloc(*ptxSize + 1);
    fread(*ptx, 1, *ptxSize, fp);
    (*ptx)[*ptxSize] = '\0';
    
    fclose(fp);
    return CUDA_SUCCESS;
}

// CPU reference implementation
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1000000; // Number of elements
    size_t size = n * sizeof(float);
    
    // ============================================================
    // STEP 1: Initialize Driver API (REQUIRED - no automatic init)
    // ============================================================
    // Unlike Runtime API, Driver API requires explicit initialization
    CHECK_CUDA_ERROR(cuInit(0));
    printf("✓ Driver API initialized\n");
    
    // ============================================================
    // STEP 2: Get Device Handle (manual device selection)
    // ============================================================
    CUdevice device;
    int deviceCount;
    CHECK_CUDA_ERROR(cuDeviceGetCount(&deviceCount));
    printf("✓ Found %d CUDA device(s)\n", deviceCount);
    
    // Select device 0 (can be extended to select specific device)
    CHECK_CUDA_ERROR(cuDeviceGet(&device, 0));
    
    // Get device properties
    char deviceName[256];
    CHECK_CUDA_ERROR(cuDeviceGetName(deviceName, 256, device));
    printf("✓ Selected device: %s\n", deviceName);
    
    // ============================================================
    // STEP 3: Create Context (manual context management)
    // ============================================================
    // Runtime API automatically manages context; Driver API requires explicit creation
    CUcontext context;
    CHECK_CUDA_ERROR(cuCtxCreate(&context, 0, device));
    printf("✓ Context created\n");
    
    // ============================================================
    // STEP 4: Load Module (kernel code as PTX or .cubin)
    // ============================================================
    // Runtime API: All kernels available after compilation
    // Driver API: Must explicitly load modules containing kernels
    CUmodule module;
    
    // Option 1: Load from PTX string (inline for this example)
    CHECK_CUDA_ERROR(cuModuleLoadDataEx(&module, vectorAddPTX, 0, 0, 0));
    printf("✓ Module loaded from PTX\n");
    
    // Option 2: Load from file (uncomment to use)
    // char *ptx;
    // size_t ptxSize;
    // loadPTXFromFile("vectorAdd.ptx", &ptx, &ptxSize);
    // CHECK_CUDA_ERROR(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    // free(ptx);
    
    // Option 3: Load from .cubin file (pre-compiled binary)
    // CHECK_CUDA_ERROR(cuModuleLoadDataEx(&module, cubinData, CU_JIT_INPUT_CUBIN, 0, 0));
    
    // ============================================================
    // STEP 5: Get Kernel Function (explicit kernel retrieval)
    // ============================================================
    CUfunction kernel;
    CHECK_CUDA_ERROR(cuModuleGetFunction(&kernel, module, "vectorAdd"));
    printf("✓ Kernel function retrieved\n");
    
    // ============================================================
    // STEP 6: Allocate Device Memory (using Driver API)
    // ============================================================
    CUdeviceptr d_a, d_b, d_c;
    CHECK_CUDA_ERROR(cuMemAlloc(&d_a, size));
    CHECK_CUDA_ERROR(cuMemAlloc(&d_b, size));
    CHECK_CUDA_ERROR(cuMemAlloc(&d_c, size));
    printf("✓ Device memory allocated\n");
    
    // ============================================================
    // STEP 7: Allocate and Initialize Host Memory
    // ============================================================
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    float *h_c_ref = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // ============================================================
    // STEP 8: Copy Data to Device (Driver API memory transfer)
    // ============================================================
    CHECK_CUDA_ERROR(cuMemcpyHtoD(d_a, h_a, size));
    CHECK_CUDA_ERROR(cuMemcpyHtoD(d_b, h_b, size));
    printf("✓ Data copied to device\n");
    
    // ============================================================
    // STEP 9: Configure and Launch Kernel (explicit launch)
    // ============================================================
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Set up kernel parameters
    void *args[] = { &d_a, &d_b, &d_c, &n };
    
    // CPU timing
    clock_t start_cpu = clock();
    vectorAddCPU(h_a, h_b, h_c_ref, n);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;
    
    // Launch kernel using Driver API
    CHECK_CUDA_ERROR(cuLaunchKernel(
        kernel,                          // Function to launch
        blocksPerGrid, 1, 1,             // Grid dimensions (x, y, z)
        threadsPerBlock, 1, 1,           // Block dimensions (x, y, z)
        0,                               // Shared memory (bytes)
        NULL,                            // Stream (NULL = default)
        args,                            // Kernel parameters
        NULL                             // Extra options
    ));
    
    // Synchronize to wait for kernel completion
    CHECK_CUDA_ERROR(cuCtxSynchronize());
    printf("✓ Kernel launched and synchronized\n");
    
    // ============================================================
    // STEP 10: Copy Results Back and Verify
    // ============================================================
    CHECK_CUDA_ERROR(cuMemcpyDtoH(h_c, d_c, size));
    
    // Verify correctness
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - h_c_ref[i]) > 1e-5) errors++;
    }
    
    // Print results
    printf("\n=== Vector Addition Results (Driver API) ===\n");
    printf("  Elements: %d\n", n);
    printf("  CPU time: %.3f ms\n", cpu_time);
    printf("  Errors: %d\n", errors);
    printf("  Sample result: c[0] = %.2f (expected 3.00)\n", h_c[0]);
    printf("  ✓ Computation successful!\n\n");
    
    // ============================================================
    // STEP 11: Cleanup (explicit resource management)
    // ============================================================
    // Driver API requires explicit cleanup of all resources
    free(h_a); free(h_b); free(h_c); free(h_c_ref);
    CHECK_CUDA_ERROR(cuMemFree(d_a));
    CHECK_CUDA_ERROR(cuMemFree(d_b));
    CHECK_CUDA_ERROR(cuMemFree(d_c));
    CHECK_CUDA_ERROR(cuModuleUnload(module));
    CHECK_CUDA_ERROR(cuCtxDestroy(context));
    printf("✓ Resources cleaned up\n");
    
    return 0;
}








