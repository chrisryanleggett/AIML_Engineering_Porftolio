# CUDA Programming Examples

This directory demonstrates fundamental CUDA programming concepts through practical examples. Each example showcases essential GPU programming patterns including kernel execution, memory management, and parallel computation.

## Why This Matters

Understanding CUDA is critical for:

- **High-performance computing** - Leveraging GPU parallelism for computationally intensive tasks
- **AI/ML acceleration** - Training neural networks and processing large datasets
- **Scientific computing** - Simulations, data analysis, and numerical methods
- **Real-time processing** - Graphics, video encoding, signal processing

## Structure

Each example follows CUDA best practices:

- **Host code (CPU)** - Memory allocation, data transfer, kernel launches, error handling
- **Device code (GPU)** - Parallel kernels marked with `__global__`
- **CUDA runtime API** - Proper memory management and error checking

## Examples Included

### 1. [Vector Addition](./vector_add/main.cu)

**Purpose**: Demonstrates basic parallel computation, memory transfer, and kernel launch syntax.

**Key Concepts**:
- Thread indexing (`threadIdx.x`, `blockIdx.x`)
- Grid/block configuration
- `cudaMalloc`/`cudaMemcpy` for GPU memory management
- Kernel launch syntax `<<<blocks, threads>>>`
- Error handling with `cudaError_t`

**Use Case**: Foundation for understanding how threads map to data elements in parallel operations.

### 2. [Matrix Multiplication](./matrix_multiply/main.cu)

**Purpose**: Demonstrates 2D thread indexing, shared memory optimization, and tiling for performance.

**Key Concepts**:
- 2D grid/block dimensions
- Shared memory for performance optimization
- Memory coalescing patterns
- Computational complexity reduction through tiling

**Use Case**: Essential pattern for neural network operations, image processing, and scientific computing.

### 3. [Reduction (Sum)](./reduction/main.cu)

**Purpose**: Demonstrates parallel reduction pattern, warp-level primitives, and atomic operations.

**Key Concepts**:
- Shared memory usage
- Thread synchronization (`__syncthreads()`)
- Reduction pattern (tree-based aggregation)
- Warp divergence avoidance

**Use Case**: Common pattern for aggregating results (sum, max, min) across large datasets.

### 4. [Driver API Example](./driver_api_example/main.cu)

**Purpose**: Demonstrates the CUDA Driver API, providing lower-level control over GPU execution compared to the Runtime API.

**Key Concepts**:
- Explicit initialization with `cuInit()`
- Manual device selection with `cuDeviceGet()`
- Context creation and management with `cuCtxCreate()`
- Module loading from PTX or `.cubin` files with `cuModuleLoadDataEx()`
- Kernel function retrieval with `cuModuleGetFunction()`
- Explicit kernel launch with `cuLaunchKernel()`

**Use Case**: Essential for understanding CUDA software architecture, integrating CUDA into non-C++ languages, and applications requiring fine-grained GPU control.

**Why Included**: This example demonstrates the lower-level Driver API that sits beneath the Runtime API. Understanding both APIs is crucial for developers working with language bindings (Python, Java), multi-GPU applications, or dynamic kernel loading scenarios.

## Runtime API vs Driver API

The CUDA software stack provides two APIs for GPU programming, each with distinct advantages:

### Runtime API (Examples 1-3: vector_add, matrix_multiply, reduction)

The **Runtime API** is a higher-level abstraction that simplifies GPU programming:

- **Automatic Initialization**: No need to call `cuInit()` - initialization happens automatically
- **Simplified Context Management**: Contexts are created and managed automatically
- **All Kernels Available**: After compilation, all kernels in your code are immediately available
- **Requires C++ Compilation**: Code must be compiled with `nvcc` as C++ code
- **Simpler Syntax**: Cleaner, more intuitive API calls (`cudaMalloc`, `cudaMemcpy`, etc.)

**Best For**: Most CUDA applications, especially when developing in C++ and when simplicity is preferred.

**Example Pattern**:
```cpp
// Automatic initialization - no setup needed
cudaMalloc(&d_a, size);                    // Allocate device memory
cudaMemcpy(d_a, h_a, size, ...);          // Transfer data
vectorAdd<<<blocks, threads>>>(...);      // Launch kernel
cudaMemcpy(h_c, d_c, size, ...);          // Copy results back
```

### Driver API (Example 4: driver_api_example)

The **Driver API** provides lower-level control and flexibility:

- **Manual Initialization**: Must explicitly call `cuInit()` to initialize the driver
- **Explicit Context Management**: You create and manage contexts with `cuCtxCreate()`
- **Module Loading Required**: Kernels must be loaded from PTX or `.cubin` files using `cuModuleLoadDataEx()`
- **Language Agnostic**: Can be used from any language that can link `.cubin` objects (Python, Java, C, etc.)
- **More Control**: Fine-grained control over devices, contexts, and kernel execution

**Best For**: 
- Language bindings (PyCUDA, JCuda, etc.)
- Multi-GPU applications with complex context switching
- Dynamic kernel loading from pre-compiled binaries
- Applications requiring explicit resource management

**Example Pattern**:
```cpp
cuInit(0);                                 // Explicit initialization
cuDeviceGet(&device, 0);                   // Get device handle
cuCtxCreate(&context, 0, device);           // Create context
cuModuleLoadDataEx(&module, ptx, ...);      // Load module
cuModuleGetFunction(&kernel, module, ...);  // Get kernel function
cuMemAlloc(&d_a, size);                     // Allocate device memory
cuLaunchKernel(kernel, ...);                // Launch kernel
cuCtxDestroy(context);                      // Cleanup context
```

### When to Use Which API?

| Scenario | Recommended API |
|----------|----------------|
| Standard C++ CUDA development | **Runtime API** |
| Python/Java/other language bindings | **Driver API** |
| Simple GPU applications | **Runtime API** |
| Multi-GPU with context switching | **Driver API** |
| Dynamic kernel loading | **Driver API** |
| Learning CUDA fundamentals | **Runtime API** (start here) |
| Building high-performance libraries | **Driver API** |

### Understanding the CUDA Software Stack

```
┌─────────────────────────────────────┐
│   Your Application Code             │
├─────────────────────────────────────┤
│   Runtime API (cudaMalloc, etc.)   │  ← Examples 1-3
│   OR                                │
│   Driver API (cuInit, cuCtx, etc.) │  ← Example 4
├─────────────────────────────────────┤
│   CUDA Driver                       │
├─────────────────────────────────────┤
│   GPU Hardware                      │
└─────────────────────────────────────┘
```

The Runtime API is built on top of the Driver API, providing a simpler interface while the Driver API offers direct access to lower-level functionality.

## Compilation & Execution

### Prerequisites

- NVIDIA GPU with compute capability 3.0+
- CUDA Toolkit installed
- `nvcc` compiler in PATH

### Local Compilation

```bash
# Compile
nvcc main.cu -o program

# Run
./program

# With specific GPU architecture
nvcc -arch=sm_75 main.cu -o program

# Check CUDA installation
nvcc --version

# List available GPUs
nvidia-smi
```

### Google Colab

1. Enable GPU runtime: Runtime → Change runtime type → GPU
2. Install CUDA toolkit:
   ```python
   !apt-get install nvcc
   ```
3. Compile and run:
   ```python
   !nvcc /path/to/main.cu -o program && !./program
   ```

### Cloud GPU (AWS EC2, Google Cloud, Azure)

1. Launch GPU instance (e.g., g4dn.xlarge on AWS)
2. Install CUDA Toolkit:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda
   ```
3. Compile and execute as shown in Local Compilation section

## Performance Comparison

Each example includes CPU vs GPU timing to demonstrate speedup:

```
Vector Addition (1M elements):
  CPU: 2.34ms
  GPU: 0.18ms
  Speedup: 13x

Matrix Multiplication (1024x1024):
  CPU: 2847ms
  GPU: 12ms
  Speedup: 237x

Reduction (10M elements):
  CPU: 45ms
  GPU: 0.8ms
  Speedup: 56x
```

## Demonstrated Key Concepts

✅ CUDA C/C++ programming  
✅ GPU memory management (global, shared, registers)  
✅ Parallel algorithm design  
✅ Performance optimization techniques  
✅ Thread hierarchy understanding (grid → block → thread)  
✅ Error handling and debugging  
✅ Runtime API vs Driver API understanding  
✅ CUDA software architecture layers  

## Hardware Used

- **Development**: Google Colab (Tesla T4)
- **Testing**: Local GPU with compute capability 7.5+

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)



