# CUDA Driver API Example

This example demonstrates the **CUDA Driver API**, which provides lower-level control over GPU execution compared to the Runtime API.

## Why This Example Is Included

### 1. **Understanding CUDA Software Architecture**
The Driver API sits below the Runtime API in the CUDA software stack. Understanding both APIs is essential for:
- Developers who need fine-grained control over GPU execution
- Those integrating CUDA into languages other than C++ (Python, Java, etc.)
- Building high-performance libraries with complex GPU management

### 2. **Real-World Applications**
The Driver API is used in:
- **Language Bindings**: Python (PyCUDA, CuPy), Java (JCuda), and other language wrappers
- **Multi-GPU Applications**: Complex context switching and device management
- **Dynamic Kernel Loading**: Loading pre-compiled `.cubin` files at runtime
- **High-Performance Libraries**: Managing multiple GPU contexts efficiently

### 3. **Key Differences from Runtime API**

| Feature | Runtime API | Driver API |
|---------|-------------|------------|
| Initialization | Automatic | Manual (`cuInit()`) |
| Context Management | Automatic | Manual (`cuCtxCreate()`) |
| Module Loading | All kernels available | Explicit (`cuModuleLoad()`) |
| Language Support | C++ only | Any language that can link `.cubin` |
| Control Level | High-level abstraction | Low-level control |

## How to Use This Example

### Prerequisites
- NVIDIA GPU with compute capability 3.0+
- CUDA Toolkit installed
- `nvcc` compiler in PATH
- Driver API library (`libcuda.so` on Linux, `cuda.lib` on Windows)

### Compilation Methods

#### Method 1: Using Inline PTX (Current Implementation)
The `main.cu` file includes inline PTX code for simplicity:

```bash
nvcc driver_api_example/main.cu -o driver_api_example -lcuda
./driver_api_example
```

#### Method 2: Compile Kernel Separately (Recommended for Production)

1. **Compile kernel to PTX:**
   ```bash
   nvcc -ptx -arch=sm_75 driver_api_example/vectorAdd_kernel.cu -o vectorAdd.ptx
   ```

2. **Modify main.cu** to load from file:
   ```cpp
   // Replace inline PTX loading with:
   char *ptx;
   size_t ptxSize;
   loadPTXFromFile("vectorAdd.ptx", &ptx, &ptxSize);
   CHECK_CUDA_ERROR(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
   ```

3. **Compile and run:**
   ```bash
   nvcc driver_api_example/main.cu -o driver_api_example -lcuda
   ./driver_api_example
   ```

#### Method 3: Compile to .cubin (Binary Format)

1. **Compile kernel to .cubin:**
   ```bash
   nvcc -cubin -arch=sm_75 driver_api_example/vectorAdd_kernel.cu -o vectorAdd.cubin
   ```

2. **Load .cubin in code:**
   ```cpp
   // Read .cubin file into memory, then:
   CHECK_CUDA_ERROR(cuModuleLoadDataEx(&module, cubinData, CU_JIT_INPUT_CUBIN, 0, 0));
   ```

### Finding Your GPU Architecture

To determine the correct `-arch` flag:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# Or check CUDA samples deviceQuery output
```

Common architectures:
- `sm_75` - Turing (RTX 20xx, GTX 16xx)
- `sm_80` - Ampere (RTX 30xx, A100)
- `sm_86` - Ampere (RTX 30xx mobile)
- `sm_89` - Ada Lovelace (RTX 40xx)

## What This Example Demonstrates

1. **`cuInit(0)`** - Explicit Driver API initialization
2. **`cuDeviceGet()`** - Manual device selection
3. **`cuCtxCreate()`** - Context creation and management
4. **`cuModuleLoadDataEx()`** - Loading kernel modules from PTX or .cubin
5. **`cuModuleGetFunction()`** - Retrieving kernel function handles
6. **`cuMemAlloc()`** - Device memory allocation
7. **`cuMemcpyHtoD()` / `cuMemcpyDtoH()`** - Host-device memory transfers
8. **`cuLaunchKernel()`** - Explicit kernel launch with parameter setup
9. **`cuCtxSynchronize()`** - Synchronization
10. **Resource cleanup** - Explicit destruction of contexts, modules, and memory

## Comparison with Runtime API Example

Compare this example with `../vector_add/main.cu` to see the differences:

- **Runtime API**: Automatic initialization, simpler syntax, C++ only
- **Driver API**: Manual control, more verbose, language-agnostic

## Use Cases for Driver API

- **Embedding in Other Languages**: Load pre-compiled `.cubin` files from Python, Java, etc.
- **Dynamic Kernel Loading**: Load different kernels based on runtime conditions
- **Multi-Context Management**: Create and switch between multiple GPU contexts
- **Library Development**: Build CUDA libraries that need fine-grained control

## References

- [CUDA Driver API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/)
- [CUDA Programming Guide - Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#driver-api)
- Compare with Runtime API examples in `../vector_add/`, `../matrix_multiply/`, `../reduction/`






