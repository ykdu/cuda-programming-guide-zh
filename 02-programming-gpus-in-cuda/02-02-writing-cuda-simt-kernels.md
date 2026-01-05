# 2.2 Writing CUDA SIMT Kernels

CUDA C++ kernels 在很大程度上可以按照针对特定问题编写传统 CPU code 的方式来编写。不过，GPU 具有一些独特的特性，可以用来提升性能。此外，理解 GPU 上的 threads 如何被 schedule、它们如何访问 memory，以及它们的执行是如何推进的，有助于开发者编写出能够最大化利用可用计算资源的 kernels。

### 2.2.1. Basics of SIMT

从开发者的角度来看，CUDA thread 是并行性的基本单元。Warps and SIMT 描述了 GPU 执行的基本 SIMT 模型，而 SIMT Execution Model 提供了 SIMT 模型的更多细节。SIMT 模型允许每个 thread 维护自己的状态和 control flow。从功能角度来看，每个 thread 都可以执行一条独立的 code path。然而，如果在编写 kernel code 时注意尽量减少同一个 warp 中的 threads 走向不同 code paths 的情况，就可以获得显著的性能提升。

### 2.2.2. Thread Hierarchy

Threads 被组织成 thread blocks，而 thread blocks 又被组织成一个 grid。Grid 可以是一维、二维或三维，其 size 可以在 kernel 内通过内建变量 gridDim 查询。Thread blocks 也可以是一维、二维或三维，其 size 可以在 kernel 内通过内建变量 blockDim 查询。Thread block 的 index 可以通过内建变量 blockIdx 查询。在一个 thread block 内，thread 的 index 通过内建变量 threadIdx 获取。这些内建变量用于为每个 thread 计算唯一的 global thread index，从而使每个 thread 能够按需从 global memory 中 load/store 特定的数据，并执行一条唯一的 code path。

- `gridDim.{x|y|z}`：grid 在 x、y 和 z 维度上的 size。这些值在 kernel launch 时设置。
- `blockDim.{x|y|z}`：block 在 x、y 和 z 维度上的 size。这些值在 kernel launch 时设置。
- `blockIdx.{x|y|z}`：block 在 x、y 和 z 维度上的 index。这些值会随着正在执行的 block 不同而变化。
- `threadIdx.{x|y|z}`：thread 在 x、y 和 z 维度上的 index。这些值会随着正在执行的 thread 不同而变化。

使用多维的 thread blocks 和 grids 仅仅是为了使用上的便利，并不会影响性能。一个 block 内的 threads 会以可预测的方式被线性化：第一个 index x 变化得最快，其次是 y，然后是 z。这意味着在 thread indices 的线性化过程中，threadIdx.x 的连续取值表示连续的 threads，threadIdx.y 的 stride 为 blockDim.x，而 threadIdx.z 的 stride 为 blockDim.x * blockDim.y。这会影响 threads 如何被分配到 warps，具体细节见 Hardware Multithreading。

Figure 9 展示了一个使用一维 thread blocks 的二维 grid 的简单示例。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)

*Figure 9: Grid of Thread Blocks*

### 2.2.3. GPU Device Memory Spaces

CUDA devices 具有多个 memory spaces，可以在 kernels 中被 CUDA threads 访问。Table 1 总结了常见的 memory types、它们的 thread scopes 以及 lifetimes。后续小节将对这些 memory types 分别进行更详细的说明。

**Table 1: Memory Types, Scopes and Lifetimes**

| Memory Type | Scope  | Lifetime    | Location |
| ----------- | ------ | ----------- | -------- |
| Global      | Grid   | Application | Device   |
| Constant    | Grid   | Application | Device   |
| Shared      | Block  | Kernel      | SM       |
| Local       | Thread | Kernel      | Device   |
| Register    | Thread | Kernel      | SM       |

#### 2.2.3.1. Global Memory

Global memory（也称为 device memory）是用于存储 kernel 中所有 threads 都可以访问的数据的主要 memory space。它类似于 CPU 系统中的 RAM。在 GPU 上运行的 kernels 可以直接访问 global memory，这种方式与在 CPU 上运行的 code 访问 system memory 的方式相同。

Global memory 是持久的。也就是说，在 global memory 中进行的一次 allocation 以及存储在其中的数据会一直存在，直到该 allocation 被释放，或者 application 被终止。cudaDeviceReset 也会释放所有 allocations。

Global memory 通过 CUDA API 调用（例如 cudaMalloc 和 cudaMallocManaged）进行分配。可以使用 CUDA runtime API（例如 cudaMemcpy）将数据从 CPU memory 拷贝到 global memory。通过 CUDA APIs 创建的 global memory allocations 使用 cudaFree 释放。

在 kernel launch 之前，global memory 由 CUDA API 调用进行分配和初始化。在 kernel 执行期间，CUDA threads 可以从 global memory 中读取数据，并将 CUDA threads 执行操作得到的结果写回 global memory。一旦 kernel 执行完成，写入 global memory 的结果可以被拷贝回 host，或者被 GPU 上的其它 kernels 使用。

由于 global memory 可以被一个 grid 中的所有 threads 访问，因此必须注意避免 threads 之间的数据竞争。由于从 host launch 的 CUDA kernels 的返回类型是 void，kernel 计算得到的数值结果返回给 host 的唯一方式，就是将这些结果写入 global memory。

一个用于说明 global memory 使用方式的简单示例是下面的 vecAdd kernel，其中数组 A、B 和 C 位于 global memory 中，并由该 vector add kernel 进行访问。

```
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

#### 2.2.3.2. Shared Memory

Shared memory 是一种可被同一个 thread block 中所有 threads 访问的 memory space。它在物理上位于每个 SM 上，并与 L1 cache（统一数据缓存）使用相同的物理资源。shared memory 中的数据在整个 kernel 执行期间持续存在。shared memory 可以被视为在 kernel 执行期间使用的一块由用户管理的 scratchpad。与 global memory 相比，shared memory 的容量较小，但由于 shared memory 位于每个 SM 上，其 bandwidth 更高，latency 也低于访问 global memory。

由于 shared memory 可被同一个 thread block 中的所有 threads 访问，因此必须注意避免同一个 thread block 内 threads 之间的数据竞争。同一个 thread block 内 threads 之间的 synchronize 可以通过 `__syncthreads()` 函数实现。该函数会阻塞 thread block 中的所有 threads，直到所有 threads 都到达` __syncthreads()` 调用点。

```cpp
// 假设 blockDim.x 为 128
__global__ void example_syncthreads(int* input_data, int* output_data) {
    __shared__ int shared_data[128];

    // 每个 thread 向 shared_data 的一个不同元素写入数据
    shared_data[threadIdx.x] = input_data[threadIdx.x];

    // 所有 threads 进行 synchronize，保证在任何 thread
    // 从 __syncthreads() 解除阻塞之前，对 shared_data 的所有写入都已完成
    __syncthreads();

    // 单个 thread 可以安全地读取 shared_data
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[blockIdx.x] = sum;
    }
}
```

shared memory 的大小取决于所使用的 GPU architecture。由于 shared memory 与 L1 cache 共享相同的物理空间，使用 shared memory 会减少 kernel 可用的 L1 cache 大小。此外，如果 kernel 未使用 shared memory，那么整个物理空间都会被 L1 cache 使用。CUDA runtime API 提供了函数用于在每个 SM 以及每个 thread block 的粒度上查询 shared memory 的大小，可通过使用 cudaGetDeviceProperties 函数并检查 cudaDeviceProp.sharedMemPerMultiprocessor 和 cudaDeviceProp.sharedMemPerBlock 这两个 device properties。

CUDA runtime API 提供了函数 cudaFuncSetCacheConfig，用于告知 runtime 是为 shared memory 分配更多空间，还是为 L1 cache 分配更多空间。该函数向 runtime 指定一个偏好，但不保证一定会被遵循。runtime 可以根据可用资源以及 kernel 的需求自由做出决策。

shared memory 可以通过静态方式和动态方式进行分配。

##### 2.2.3.2.1. Static Allocation of Shared Memory

要以静态方式分配 shared memory，程序员必须在 kernel 内部使用 `__shared__` 修饰符声明一个变量。该变量会被分配在 shared memory 中，并在 kernel 执行期间持续存在。通过这种方式声明的 shared memory 的大小必须在编译期确定。例如，下面位于 kernel 函数体中的代码片段声明了一个包含 1024 个元素、类型为 float 的 shared memory 数组。

```cpp
__shared__ float sharedArray[1024];
```

在完成该声明之后，thread block 中的所有 threads 都可以访问这个 shared memory 数组。必须注意避免同一个 thread block 内 threads 之间的数据竞争，通常需要配合使用` __syncthreads()`。

##### 2.2.3.2.2. Dynamic Allocation of Shared Memory

要以动态方式分配 shared memory，程序员可以在 kernel launch 时，在三重尖括号语法中，将每个 thread block 所需的 shared memory 大小（以字节为单位）作为第三个（可选）参数指定，例如 functionName<<<grid, block, sharedMemoryBytes>>>()。

随后，在 kernel 内部，程序员可以使用 `extern __shared__` 修饰符声明一个变量，该变量会在 kernel launch 时被动态分配。

```cpp
extern __shared__ float sharedArray[];
```

需要注意的一点是，如果希望使用多个动态分配的 shared memory 数组，则必须通过指针算术手动对这一个 `extern __shared__` 进行分区。例如，如果希望在动态分配的 shared memory 中实现与下面代码等价的布局：

```cpp
short array0[128];
float array1[64];
int   array2[256];
```

可以通过以下方式声明并初始化这些数组：

```cpp
extern __shared__ float array[];

short* array0 = (short*)array;
float* array1 = (float*)&array0[128];
int*   array2 = (int*)&array1[64];
```

需要注意，指针必须按照其所指向类型进行对齐。因此，例如下面的代码是不可行的，因为 array1 没有按照 4 字节对齐。

```cpp
extern __shared__ float array[];
short* array0 = (short*)array;
float* array1 = (float*)&array0[127];
```

#### 2.2.3.3. Registers

Registers（寄存器）位于 SM 上，并具有 thread local scope（线程局部作用域）。寄存器的使用由编译器管理，且在 kernel 执行期间用于 thread local storage（线程局部存储）。可以通过 GPU 的设备属性 `regsPerMultiprocessor` 和 `regsPerBlock` 查询每个 SM 的寄存器数量以及每个 thread block 的寄存器数量。

NVCC 允许开发者通过 `-maxrregcount` 选项指定一个 kernel 可以使用的最大寄存器数。使用该选项减少 kernel 可用的寄存器数可能会导致更多的 thread blocks 能够同时在 SM 上被调度，但也可能导致更多的 register spilling（寄存器溢出）。

#### 2.2.3.4. Local Memory 本地内存

 Local memory 是类似于 registers 的 thread local 存储，由 NVCC 管理，但本地内存的物理位置在 global memory 空间。“local” 标签指的是它的**逻辑作用域**，而不是物理位置。Local memory 在 kernel 执行期间用于 thread local storage。编译器可能放入 local memory 的 automatic 变量包括：

- 编译器无法确定其访问index是常量的 arrays，
- 会消耗过多 register 空间的大型结构体或 arrays，
- 如果 kernel 使用的 register 超过可用数量（也就是 register spilling）时的任何变量。

由于 local memory 空间驻留在 device memory 中，local memory 的访问具有与 global memory 访问相同的延迟和带宽，并且受与 Coalesced Global Memory Access 描述的相同内存合并要求的约束。然而 local memory 组织方式使得连续的 32-bit 字按连续的 thread IDs 访问。因此，只要一个 warp 中的所有 threads 访问相同的相对地址（例如在数组变量中相同的索引或结构体变量中的相同成员），访问就是完全 coalesced 的。

#### 2.2.3.5. Constant Memory

Constant memory 在 grid scope可见，并且在应用程序的整个生命周期内都可以访问。Constant memory 位于 device 上，对 kernel 是只读的。因此，它必须在 host 端使用 `__constant__` 修饰符声明并初始化，并且放在任何函数之外。

`__constant__` 内存空间修饰符声明的变量：

  * 位于 constant memory 空间，

  * 具有其创建时 CUDA context 的生命周期，

  * **每个 device 有一个独立的对象**，

  * 可以从 grid 中所有 threads 以及通过运行时库（`cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`）从 host 访问。

可以通过 `totalConstMem` device 属性项查询 constant memory 的总量。

Constant memory 对于每个 thread 以只读方式使用小量数据的场景很有用。与其它 memory 相比，constant memory 很小，通常每个 device 大约 64KB。

下面给出声明和使用 constant memory 的示例代码。

```c++
// 在你的 .cu 文件中
__constant__ float coeffs[4];

__global__ void compute(float *out) {
    int idx = threadIdx.x;
    out[idx] = coeffs[0] * idx + coeffs[1];
}
// 在你的 host 代码中
float h_coeffs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
cudaMemcpyToSymbol(coeffs, h_coeffs, sizeof(h_coeffs));
compute<<<1, 10>>>(device_out);
```

#### 2.2.3.6. Caches

GPU devices 有一个多级缓存结构，包括 L2 和 L1 caches。

L2 cache 位于 device 上，并被所有的 SMs 共享。可以通过 `cudaGetDeviceProperties` 函数返回的 `l2CacheSize` device property 元素 查询 L2 cache 的大小。

如上面 Shared Memory 所述，L1 cache 物理位于每个 SM 上，并与 shared memory 使用相同的物理空间。如果 kernel 没有使用 shared memory，那么整个物理空间都会作为 L1 cache 被利用。

可以通过一些函数来控制 L2 和 L1 caches，使开发者指定各种 cache 行为。这些函数的详细信息可见于 [Configuring L1/Shared Memory Balance]、[L2 Cache Control] 和 [Low-Level Load and Store Functions]。

如果不使用这些提示，compiler 和 runtime 会尽最大努力高效利用 caches。

#### 2.2.3.7. Texture and Surface Memory

> **Note**
>
> 某些较早的 CUDA 代码可能会使用 texture memory，因为在较早的 NVIDIA GPUs 上，在某些情境下这样做会带来性能优势。在所有当前受支持的 GPUs 上，这些情境都可以通过直接 load 和 store 指令来处理，使用 texture 和 surface memory instructions 不再提供任何性能好处。

GPU 可能具有用于从图像加载数据以用于 3D rendering 中的 textures 的专用指令。CUDA 在 texture object API 和 surface object API 中提供了这些指令以及使用它们的机制。

本指南不再进一步讨论 Texture 和 Surface memory，因为在任何当前受支持的 NVIDIA GPU 上在 CUDA 中使用它们没有优势。CUDA developers 可以放心忽略这些 APIs。对于仍在旧代码基中使用它们的开发者，这些 APIs 的解释仍然可以在 legacy CUDA C++ Programming Guide 中找到。

#### 2.2.3.8. Distributed Shared Memory

在 compute capability 9.0 中引入的 thread block cluster，并通过 Cooperative Groups 提供支持，使得一个 thread block cluster 中的 threads 能够访问该 cluster 内所有参与 thread blocks 的 shared memory。这种被分区的 shared memory 被称为 Distributed Shared Memory，相应的地址空间称为 Distributed Shared Memory address space。属于同一个 thread block cluster 的 threads 可以在 distributed address space 中进行 read、write 或 atomic 操作，而不管该地址属于本地 thread block 还是远程 thread block。无论一个 kernel 是否使用 distributed shared memory，shared memory 的大小规格（无论是 static 还是 dynamic）仍然是以每个 thread block 为单位。Distributed shared memory 的总大小等于 cluster 中 thread blocks 的数量乘以每个 thread block 的 shared memory 大小。

访问 distributed shared memory 要求所有 thread blocks 都存在。用户可以通过 cluster_group 类中的 `cluster.sync()` 来保证所有 thread blocks 都已经开始执行。用户还需要确保所有 distributed shared memory 操作都发生在 thread block 退出之前，例如，如果一个远程 thread block 正在尝试读取某个 thread block 的 shared memory，程序必须确保该 shared memory 的读取在该 thread block 退出之前完成。

下面我们来看一个简单的 histogram 计算示例，以及如何使用 thread block cluster 在 GPU 上对其进行优化。计算 histogram 的一种常见方式是在每个 thread block 的 shared memory 中完成计算，然后再对 global memory 执行 atomic 操作。这种方法的一个限制是 shared memory 的容量。一旦 histogram 的 bins 数量无法再放入 shared memory，用户就需要直接在 global memory 中计算 histogram，从而也就需要在 global memory 中执行 atomics。通过 distributed shared memory，CUDA 提供了一个中间层级：根据 histogram bins 的大小，histogram 可以在 shared memory、distributed shared memory，或者直接在 global memory 中计算。

下面的 CUDA kernel 示例展示了如何根据 histogram bins 的数量，在 shared memory 或 distributed shared memory 中计算 histogram。

```C++
#include <cooperative_groups.h>
// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();
  // cluster 初始化、大小以及本地 bin 偏移量的计算
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; // 将 shared memory 中的 histogram 初始化为 0
  }
  // cluster 同步确保 cluster 中所有 thread blocks 的 shared memory
  // 都已经初始化为 0。同时也保证所有 thread blocks
  // 已经开始执行并且是并发存在的。
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];
    // 找到正确的 histogram bin
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    // 找到用于计算 distributed shared memory histogram 的
    // 目标 block rank 和偏移量
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;
    // 指向目标 block 的 shared memory 的指针
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // 对 histogram bin 执行 atomic 更新
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // 需要 cluster 同步来确保所有 distributed shared memory 操作
  // 都已经完成，并且在其他 thread blocks
  // 仍在访问 distributed shared memory 时
  // 不会有 thread block 提前退出
  cluster.sync();
  // 使用本地的 distributed memory histogram
  // 执行 global memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

上述 kernel 可以在 runtime 根据所需的 distributed shared memory 大小，以不同的 cluster size 进行 launch。如果 histogram 足够小，可以完全放入单个 block 的 shared memory 中，那么用户可以使用 cluster size 为 1 来 launch 该 kernel。下面的代码片段展示了如何根据 shared memory 需求动态地 launch 一个 cluster kernel。

```C++
// 通过 extensible launch 启动
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // cluster_size 取决于 histogram 的大小
  // ( cluster_size == 1 ) 表示不使用 distributed shared memory，仅使用 thread block 本地的 shared memory
  int cluster_size = 2; // 这里以 size 2 作为示例
  int nbins_per_block = nbins / cluster_size;
  // dynamic shared memory 的大小是按 block 计算的
  // Distributed shared memory 的总大小 = cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

### 2.2.4. Memory Performance

确保正确使用内存是实现 CUDA kernels 高性能的关键。本节讨论了一些实现 CUDA kernels 高内存吞吐量的一般原则和示例。

#### 2.2.4.1. Coalesced Global Memory Access（合并的全局内存访问）

全局内存通过 **32 字节**内存传输进行访问。当一个 CUDA 线程从全局内存请求一个数据字时，相关的 warp 会将该 warp 内所有线程的内存请求合并为满足该请求所需的内存传输次数，这取决于每个线程访问的数据字大小和线程之间内存地址的分布情况。例如，如果一个线程请求一个 4 字节的数据字，那么 warp 生成的实际内存传输将是总共 32 字节。要最有效地利用内存系统，warp 应该使用在单次内存传输中提取的所有内存。也就是说，如果一个线程从全局内存请求一个 4 字节的数据字，并且传输大小是 32 字节，如果该 warp 中的其他线程能够利用该 32 字节请求中的其他 4 字节数据，那么这将实现内存系统的最高效使用。

作为一个简单示例，如果 warp 中的连续线程在内存中请求连续的 4 字节字，那么 warp 总共将请求 128 字节的内存，而这 128 字节将通过四个 32 字节内存传输提取。这将导致内存系统的 100 % 利用率。也就是说，该 warp 的 100 % 内存流量都被利用。图 10 展示了这种完全合并内存访问的例子。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/perfect_coalescing_32byte_segments.png)

*图 10. 合并的内存访问*

相反，最糟糕的情况是连续线程访问在内存中相距 32 字节或更远的数据元素。在这种情况下，warp 将被迫为每个线程发出一个 32 字节的内存传输，总内存流量将是 32 字节 × 32 线程/warp = 1024 字节。然而，实际使用的数据仅为 128 字节（warp 中每个线程 4 字节），因此内存利用率仅为 128 / 1024 = 12.5 %。这是一种非常低效的内存系统使用情况。图 11 展示了这种未合并内存访问的例子。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/no_coalescing_32byte_segments.png)

*图 11. 未合并的内存访问*

实现合并内存访问的最直接方法是让连续线程访问内存中连续的元素。例如，对于用 1D thread blocks 启动的 kernel，下面的 `vecAdd` kernel 将实现合并的内存访问。注意线程 `workIndex` 如何访问三个数组，以及连续线程（由连续的 `workIndex` 值表示）如何访问数组中的连续元素。

```c++
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}
```

实际上，并不要求连续线程必须访问内存中的连续元素才能实现合并内存访问，这只是常见的情况。只要 warp 中的所有线程以某种线性或置换方式访问相同的 32 字节内存段中的元素，就会发生合并内存访问。换句话说，实现合并内存访问的最佳方式是最大化使用到的字节与传输字节的比率。

> **注意：** 确保全局内存访问的正确合并是编写高性能 CUDA kernels 最重要的性能考虑因素之一。务必让应用尽可能高效地使用内存系统。

##### 2.2.4.1.1. 使用 global memory 的矩阵转置示例

作为一个简单的示例，考虑一个 out-of-place 的矩阵转置 kernel，它将一个大小为 N × N 的 32 位 float 方阵从矩阵 a 转置到矩阵 c。这个示例使用一个二维 grid，并假设 launch 的是大小为 32 × 32 个线程的二维 thread block，也就是说 `blockDim.x = 32`、`blockDim.y = 32`，因此每个二维 thread block 会处理矩阵中的一个 32 × 32 的 tile。每个线程只操作矩阵中的一个唯一元素，所以不需要对线程进行显式同步。图 12 展示了这个矩阵转置操作。kernel 的源代码在该图之后给出。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/global_transpose.png)

*图 12 使用 global memory 的矩阵转置*

> 每个矩阵顶部和左侧的标签表示二维 thread block 的索引，也可以看作 tile 的索引，其中每个小方块表示矩阵中的一个 tile，由一个二维 thread block 来处理。在这个示例中，tile 的大小是 32 × 32 个元素，因此每个小方块都代表矩阵中的一个 32 × 32 的 tile。绿色阴影的方块展示了某个示例 tile 在转置操作前后的位置。

```c++
    /* 用于在行主序中使用 2D 索引索引 1D 内存数组 的宏 */
    /* ld 是 leading dimension，即矩阵的列数          */

    #define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )

    /* 用于朴素矩阵转置的 CUDA kernel */

    __global__ void naive_cuda_transpose(int m, float *a, float *c )
    {
        int myCol = blockDim.x * blockIdx.x + threadIdx.x;
        int myRow = blockDim.y * blockIdx.y + threadIdx.y;
        if( myRow < m && myCol < m )
        {
            c[INDX( myCol, myRow, m )] = a[INDX( myRow, myCol, m )];
        } /* end if */
        return;
    } /* end naive_cuda_transpose */
```

为了判断这个 kernel 是否实现了coalesced memory access，需要看连续的线程是否在访问连续的内存元素。在一个2d thread block 中，`x` 维索引变化最快，因此连续的 `threadIdx.x` 值应该访问连续的内存元素。`threadIdx.x` 出现在 myCol 中，可以观察到，当 `myCol` 作为 `INDX` 宏的第二个参数时，连续的线程会读取 a 中连续的元素，因此对 a 的读取是完美 coalesced 的。

然而，对 `c` 的写不是 coalesced 写，因为连续的 `threadIdx.x` 值（再次检查 `myCol`）正在将元素写入 `c`，这些元素彼此之间相隔 `ld`（leading dimension）个元素。之所以观察到这种情况，是因为现在 `myCol` 是传递给 `INDX` 宏的第一个参数，而作为 `INDX` 的第一个参数每增加 1，内存位置就会改变 `ld`。当 `ld` 大于 32（即矩阵尺寸大于 32 时会发生这种情况），这相当于 **图 11** 中所示的病态情况。

为了解决这些非 coalesced 的写入，可以使用 shared memory，这将在下一节中描述。

#### 2.2.4.2. Shared Memory 访问模式

Shared memory 有 32 个 banks，这些 banks 被组织成连续的 **32-bit** 字映射到连续的 banks。每个 bank 在一个时钟周期内具有 32 bit 的带宽。

当同一个 warp 中的多个 threads 试图访问同一个 bank 中不同元素时，就会发生 bank conflict。在这种情况下，该 bank 中的数据访问会被串行化，直到所有请求该 bank 的 threads 都完成访问。这种访问的串行化会造成性能损失。

有两种情况不属于这种 bank conflict：当同一个 warp 中的多个 threads 访问（读或写）同一个 shared memory 位置时，对于读访问，该 word 会广播给请求它的所有 threads；对于写访问，每个 shared memory 地址只会被其中一个 thread 写入（到底是哪一个 thread 执行写操作未定义）。

下图展示了一些带 stride 的访问示例。其中红色方框表示 shared memory 中的一个唯一位置。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/examples-of-strided-shared-memory-accesses.png)



**图 13. 32 bit bank size模式下的带 stride 的 Shared Memory 访问。*
 *左：stride 为 1 个 32-bit word 的线性寻址（无 bank conflict）。*
 *中：stride 为 2 个 32-bit word 的线性寻址（2 路 bank conflict）。*
 右：stride 为 3 个 32-bit word 的线性寻址（无 bank conflict）。*

下图展示了涉及 broadcast 机制的内存读访问示例。同样红色方框表示 shared memory 中的唯一位置。如果有多个箭头指向同一个位置，则数据会广播到所有请求它的 threads。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/examples-of-irregular-shared-memory-accesses.png)

图 14. 不规则的 Shared Memory 访问。
 左：通过随机排列的无冲突访问。
 中：由于 threads 3、4、6、7 和 9 都访问 bank 5 中的**同一个 word**，所以也是无冲突访问。
 右：所有 threads 访问同一个 bank 内的同一个 word，所以也是无冲突访问。

> **Note**
>  避免 bank conflict 是编写高性能 Shared Memory CUDA kernels 时的重要性能考虑因素。

##### 2.2.4.2.1. 使用 Shared Memory 的矩阵转置示例

在前面的示例 Matrix Transpose Example Using Global Memory 中，说明了一个功能上正确但没有针对全局 memory 进行高效访问的朴素矩阵转置实现，因为对 `c` 矩阵的写不会正确地实现 coalesced 访问。在此示例中，shared memory 被作为用户管理的缓存来暂存从全局 memory 加载和存储的数据，从而使全局 memory 的读写都实现 coalesced 访问。

示例

```C++
 1/* 定义在 X 和 Y 方向的 thread block 大小 */
 2
 3#define THREADS_PER_BLOCK_X 32
 4#define THREADS_PER_BLOCK_Y 32
 5
 6/* 宏用于用 2D 索引访问按行主序存放的 1D memory 数组 */
 7/* ld 是 leading dimension，即矩阵的列数 */
 8
 9#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )
10
11/* 使用 shared memory 的 CUDA kernel 实现矩阵转置 */
12
13 __global__ void smem_cuda_transpose(int m, float *a, float *c )
14{
15
16    /* 声明一个静态分配的 shared memory 数组 */
17
18    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];
19
20    /* 计算当前 tile 的行和列起始索引 */
21
22    const int tileCol = blockDim.x * blockIdx.x;
23    const int tileRow = blockDim.y * blockIdx.y;
24
25    /* 从全局 memory 读取数据到 shared memory 数组 */
26    smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];
27
28    /* 在线程块内同步所有线程 */
29    __syncthreads();
30
31    /* 从 shared memory 写回结果到全局 memory */
32    c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];
33    return;
34
35} /* end smem_cuda_transpose */
```

带数组边界检查的示例

```C++
 1/* 定义在 X 和 Y 方向的 thread block 大小 */
 2
 3#define THREADS_PER_BLOCK_X 32
 4#define THREADS_PER_BLOCK_Y 32
 5
 6/* 宏用于用 2D 索引访问按列主序存放的 1D memory 数组 */
 7/* ld 是 leading dimension，即矩阵的行数 */
 8
 9#define INDX( row, col, ld ) ( ( (col) * (ld) ) + (row) )
10
11/* 使用 shared memory 的 CUDA kernel 实现矩阵转置 */
12
13 __global__ void smem_cuda_transpose(int m,
14                                    float *a,
15                                    float *c )
16{
17
18    /* 声明一个静态分配的 shared memory 数组 */
19
20    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];
21
22    /* 计算用于边界检查的线程全局行和列索引 */
23
24    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
25    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;
26
27    /* 计算当前 tile 的行和列起始索引 */
28
29    const int tileX = blockDim.x * blockIdx.x;
30    const int tileY = blockDim.y * blockIdx.y;
31
32    if( myRow < m && myCol < m )
33    {
34        /* 从全局 memory 读取数据到 shared memory 数组 */
35        smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileX + threadIdx.x, tileY + threadIdx.y, m )];
36    } /* end if */
37
38    /* 在线程块内同步所有线程 */
39    __syncthreads();
40
41    if( myRow < m && myCol < m )
42    {
43        /* 从 shared memory 写回结果到全局 memory */
44        c[INDX( tileY + threadIdx.x, tileX + threadIdx.y, m )] = smemArray[threadIdx.y][threadIdx.x];
45    } /* end if */
46    return;
47
48} /* end smem_cuda_transpose */
```

本示例展示的基本性能优化是确保访问全局 memory 时内存访问是 coalesced 的。在执行复制之前，每个线程先计算自己的 `tileRow` 和 `tileCol` 索引。这些索引是该线程所在 thread block 要处理的特定 tile 的起始索引，并且同一个 thread block 内的所有线程具有相同的 `tileRow` 和 `tileCol` 值，因此可以将其视为该 thread block 执行的 tile 的起始位置。

随后 kernel 对每个线程块执行以下语句，将矩阵的 32 x 32 tile 从全局 memory 复制到 shared memory。由于一个 warp 包含 32 个 threads，此复制操作将由 32 个 warp 执行，warp 之间执行顺序不保证。

    smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];

请注意，因为 `threadIdx.x` 出现在 `INDX` 的第二个参数中，所以连续 threads 正在访问内存中连续的元素，对 `a` 的读取是完全 coalesced 的。

kernel 的下一步是调用 `__syncthreads()`。这确保了线程块内的所有线程完成前面的代码执行后才继续下一步，因此写入 shared memory 的操作在下一步之前完成。这一点非常重要，因为下一步将涉及从 shared memory 读取数据。如果没有 `__syncthreads()`，则不能保证 shared memory 中的 `a` 数据写操作在某些 warp 继续执行代码之前已由所有 warp 完成。

此时，对每个 thread block 来说，`smemArray` 中已经按原矩阵顺序存放了一个 32 x 32 的 tile。为了确保 tile 内的元素被正确转置，在从 `smemArray` 读取时交换 `threadIdx.x` 和 `threadIdx.y`。为了确保整体 tile 被写回到 `c` 中正确位置，还要交换 `tileRow` 和 `tileCol`。为了确保正确 coalesced，这里在 `INDX` 的第二个参数中使用了 `threadIdx.x`，如下所示：

    c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];

该 kernel 展示了 shared memory 的两种常见用途。

  * 使用 shared memory 暂存从全局 memory 读取的数据，以确保对全局 memory 的读和写都实现 coalesced。
  * 使用 shared memory 使同一 thread block 内的 threads 之间共享数据。

##### 2.2.4.2.2. Shared Memory Bank Conflicts

在本节中，我们已经描述了 shared memory 的 bank 结构。在前面的矩阵转置示例中，虽然实现了 global memory 访问的 coalesced，但没有考虑是否存在 shared memory bank conflict。考虑如下 2D shared memory 声明：

```
__shared__ float smemArray[32][32];
```

因为一个 warp 有 32 个 threads，同一个 warp 中的每个 thread 会有固定的 `threadIdx.y` 值，并且 `0 <= threadIdx.x < 32`。

图 15 的左图说明了当 warp 中的 threads 在访问 `smemArray` 的一列数据时的情况。Warp 0 正在访问 memory locations `smemArray[0][0]` 到 `smemArray[31][0]`。在 C++ 的多维数组顺序中，最后一个索引变化最快，所以 Warp 0 中的连续 threads 访问的数据地址相差 32 个元素，如图所示，不同颜色表示不同 bank，这种按列访问会导致 32 路 bank conflict。

图 15 的右图说明了当 warp 中的 threads 在访问 `smemArray` 的一行数据时的情况。在这种情况下，Warp 0 中连续的 threads 访问的是相邻的数据地址，如图所示，这种按行访问不会产生 bank conflict。理想情况是同一个 warp 中的每个 thread 都访问不同颜色的 bank。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/bank-conflicts-shared-mem.png)

**图 15. 32 x 32 shared memory 数组的 bank 结构。*
 *方框中的数字表示 warp 索引；颜色表示对应 shared memory 位置的 bank。*

回到 Section 2.2.4.2.1 的示例，我们可以检查 shared memory 是否存在 bank conflict。第一次使用 shared memory 时，是将 global memory 数据写入 shared memory：

```c++
smemArray[threadIdx.x][threadIdx.y] = a[INDX( tileRow + threadIdx.y, tileCol + threadIdx.x, m )];
```

由于 C++ 数组按行优先存储，同一个 warp 中 `threadIdx.x` 连续变化的 threads 会以 32 元素的 stride 访问 `smemArray`。这会导致一个 32 路的 bank conflict，如图 15 左图所示。

第二次使用 shared memory 是将 shared memory 中的数据写回 global memory：

```c++
c[INDX( tileCol + threadIdx.y, tileRow + threadIdx.x, m )] = smemArray[threadIdx.y][threadIdx.x];
```

在这种情况下，因为 `threadIdx.x` 是第二个索引，连续 threads 会以 1 元素的 stride 访问 `smemArray`，不会产生 bank conflict，如图 15 右图所示。

这个矩阵转置 kernel 有一次 shared memory 访问不会产生 bank conflict，还有一次会产生 32 路 bank conflict。一个常见的解决方法是通过在数组的列维度上加 1 来 pad shared memory，如下：

```c++
__shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];
```

这个简单的调整会消除 bank conflict。如下图中声明了一个 32 x 33 的 shared memory 数组，无论 threads 是按列还是按行访问 shared memory，都会访问不同颜色的 bank，从而消除 bank conflict。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/no-bank-conflicts-shared-mem.png)

图 16. 32 x 33 shared memory 数组的 bank 结构。

### 2.2.5. Atomics

高性能的 CUDA kernels 依赖于尽可能表达更多的算法并行性。GPU kernel 执行的异步特性要求 threads 尽可能独立地运行。threads 并不总是能够完全独立，正如在 Shared Memory 中看到的那样，同一个 thread block 内的 threads 存在一种用于交换数据和进行同步的机制。

在整个 grid 的层级上，并不存在一种机制可以对 grid 中的所有 threads 进行同步。不过，可以通过 atomic functions 来为 global memory 位置提供同步访问机制。Atomic functions 允许某个 thread 获取一个 global memory 位置的锁，并在该位置上执行一次 read-modify-write 操作。在锁被持有期间，其他 thread 无法访问同一个位置。CUDA 提供了与 C++ 标准库 atomics 具有相同行为的 atomics，即 `cuda::std::atomic` 和 `cuda::std::atomic_ref`。CUDA 还提供了扩展的 C++ atomics，即 `cuda::atomic` 和 `cuda::atomic_ref`，它们允许用户指定 atomic 操作的 thread scope。Atomic functions 的具体细节在 Atomic Functions 中介绍。

下面展示了一个使用 `cuda::atomic_ref` 执行 device 范围 atomic 加法的示例，其中 `array` 是一个 float 数组，`result` 是一个指向 global memory 中某个位置的 float 指针，该位置用于存储数组求和的结果。

```c++
__global__ void sumReduction(int n, float *array, float *result) {

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);
    result_ref.fetch_add(array[tid]);
    ...
}
```

应当谨慎使用 atomic functions，因为它们会强制进行 thread 同步，从而可能影响性能。

### 2.2.6. Cooperative Groups （协作组）

Cooperative groups 是 CUDA C++ 中提供的一种软件工具，允许应用程序定义可以彼此同步的 threads 组，即使这些 threads 分布在多个 thread blocks 中，或分布在单个 GPU 上的多个 grids 中；该工具同样适用于多 GPU 应用场景。CUDA 编程模型通常允许 thread block 或 thread block cluster 内的 threads 高效地进行同步，但并未提供一种机制来指定比 thread block 或 cluster 更小的 thread groups。同样，CUDA 编程模型也未提供能够在 thread blocks 之间实现同步的机制或保证。

Cooperative groups 通过软件方式提供了这两种能力。Cooperative groups 允许应用程序创建跨越 thread blocks 和 clusters 边界的 thread groups，但这样做会带来一些语义限制和性能方面的影响，这些内容在介绍 cooperative groups 的 feature section 中有详细说明。

### 2.2.7. Kernel Launch and Occupancy

当一个 CUDA kernel 被 launch 时，CUDA threads 会根据 kernel launch 时指定的 execution configuration 组织成 thread blocks 和一个 grid。一旦 kernel 被 launch，scheduler 会把 thread blocks 分配给 SMs。哪些 thread blocks 被 schedule 到哪些 SM 的细节既不能被应用控制也不能被查询，并且 scheduler 不保证任何顺序，因此程序不能依赖特定的调度顺序或方案来保证正确执行。

能够被 schedule 到一个 SM 的 blocks 数量取决于给定 thread block 所需的 hardware resources 和该 SM 上可用的 hardware resources。当 kernel 第一次被 launch 时，scheduler 开始把 thread blocks 分配到 SMs。只要 SMs 上有未被其它 thread blocks 占用的足够硬件资源，scheduler 就会继续把 thread blocks 分配给 SMs。如果某一时刻没有任何 SM 有能力接受另一个 thread block，scheduler 就会等待这些 SM 完成之前被分配的 thread blocks。一旦这种情况发生，SM 就可以接受更多的工作，scheduler 会把 thread blocks 分配给它们。这个过程会持续直到所有 thread blocks 被 schedule 并执行完毕。

`cudaGetDeviceProperties` 函数允许应用通过 device properties 查询每个 SM 的限制。注意这些限制是针对每个 SM 和每个 thread block 的。

- `maxBlocksPerMultiProcessor`: 每个 SM 的最大 resident blocks 数量。
- `sharedMemPerMultiprocessor`: 每个 SM 可用的 shared memory 字节数。
- `regsPerMultiprocessor`: 每个 SM 上可用的 32-bit registers 数量。
- `maxThreadsPerMultiProcessor`: 每个 SM 的最大 resident threads 数量。
- `sharedMemPerBlock`: 每个 thread block 可以分配的最大 shared memory 字节数。
- `regsPerBlock`: 每个 thread block 可以分配的最大 32-bit registers 数量。
- `maxThreadsPerBlock`: 每个 thread block 的最大 threads 数量。

一个 CUDA kernel 的 occupancy 是 active warps 数量与 SM 支持的最大 active warps 数量之比。通常，尽可能提升 occupancy 是一个好实践，这可以隐藏 latency 并提升性能。

要计算 occupancy，需要知道刚刚描述的 SM 的资源限制，同时还需要知道特定 CUDA kernel 所需的资源。要确定每个 kernel 的资源使用情况，在程序编译期间可以使用 `--resource-usage` 选项传给 `nvcc`，这样会显示 kernel 所需的 registers 和 shared memory 数量。

下面举例说明，考虑一个 compute capability 10.0 的设备，其 device properties 如 Table 2 所列。

*Table 2 SM Resource Example*

| 资源                          | 值     |
| ----------------------------- | ------ |
| `maxBlocksPerMultiProcessor`  | 32     |
| `sharedMemPerMultiprocessor`  | 233472 |
| `regsPerMultiprocessor`       | 65536  |
| `maxThreadsPerMultiProcessor` | 2048   |
| `sharedMemPerBlock`           | 49152  |
| `regsPerBlock`                | 65536  |
| `maxThreadsPerBlock`          | 1024   |

如果一个 kernel 以 `testKernel<<<512, 768>>>()` launch，即每个 block 有 768 个 threads，那么每个 SM 在同一时间最多只能执行 2 个 thread blocks。因为 `maxThreadsPerMultiProcessor` 是 2048，scheduler 不能为每个 SM 分配超过 2 个 thread blocks。所以 occupancy 是 (768 * 2) / 2048，即 75%。

如果一个 kernel 以 `testKernel<<<512, 32>>>()` launch，即每个 block 有 32 个 threads，那么每个 SM 就不会碰到 `maxThreadsPerMultiProcessor` 的限制，但由于 `maxBlocksPerMultiProcessor` 是 32，scheduler 最多只能为每个 SM 分配 32 个 thread blocks。因为每个 block 的 threads 是 32，总共的 resident threads 是 32 blocks * 32 threads per block，即 1024 个 threads。由于 compute capability 10.0 SM 的最大 resident threads 数是 2048，因此在这种情况下 occupancy 是 1024 / 2048，即 50%。

同样的分析也可以应用于 shared memory。例如如果一个 kernel 使用了 100KB 的 shared memory，那么 scheduler 每个 SM 最多只能分配 2 个 thread blocks，因为第三个 thread block 需要额外的 100KB shared memory，总计 300KB，这超过了每个 SM 可用的 233472 bytes。

block 中的 threads 数和每个 block 使用的 shared memory 是由程序员明确控制的，可以调整以实现期望的 occupancy。程序员对 register 使用的控制是有限的，因为编译器和 runtime 会试图优化 register 使用。然而，程序员可以通过传给 `nvcc` 的 `--maxrregcount` 选项来指定每个 thread block 的最大 registers 数量。如果 kernel 需要的 registers 数超过这个指定的数量，kernel 很可能会出现 spill 到 local memory 的情况，这会改变 kernel 的性能特性。在某些情况下，即使发生 spilling，限制 registers 也可以允许更多的 thread blocks 被 schedule，从而提高 occupancy 并可能带来性能净提升。
