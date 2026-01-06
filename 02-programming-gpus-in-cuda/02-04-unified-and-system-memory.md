# 2.4 Unified and System Memory

异构系统有多个物理内存可以存储数据。host CPU 连接有 DRAM，并且系统中的每个 GPU 也有自己的 DRAM。当数据驻留在执行访问的处理器的内存中时性能最好。CUDA 提供了明确管理 memory placement 的 APIs，但这可能繁琐并使软件设计复杂。CUDA 提供的特性和能力旨在简化不同物理内存之间的分配、放置和迁移。

本章的目的是介绍和解释这些特性，以及它们对应用开发者在功能和性能方面的意义。unified memory 有几种不同的表现形式，这取决于 OS、driver 版本和 GPU。本章将展示如何确定哪个 unified memory 范式适用，以及 unified memory 各特性的行为。后面的 unified memory 章节将更详细解释 unified memory。

本章将定义和解释以下概念：

* Unified Virtual Address Space — CPU memory 和每个 GPU 的 memory 在单一虚拟 address space 中具有各自区间
* Unified Memory — 一个 CUDA 特性，使 managed memory 能够在 CPU 和 GPUs 之间自动迁移
    * Limited Unified Memory — 有一些限制的 unified memory 范式
    * Full Unified Memory — 完整支持 unified memory 特性
    * Full Unified Memory with Hardware Coherency — 使用硬件功能完整支持 unified memory
    * Unified memory hints — 指导 unified memory 行为的 APIs
* Page-locked Host Memory — 不可分页的系统 memory，某些 CUDA 操作需要
    * Mapped memory — 一种（不同于 unified memory）可以直接从 kernel 访问 host memory 的机制

此外，本章引入讨论 unified 和系统 memory 时使用的术语：

* Heterogeneous Managed Memory (HMM) — Linux kernel 提供的软件一致性以支持完整 unified memory 的特性
* Address Translation Services (ATS) — 一种硬件特性，当 GPU 与 CPU 通过 NVLink Chip-to-Chip (C2C) 互联时可用，它为完整 unified memory 提供硬件一致性

### 2.4.1. Unified Virtual Address Space
单一虚拟 address space 被用于系统内所有 host memory 和所有 GPUs 的 global memory。系统内 host 和所有 devices 上的所有 memory allocations 都位于这个虚拟 address space 中。不论 allocations 是用 CUDA APIs（例如 `cudaMalloc`, `cudaMallocHost`）还是用系统 allocation APIs（例如 `new`, `malloc`, `mmap`）进行。CPU 和每个 GPU 在 unified virtual address space 中都有独特的区间。
这意味着：

* 可以使用 `cudaPointerGetAttributes()` 从 pointer 的值确定任意 memory 的位置（即在 CPU 还是某个 GPU 上）
* `cudaMemcpy*()` 的 `cudaMemcpyKind` 参数可以设为 `cudaMemcpyDefault`，从 pointers 自动确定 copy 类型

### 2.4.2. Unified Memory

Unified memory 是一种 CUDA 内存特性，它允许称为 managed memory 的内存分配可以被运行在 CPU 或 GPU 上的代码访问。在 CUDA 在 C++ 中的介绍部分展示了 unified memory。Unified memory 在所有受 CUDA 支持的系统上都可用。 

在某些系统上，必须显式分配 managed memory。Managed memory 可以通过几种不同的方式在 CUDA 中显式分配： 

* 使用 CUDA API `cudaMallocManaged` 
* 使用 CUDA API `cudaMallocFromPoolAsync`，其中 pool 是通过 `allocType` 为 `cudaMemAllocationTypeManaged` 的方式创建的 
* 使用 `__managed__` specifier 的全局变量（参见 Memory Space Specifiers）

在具有 HMM 或 ATS 的系统上，无论如何分配，所有 system memory 均是隐式 managed memory，不需要特殊的分配。 

#### 2.4.2.1. Unified Memory Paradigms

Unified memory 的特性和行为会随着操作系统、Linux 上的内核版本、GPU 硬件以及 GPU-CPU 互联方式的不同而变化。可用的 unified memory 形式可以通过使用 `cudaDeviceGetAttribute` 查询一些属性来确定：

- `cudaDevAttrConcurrentManagedAccess` - 如果是 1 表示支持完整 unified memory，否则为 0 表示支持有限的 unified memory
- `cudaDevAttrPageableMemoryAccess` - 如果是 1 表示所有系统内存都是完全支持的 unified memory，否则为 0 表示只有明确分配为 managed memory 的内存才是完全支持的 unified memory
- `cudaDevAttrPageableMemoryAccessUsesHostPageTables` - 指示 CPU/GPU 一致性的机制：值为 1 表示硬件一致性，值为 0 表示软件一致性。

图 18 展示了如何以可视化的方式确定 unified memory paradigm，并且随后附有一个按相同逻辑实现的代码示例。

Unified memory 操作有四种 paradigm：

- 对显式 managed memory allocations 的完全支持
- 对所有 allocations 的完全支持并使用软件一致性
- 对所有 allocations 的完全支持并使用硬件一致性
- 有限的 unified memory 支持

当提供完整支持时，它可以要求显式的 allocations，或者所有系统内存可能隐式地成为 unified memory。当所有内存隐式地是 unified 时，一致性机制可以是软件或硬件。在 Windows 和某些 Tegra 设备上，unified memory 的支持是有限的。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/unified-memory-explainer.png)

***图 18：** 当前所有 GPU 都使用统一虚拟地址空间，并且提供 unified memory。当 `cudaDevAttrConcurrentManagedAccess` 的值为 1 时，表示可用完整的 unified memory 支持；否则只提供有限支持。当提供完整支持时，如果 `cudaDevAttrPageableMemoryAccess` 的值也是 1，则所有系统内存都是 unified memory。否则，只有通过 CUDA API（例如 `cudaMallocManaged`）分配的内存才是 unified memory。当所有系统内存都是 unified memory 时，`cudaDevAttrPageableMemoryAccessUsesHostPageTables` 用于指示一致性是由硬件提供（值为 1）还是由软件提供（值为 0）。*

下表（Table 3）用表格形式展示了与图 18 相同的信息，并链接到本章的相关小节以及本指南后续部分的更完整文档。

| Unified Memory Paradigm                      | Device Attributes                                            | Full Documentation                                           |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 有限的 unified memory 支持                   | `cudaDevAttrConcurrentManagedAccess` 是 0                    | Unified Memory on Windows, WSL, and Tegra                    |
| 对显式 managed memory allocations 的完全支持 | `cudaDevAttrPageableMemoryAccess` 是 0 并且 `cudaDevAttrConcurrentManagedAccess` 是 1 | Unified Memory on Devices with only CUDA Managed Memory Support |
| 对所有 allocations 的完全支持（软件一致性）  | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 是 0 并且 `cudaDevAttrPageableMemoryAccess` 是 1 并且 `cudaDevAttrConcurrentManagedAccess` 是 1 | Unified Memory on Devices with Full CUDA Unified Memory Support |
| 对所有 allocations 的完全支持（硬件一致性）  | `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 是 1 并且 `cudaDevAttrPageableMemoryAccess` 是 1 并且 `cudaDevAttrConcurrentManagedAccess` 是 1 | Unified Memory on Devices with Full CUDA Unified Memory Support |

##### **2.4.2.1.1. Unified Memory Paradigm: Code Example**

下面的示例代码展示了如何查询 device attributes 并根据 **Figure 18** 的逻辑为系统中的每个 GPU 确定 unified memory paradigm。

```c++
void queryDevices()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices); // 获取设备数量
    for(int i=0; i<numDevices; i++)
    {
        cudaSetDevice(i); // 设定当前 device
        cudaInitDevice(0, 0, 0); // 初始化 device
        int deviceId = i;

        int concurrentManagedAccess = -1;     
        cudaDeviceGetAttribute (&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId);    
        int pageableMemoryAccess = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId);
        int pageableMemoryAccessUsesHostPageTables = -1;
        cudaDeviceGetAttribute (&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId);

        printf("Device %d has ", deviceId);
        if(concurrentManagedAccess){
            if(pageableMemoryAccess){
                printf("full unified memory support");
                if( pageableMemoryAccessUsesHostPageTables)
                    { printf(" with hardware coherency\n");  }
                else
                    { printf(" with software coherency\n"); }
            }
            else
                { printf("full unified memory support for CUDA-made managed allocations\n"); }
        }
        else
        {   printf("limited unified memory support: Windows, WSL, or Tegra\n");  }
    }
}
```

#### 2.4.2.2. Full Unified Memory Feature Support

大多数 Linux 系统具有完整的 unified memory 支持。如果设备属性 `cudaDevAttrPageableMemoryAccess` 是 1，那么所有系统内存——不论是通过 CUDA APIs 分配的还是通过系统 API 分配的——都会作为具有完整特性支持的 unified memory 来操作。这也包括通过 `mmap` 创建的文件支持内存分配。

如果 `cudaDevAttrPageableMemoryAccess` 是 0，那么只有通过 CUDA 分配成 managed memory 的内存会表现为 unified memory。通过系统 API 分配的内存不是 managed 内存，并且不一定可以从 GPU kernels 访问。

一般来说，对于具有完整支持的 unified 内存分配：

- Managed memory 通常会在首次使用它的处理器的内存空间中分配
- 当 managed memory 被不同于当前所在处理器的处理器使用时，它通常会发生迁移
- Managed memory 的迁移或访问以内存页（软件一致性）或 cache line（硬件一致性）的粒度进行
- 支持 oversubscription：应用程序可以分配比 GPU 物理可用更大的 managed memory

分配和迁移行为可能会偏离上述情况。这些行为可以通过程序员使用 hints 和 prefetches 来影响。完整的统一内存支持内容可在 *Unified Memory on Devices with Full CUDA Unified Memory Support* 中找到。

##### 2.4.2.2.1. 具有硬件一致性的完整统一内存

在例如 Grace Hopper 和 Grace Blackwell 这样的硬件上，当使用 NVIDIA CPU 并且 CPU 与 GPU 之间通过 NVLink Chip-to-Chip (C2C) 互连时，可用 address translation services（ATS）。当 ATS 可用时，设备属性 `cudaDevAttrPageableMemoryAccessUsesHostPageTables` 为 1。

在有 ATS 的情况下，除了对所有 host allocations 提供完整的 unified memory 支持外：

- GPU allocations（例如 `cudaMalloc`）可以从 CPU 访问（此时 `cudaDevAttrDirectManagedMemAccessFromHost` 将为 1）
- CPU 与 GPU 之间的链路支持原生 atomics（此时 `cudaDevAttrHostNativeAtomicSupported` 将为 1）
- 与软件一致性相比，硬件一致性的支持可以提升性能

ATS 提供了 HMM 的全部能力。当 ATS 可用时，HMM 会自动禁用。关于硬件与软件一致性的进一步讨论可参见 *CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency*。

##### 2.4.2.2.2. HMM — 具有软件一致性的完整统一内存

Heterogeneous Memory Management（HMM）是 Linux 操作系统上一项可用的特性（需要合适的 kernel 版本），它提供了基于软件一致性的完整 unified memory 支持。Heterogeneous memory management 将 ATS 提供给 PCIe-connected GPUs 的部分能力和便利性引入到了这一场景中。

在至少运行 Linux Kernel 6.1.24、6.2.11，或 6.3 及更高版本的 Linux 系统上，可能支持 heterogeneous memory management（HMM）。可以使用下面的命令来检查 addressing mode 是否为 `HMM`。

```
$ nvidia-smi -q | grep Addressing
Addressing Mode : HMM
```

当 HMM 可用时，就支持完整的 unified memory，所有系统分配都会隐式成为 unified memory。如果系统还具有 ATS，那么 HMM 会被禁用，而使用 ATS，因为 ATS 相对于 HMM 提供了更多能力。

#### 2.4.2.3. 有限的统一内存支持

在 Windows（包括 Windows Subsystem for Linux，WSL）以及某些 Tegra 系统上，只提供 unified memory 功能的一个有限子集。在这些系统上，managed memory 是可用的，但 CPU 与 GPU 之间的迁移行为有所不同。

- Managed memory 首先分配在 CPU 的物理内存中
- Managed memory 的迁移粒度大于虚拟内存页
- 当 GPU 开始执行时，managed memory 会迁移到 GPU
- 当 GPU 处于活动状态时，CPU 不能访问 managed memory
- 当 GPU 被 synchronize 之后，managed memory 会迁移回 CPU
- 不允许对 GPU memory 进行 oversubscription
- 只有通过 CUDA 显式分配为 managed memory 的内存才是 unified 的

关于这一模型的完整说明可参见 *Unified Memory on Windows, WSL, and Tegra*。

#### 2.4.2.4. Memory Advise 和 Prefetch

程序员可以向负责管理 unified memory 的 NVIDIA Driver 提供 hints，以帮助其最大化应用程序性能。CUDA API `cudaMemAdvise` 允许程序员指定内存分配的属性，这些属性会影响内存的放置位置，以及当内存被其他 device 访问时是否发生迁移。

`cudaMemPrefetchAsync` 允许程序员建议对某个特定分配启动一次到不同位置的异步迁移。一个常见的用法是在 kernel launch 之前，提前开始传输 kernel 将要使用的数据。这使得数据拷贝可以在其他 GPU kernels 正在执行的同时进行。

*Performance Hints* 一节介绍了可以传递给 `cudaMemAdvise` 的不同 hints，并展示了使用 `cudaMemPrefetchAsync` 的示例。

### 2.4.3. Page-Locked Host Memory（页面锁定的 Host 内存）

在 **入门代码示例** 中，使用 `cudaMallocHost` 在 CPU 上分配内存。这个函数在 host 端分配的是页面锁定内存（也称为 pinned memory）。通过传统分配机制（如 `malloc`、`new` 或 `mmap`）分配的主机内存不是页面锁定的，这意味着这些内存可能被操作系统换出到磁盘或发生物理重定位。

页面锁定的 host 内存是 CPU 和 GPU 之间进行异步拷贝 所必需的。页面锁定内存也能提升同步拷贝的性能。页面锁定的内存可以映射到 GPU，使 GPU kernels 能直接访问它。

CUDA runtime 提供了一些 API 来分配页面锁定的 host 内存或将已有内存设置为页面锁定：

- `cudaMallocHost` 分配页面锁定的 host 内存
- `cudaHostAlloc` 默认行为与 `cudaMallocHost` 相同，同时可以通过 flags 指定其它内存参数
- `cudaFreeHost` 释放由 `cudaMallocHost` 或 `cudaHostAlloc` 分配的内存
- `cudaHostRegister` 将在 CUDA API 之外（比如用 `malloc` 或 `mmap` 分配的）内存区域设置为页面锁定

使用 `cudaHostRegister` 可以让第三方库分配的 host 内存或其他由开发者无法控制的代码分配的内存变成页面锁定内存，以便用于异步拷贝或映射。

> **注意**
>
> 页面锁定的 host 内存可被系统中任意 GPU 用于异步拷贝和映射内存。
>
> 在非 I/O 一致性的 Tegra 设备上页面锁定的 host 内存不会被缓存。同时，`cudaHostRegister()` 在非 I/O 一致性的 Tegra 设备上不受支持。

#### 2.4.3.1. Mapped Memory（映射内存）

在支持 HMM 或 ATS 的系统上，所有 host 内存都可以使用 host 指针直接被 GPU 访问。当 ATS 或 HMM 不可用时，可以通过将主机内存映射到 GPU 的内存空间，使 host 分配的内存可被 GPU 访问。映射内存总是页面锁定的。

下面的代码示例展示了一个操作映射的 host 内存的数组拷贝 kernel。

```
__global__ void copyKernel(float* a, float* b)
{
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        a[idx] = b[idx];
}
```

虽然在某些情况下映射内存对需要从 kernel 访问但不拷贝到 GPU 的数据是有用的，但在 kernel 中访问映射内存需要通过 CPU-GPU 互连（如 PCIe 或 NVLink C2C）进行传输。与访问设备内存相比，这些传输具有更高的延迟和更低的带宽。在大多数 kernel 的内存需求中，映射内存不应该被视为比 unified memory 或 显式的内存管理 更高效的选择。

##### 2.4.3.1.1. cudaMallocHost 和 cudaHostAlloc

**通过 `cudaHostMalloc` 或 `cudaHostAlloc` 分配的 host 内存会自动映射。** 这些 API 返回的指针可以在 kernel 代码中直接用于访问 host 内存。host 内存通过 CPU-GPU 互连进行访问。

```c++
// cudaMallocHost
    void usingMallocHost() {
      float* a = nullptr;
      float* b = nullptr;
      CUDA_CHECK(cudaMallocHost(&a, vLen*sizeof(float)));
      CUDA_CHECK(cudaMallocHost(&b, vLen*sizeof(float)));

      initVector(b, vLen);
      memset(a, 0, vLen*sizeof(float));

      int threads = 256;
      int blocks = vLen/threads;
      copyKernel<<<blocks, threads>>>(a, b);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      printf("Using cudaMallocHost: ");
      checkAnswer(a,b);
    }

// cudaAllocHost
    void usingCudaHostAlloc() {
      float* a = nullptr;
      float* b = nullptr;

      CUDA_CHECK(cudaHostAlloc(&a, vLen*sizeof(float), cudaHostAllocMapped));
      CUDA_CHECK(cudaHostAlloc(&b, vLen*sizeof(float), cudaHostAllocMapped));

      initVector(b, vLen);
      memset(a, 0, vLen*sizeof(float));

      int threads = 256;
      int blocks = vLen/threads;
      copyKernel<<<blocks, threads>>>(a, b);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      printf("Using cudaAllocHost: ");
      checkAnswer(a, b);
    }
```

##### 2.4.3.1.2. cudaHostRegister

当 ATS 和 HMM 不可用时，使用系统分配器分配的内存仍然可以通过 `cudaHostRegister` 映射，以便 GPU kernels 直接访问。不过，与通过 CUDA API 创建的内存不同，这种内存不能通过 host 指针直接在 kernel 中访问。必须通过 `cudaHostGetDevicePointer()` 获得对应的 device 内存区域的指针，然后在 kernel 代码中使用该 device 指针进行访问。

```c++
void usingRegister() {
  float* a = nullptr;
  float* b = nullptr;
  float* devA = nullptr;
  float* devB = nullptr;

  a = (float*)malloc(vLen*sizeof(float));
  b = (float*)malloc(vLen*sizeof(float));
  CUDA_CHECK(cudaHostRegister(a, vLen*sizeof(float), 0 ));
  CUDA_CHECK(cudaHostRegister(b, vLen*sizeof(float), 0  ));

  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devA, (void*)a, 0));
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&devB, (void*)b, 0));
  initVector(b, vLen);
  memset(a, 0, vLen*sizeof(float));

  int threads = 256;
  int blocks = vLen/threads;
  copyKernel<<<blocks, threads>>>(devA, devB);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("Using cudaHostRegister: ");
  checkAnswer(a, b);
}
```

##### 2.4.3.1.3. 比较 Unified Memory 和 Mapped Memory

**Mapped memory 使 GPU 可以访问 CPU 内存，但不能保证所有访问类型（例如 atomics）在所有系统上都受到支持。Unified memory 则保证所有访问类型都被支持。**

Mapped memory 保留在 CPU 内存中，这意味着所有 GPU 的访问必须经过 CPU 和 GPU 之间的连接：PCIe 或 NVLink。从这些链路进行访问的延迟显著高于访问 GPU memory，且总的可用带宽更低。因此，在所有 kernel memory 访问上使用 mapped memory 不太可能充分利用 GPU 的计算资源。

Unified memory 通常会迁移到正在访问它的 processor 的物理 memory 中。在首次迁移之后，kernel 对同一个 memory page 或 cache line 的重复访问可以利用完整的 GPU memory 带宽。

> 注意
>
> **在以前的文档中，mapped memory 也被称为 zero-copy memory。**
>
> 在所有 CUDA 应用使用统一虚拟地址空间之前，需要额外的 APIs 来启用 memory mapping（例如使用 `cudaSetDeviceFlags` 和 `cudaDeviceMapHost`）。这些 APIs 现在不再需要。
>
> 对 mapped host memory 执行的 atomic functions（参见 Atomic Functions）从 host 或其他 GPUs 的角度来看不是原子操作。
>
> CUDA runtime 要求从 device 发起的对 host memory 的 1-byte、2-byte、4-byte、8-byte 和 16-byte 自然对齐的 loads 和 stores 必须被保留为从 host 和其他 devices 的角度看是单次访问。在某些平台上，对 memory 的 atomics 可能被硬件拆分成单独的 load 和 store 操作。这些组成的 load 和 store 操作具有相同的自然对齐访问保留要求。CUDA runtime 不支持 PCI Express 总线拓扑结构中有 PCI Express 桥拆分 8-byte 自然对齐操作的情况，并且 NVIDIA 也不知道有任何拓扑会拆分 16-byte 自然对齐操作。

### 2.4.4. 总结

- 在支持 HMM 或 ATS 的 Linux 平台上，所有由系统分配的 memory 都是 managed memory
- 在不支持 HMM 或 ATS 的 Linux 平台上、在 Tegra processors 上，以及在所有 Windows 平台上，managed memory 必须通过 CUDA 进行分配：
  - 使用 `cudaMallocManaged`，或
  - 使用 `cudaMallocFromPoolAsync`，并且 pool 需要以 `allocType=cudaMemAllocationTypeManaged` 创建
  - 使用带有 `__managed__` specifier 的 global variables
- 在 Windows 和 Tegra processors 上，unified memory 存在一定限制
- 在支持 ATS 的 NVLINK C2C 连接系统上，通过 `cudaMalloc` 分配的 device memory 可以被 CPU 或其他 GPUs 直接访问
