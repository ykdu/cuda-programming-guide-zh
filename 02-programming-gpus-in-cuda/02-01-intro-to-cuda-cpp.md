# 2.1 CUDA C++简介

本章通过说明如何在 C++ 中暴露 CUDA 编程模型的一些基本概念，来介绍 CUDA 的基础知识。

本编程指南主要介绍 CUDA runtime API。CUDA runtime API 是最常用的在 C++ 中使用 CUDA 的方式，它是在较低级别的 CUDA driver API 上构建的。

[CUDA runtime API 和 CUDA driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 讨论了这两种 API 之间的区别，[CUDA driver API](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 则讨论了如何编写混合使用这两种 API 的代码。

本指南假定已安装 CUDA 工具包和 NVIDIA 驱动，并且计算机上存在受支持的 NVIDIA GPU。有关安装所需 CUDA 组件的说明，请参见 [CUDA 快速入门指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)。

### 2.1.1. 使用 NVCC 编译

在 C++ 中编写的 GPU 代码通过 NVIDIA Cuda 编译器 `nvcc` 编译。`nvcc` 是一个编译器驱动程序，它简化了编译 C++ 或 PTX 代码的过程：它提供了简单且熟悉的命令行选项，并通过调用实现不同编译阶段的工具来执行这些选项。

本指南将展示可在安装 CUDA 工具包的任何 Linux 系统、Windows 命令行或 PowerShell，或在 Windows 子系统 Linux (WSL) 上使用的 `nvcc` 命令行。有关 `nvcc` 的常见用法，请参见本指南的 [nvcc 章节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)，完整文档请参阅 [nvcc 用户手册](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)。

### 2.1.2. Kernels（kernel）

如在 [CUDA 编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 介绍中所提到的，GPU 上执行的函数可以从 host 调用的被称为 kernel。kernels 是为许多线程并行执行而编写的函数。

#### 2.1.2.1. 指定 Kernels

kernel 的代码通过 `__global__` 声明说明符来指定。这告诉编译器该函数将为 GPU 编译，以便可以从 kernel 启动中调用。kernel 启动是指从 host 启动 kernel 执行的操作。kernels 是具有 `void` 返回类型的函数。

```
// Kernel 定义
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

#### 2.1.2.2. 启动 Kernels

执行 kernel 的线程数量通过 kernel 启动时的执行配置来指定。使用相同的 kernel 的不同调用可以使用不同的执行配置，如不同数量的线程或线程块。

有两种方式可以从 host 代码启动 kernels，分别是三重尖括号符号（triple chevron notation）和 `cudaLaunchKernelEx`。三重尖括号符号是启动 kernels 的最常见方式，本节介绍了它。使用 `cudaLaunchKernelEx` 启动 kernel 的示例将在第 [3.1.1 节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 中展示并详细讨论。

##### 2.1.2.2.1. 三重尖括号符号（Triple Chevron Notation）

三重尖括号符号是 [CUDA C++ 语言扩展](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)，用于启动 kernels。之所以称其为三重尖括号，是因为它使用三个尖括号字符包裹 kernel 启动的执行配置，即 `<<< >>>`。执行配置参数通过逗号分隔列表的形式指定，类似于函数调用的参数。以下是使用三重尖括号符号启动 `vecAdd` kernel 的语法示例。

```
__global__ void vecAdd(float* A, float* B, float* C)
{

}

int main()
{
    ...
    // Kernel 调用
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

三重尖括号符号的前两个参数分别是 grid 和thread block 维度。当使用一维thread block或grid时，可以使用整数来指定维度。

上面的代码启动了一个包含 256 个线程的单个thread block。每个线程都会执行完全相同的 kernel 代码。在 [Thread and Grid Index Intrinsics](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 中，我们将展示如何使用每个线程在线程块和网格中的索引来更改它操作的数据。

线程块的线程数是有限制的，因为所有线程都驻留在同一个 streaming multiprocessor (SM) 中，并且必须共享 SM 的资源。在当前的 GPU 上，单个线程块最多可以包含 1024 个线程。如果资源允许，可以将多个线程块同时调度到一个 SM 上。

Kernel 启动与 host 线程是异步的。也就是说，kernel 将被设置好在 GPU 上执行，但 host 代码不会等待 kernel 完成（甚至不等待 kernel 开始）后再继续执行。必须使用某种形式的同步机制来确定 kernel 是否已经完成。最基本的同步方式是完全同步整个 GPU，示例在 [同步 CPU 和 GPU](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 中展示。更复杂的同步方法在 [异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html) 中有所介绍。

当使用二维或三维网格或线程块时，CUDA 类型 `dim3` 用作网格和线程块维度的参数。下面的代码片段展示了使用 16x16 的线程块网格，且每个线程块为 8x8 启动 `MatAdd` kernel 的示例。

```
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```

#### 2.1.2.3. Thread and Grid Index Intrinsics

在 kernel 代码内部，CUDA 提供了一些 intrinsics 用来访问执行配置的参数以及 thread 或 block 的索引。

> - `threadIdx` 表示一个 thread 在其所属的 thread block 中的索引。一个 thread block 中的每个 thread 都有不同的索引。
> - `blockDim` 表示 thread block 的维度，它是在 kernel launch 的 execution configuration 中指定的。
> - `blockIdx` 表示 thread block 在 grid 中的索引。每个 thread block 都有不同的索引。
> - `gridDim` 表示 grid 的维度，它是在 kernel launch 时的 execution configuration 中指定的。

这些 intrinsics 都是一个具有 `.x`、`.y`、和 `.z` 成员的 3 分量向量。launch configuration 中未指定的维度默认值是 1。`threadIdx` 和 `blockIdx` 是从 0 开始索引的。也就是说，`threadIdx.x` 的取值范围是从 0 到包括 `blockDim.x‑1`。`.y` 和 `.z` 在对应维度上的行为也是一样的。

同样地，`blockIdx.x` 的取值范围是从 0 到包括 `gridDim.x‑1`，对应的 `.y` 和 `.z` 维度也是如此。

这些 intrinsics 允许单个 thread 确定它应该执行的工作。回到上面的 `vecAdd` kernel，该 kernel 有三个参数，每个都是 float 向量。kernel 并行化使得每个 thread 执行一次加法。它计算哪一个元素是由该 thread 的 thread 和 grid 索引决定的。

```
__global__ void vecAdd(float* A, float* B, float* C)
{
   // 计算当前 thread 负责处理的元素索引
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x

   // 执行计算
   C[workIndex] = A[workIndex] + B[workIndex];
}
int main()
{
    ...
    // A、B 和 C 是具有 1024 个元素的向量
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

在这个例子中，使用了 4 个 thread blocks，每个 block 256 个 threads 来对一个 1024 元素的向量求和。在第一个 thread block 中，`blockIdx.x` 为 0，因此每个 thread 的 workIndex 就是其 `threadIdx.x`。在第二个 thread block 中，`blockIdx.x` 为 1，因此 `blockDim.x * blockIdx.x` 等于 `blockDim.x`，在本例中是 256。第二个 thread block 中每个 thread 的 `workIndex` 就是其 `threadIdx.x + 256`。在第三个 thread block 中 `workIndex` 是 `threadIdx.x + 512`。

这种 `workIndex` 的计算在一维并行化中非常常见。扩展到二维或三维时，通常在各个维度上采用相同的模式。

##### 2.1.2.3.1. Bounds Checking

上面给出的示例假设向量长度恰好是 thread block 大小的整数倍，本例中是 256 个 threads。为了让 kernel 能处理任意向量长度，我们可以如下面所示添加检查以确保内存访问不会越界，然后再 launch 一个 thread block，其中可能有一些 inactive threads。

```
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // 计算当前 thread 负责处理的元素索引
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // 执行计算
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

使用上面的 kernel 代码，可以 launch 多余的 threads 而不会导致对数组的越界访问。当 `workIndex` 超过 `vectorLength` 时，这些 threads 会退出并且不执行任何工作。在一个 block 中 launch 超过需要数量的 threads 并不会带来很大的开销，但应该避免 launch 那些完全没有任何 thread 执行工作的 thread blocks。这样修改后的 kernel 可以处理不是 block 大小整数倍的向量长度。

所需 thread blocks 的数量可以计算为所需 threads 数量（这里是向量长度）除以每个 block 的 threads 数量的上取整值。也就是说，把所需 threads 数量除以每个 block 的 threads 数量并向上取整。下面给出了一个常见的用单个整数除法表达这种上取整的方法。通过在整数除法前加上 `threads‑1`，这就表现得像一个 ceiling 函数，当向量长度不是线程数的整数倍时才增加一个 thread block。

```
// vectorLength 是一个存储向量元素数量的整数
int threads = 256;
int blocks = (vectorLength + threads‑1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

CUDA 核心计算库（CCCL）提供了一个方便的工具，`cuda::ceil_div`，用来执行此类上取整除法来计算 kernel 启动所需的 block 数量。使用时，只需要包含头文件 `<cuda/cmath>`。

```
// vectorLength 是一个存储向量元素数量的整数
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

这里选择 256 个 threads 每个 block 是拍脑袋设定的，但这是一个很常用的起始值。

### 2.1.3. Memory in GPU Computing（GPU 计算中的内存）

为了使用上面展示的 `vecAdd` kernel，数组 `A`、`B` 和 `C` 必须位于 GPU 可以访问的内存中。有几种不同的方法可以做到这一点，这里将展示其中的两种。其它方法将在后面的 **unified memory** 章节中介绍。GPU 上运行的代码可用的 memory spaces 已在 **GPU Memory** 中介绍，并在 **GPU Device Memory Spaces** 中有更详细的说明。

#### 2.1.3.1. Unified Memory

Unified memory 是 CUDA runtime 的一项特性，它让 NVIDIA Driver 管理 host 和 device 之间的数据移动。使用 `cudaMallocManaged` API 分配内存或者用 `__managed__` 声明变量时，就可以启用这项特性。无论是 GPU 还是 CPU 尝试访问该内存，NVIDIA Driver 都会确保该内存在二者之间可访问。

下面代码展示了一个完整的函数，用 unified memory 分配将用于 GPU 的输入和输出向量，并启动 `vecAdd` kernel。`cudaMallocManaged` 分配的缓冲区既可以从 CPU 访问，也可以从 GPU 访问。使用完这些缓冲区后用 `cudaFree` 释放。

```
void unifiedMemExample(int vectorLength)
{
    // 指向内存向量的指针
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // 使用 unified memory 分配缓冲区
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));
    // 在 host 上初始化向量
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // 启动 kernel。Unified memory 会确保 A、B 和 C 对 GPU 可访问
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // 等待 kernel 执行完成
    cudaDeviceSynchronize();
    // 在 CPU 上串行执行计算以作比较
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // 确认 CPU 和 GPU 得到相同结果
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }
    // 清理
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
```

Unified memory 在所有受 CUDA 支持的操作系统和 GPUs 上都支持，但底层机制和性能可能因系统架构不同而有所差异。有关更多细节，请参见 **Unified Memory**。在某些 Linux 系统上（例如具有 **address translation services** 或 **heterogeneous memory management** 的系统），所有系统内存自动就是 unified memory，无需使用 `cudaMallocManaged` 或 `__managed__`。

#### 2.1.3.2. Explicit Memory Management（显式内存管理）

显式管理内存分配和不同 memory spaces 之间的数据迁移可以帮助提高应用性能，尽管这会使代码更冗长。下面的代码演示了如何使用 `cudaMalloc` 在 GPU 上显式分配内存。GPU 上的内存用和前面 unified memory 示例中相同的 `cudaFree` API 释放。

```
void explicitMemExample(int vectorLength)
{
    // host 内存指针
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // device 内存指针
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;
    // 使用 cudaMallocHost API 分配 host 内存。这是在 host 与 GPU 之间进行内存复制时的最佳实践
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // 在 host 上初始化向量
    initArray(A, vectorLength);
    initArray(B, vectorLength);
    // 开始分配和复制
    // 在 GPU 上分配内存
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));
    // 将数据复制到 GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // 分配和复制结束
    // 启动 kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // 等待 kernel 执行完成
    cudaDeviceSynchronize();

    // 将结果复制回 host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // 在 CPU 上串行执行计算以作比较
    serialVecAdd(A, B, comparisonResult, vectorLength);
    // 确认 CPU 和 GPU 得到相同结果
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }
    // 清理
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```

CUDA API `cudaMemcpy` 用于将数据从一个驻留在 CPU 上的缓冲区复制到一个驻留在 GPU 上的缓冲区。除了目标指针、源指针和字节大小外，`cudaMemcpy` 的最后一个参数是 `cudaMemcpyKind_t`。这可以是 `cudaMemcpyHostToDevice`（CPU 到 GPU 的复制）、`cudaMemcpyDeviceToHost`（GPU 到 CPU 的复制）、或 `cudaMemcpyDeviceToDevice`（GPU 内部或 GPUs 之间的复制）。

在本示例中，`cudaMemcpyDefault` 作为最后一个参数传递给 `cudaMemcpy`。这会让 CUDA 根据源和目标指针的值来决定要执行哪种类型的复制。

`cudaMemcpy` API 是同步的。也就是说，它在复制完成之前不会返回。有关异步复制的介绍请参见 **Launching Memory Transfers in CUDA Streams**。

这段代码使用 `cudaMallocHost` 在 CPU 上分配内存。这会在 host 上分配 **page‑locked memory**，它可以提高复制性能，并且在进行 **异步** 内存传输时是必要的。通常建议只对将用于与 GPU 之间发送或接收数据的 buffers 进行 page‑locked。**最佳实践是仅对那些将用于与 GPU 发送或接收数据的缓冲区进行 page‑locked 内存分配**。

#### 2.1.3.3. Memory Management and Application Performance（内存管理与应用性能）

如上例所示，显式内存管理更啰嗦，需要程序员显式指定 host 和 device 之间的复制。这既是显式内存管理的优势也是劣势：它让程序员能够控制何时在 host 和 device 之间复制数据、数据驻留在哪个位置，以及精确控制内存的分配位置。显式内存管理可以提供性能上的机会，例如控制内存传输并与其它计算重叠执行。

当使用 unified memory 时，有一些 CUDA APIs（将在 Memory Advise and Prefetch 中介绍）可以向 NVIDIA driver 提供提示，从而在使用 unified memory 时也能够获取一些显式内存管理的性能优势。

### 2.1.4. 同步 CPU 和 GPU

如在 **Launching Kernels** 章节中提到的，kernel 启动是与调用它们的 CPU 线程异步的。这意味着 CPU 线程的控制流将在 kernel 完成之前继续执行，甚至可能在 kernel 启动之前就开始执行。为了保证 kernel 在继续执行主机代码之前已经完成执行，需要一些同步机制。

同步 GPU 和主机线程的最简单方法是使用 `cudaDeviceSynchronize`，它会阻塞主机线程，直到 GPU 上所有之前发出的工作都已完成。在本章中的示例中，由于 GPU 上仅执行单个操作，这种方式已经足够。在较大的应用程序中，可能会有多个流在 GPU 上执行工作，`cudaDeviceSynchronize` 会等待所有流中的工作完成。在这些应用程序中，建议使用 Stream Synchronization API 仅与特定流同步，或者使用 `CUDA Events` 进行同步。有关这些内容的详细介绍将在 Asynchronous Execution 章节中提供。

### 2.1.5. Putting it All Together

代码省略

在这些示例中，所有线程都在执行独立的工作，无需与其他线程协调或同步。线程通常需要合作并与其他线程进行通信，以完成它们的工作。一个 block 内的线程可以通过 shared memory 共享数据，并进行同步以协调内存访问。

**最基本的 block 级别同步机制是 `__syncthreads()` 内建函数，它充当一个屏障，所有线程必须在任何线程被允许继续之前等待**。Shared Memory 章节给出了使用 shared memory 的示例。

为了高效合作，shared memory 被期望是每个处理器核心附近的低延迟内存（类似于 L1 缓存），而 `__syncthreads()` 被期望是轻量级的。`__syncthreads()` 仅同步单个线程 block 内的线程。CUDA 编程模型不支持线程 block 之间的同步。**Cooperative Groups** 提供了设置不同于单个线程 block 的同步域的机制。

通常，当同步仅限于一个线程 block 内时，能获得最佳性能。线程 block 可以使用 **atomic memory functions** 在共同的结果上继续工作，这将在后续章节中介绍。

**3.2.4** 章节介绍了 CUDA 同步原语，这些原语提供了非常精细的控制，以最大化性能和资源利用率。

### 2.1.6. Runtime Initialization

CUDA 运行时为系统中的每个设备创建一个 CUDA context。该 context 是该设备的 primary context，并在第一个需要该设备上活动 context 的运行时函数中初始化。此 context 在应用程序的所有 host 线程之间共享。作为 context 创建的一部分，如果需要，设备代码会进行 **just-in-time compiled** 并加载到 device memory 中。所有这些都在后台透明完成。CUDA runtime创建的 primary context 可以通过驱动 API 进行访问，以实现 **Interoperability between Runtime and Driver APIs**。

从 CUDA 12.0 开始，`cudaInitDevice` 和 `cudaSetDevice` 调用初始化运行时和与指定设备关联的 primary context。运行时将隐式使用设备 0，并在需要时自我初始化以处理运行时 API 请求，如果这些请求发生在这些调用之前。此机制在计时运行时函数调用以及解释第一次调用到运行时的错误代码时很重要。在 CUDA 12.0 之前，`cudaSetDevice` 不会初始化运行时。

`cudaDeviceReset` 会销毁当前设备的 primary context。如果在 primary context 被销毁后调用 CUDA 运行时 API，将会为该设备创建一个新的 primary context。

> **注意**
>  CUDA 接口使用在 host 程序初始化期间初始化的全局状态，并在 host 程序终止时销毁。若在程序初始化或在 main 后终止期间使用任何这些接口（隐式或显式），将会导致未定义的行为。
>
> 从 CUDA 12.0 开始，`cudaSetDevice` 会显式初始化运行时（如果尚未初始化），在更改 host 线程的当前设备后进行。之前版本的 CUDA 在更改到新设备后，直到首次调用运行时函数才会初始化运行时。因此，在调用 `cudaSetDevice` 后，检查 `cudaSetDevice` 的返回值以检测初始化错误非常重要。
>
> 参考手册中的错误处理和版本管理部分的运行时函数不会初始化运行时。











CUDA runtime 为系统中的每个设备创建一个 CUDA context。该 context 是该设备的 primary context，并在第一个需要该设备上活动 context 的 runtime 函数中初始化。此 context 在应用程序的所有 host 线程之间共享。作为 context 创建的一部分，如果需要，设备代码会进行 just-in-time compiled 并加载到 device memory 中。所有这些都在后台透明完成。CUDA runtime 创建的 primary context 可以通过 driver API 进行访问，以实现 **Interoperability between Runtime and Driver APIs**。

**从 CUDA 12.0 开始，`cudaInitDevice` 和 `cudaSetDevice` 会初始化 runtime 和与指定设备关联的 primary context**。runtime 将隐式使用设备 0，并在需要时自我初始化以处理 runtime API 请求，如果这些请求发生在这些调用之前。此机制在计时 runtime 函数调用以及解释第一次调用到 runtime 的错误代码时很重要。**在 CUDA 12.0 之前，`cudaSetDevice` 不会初始化 runtime**。

`cudaDeviceReset` 会销毁当前设备的 primary context。如果在 primary context 被销毁后调用 CUDA runtime API，将会为该设备创建一个新的 primary context。

> **注意**
>  CUDA 接口使用在 host 程序初始化期间初始化的全局状态，并在 host 程序终止时销毁。若在程序初始化或在 main 后终止期间使用任何这些接口（隐式或显式），将会导致未定义的行为。
>
> 从 CUDA 12.0 开始，`cudaSetDevice` 会显式初始化 runtime（如果尚未初始化），在更改 host 线程的当前设备后进行。之前版本的 CUDA 在更改到新设备后，直到首次调用 runtime 函数才会初始化 runtime。因此，在调用 `cudaSetDevice` 后，检查 `cudaSetDevice` 的返回值以检测初始化错误非常重要。
>
> 错误处理和版本管理相关的 runtime APIs不会初始化 runtime。

### 2.1.7. Error Checking in CUDA

每个 CUDA API 都会返回一个枚举类型 `cudaError_t`。在示例代码中，这些错误通常不会被检查。在生产应用中，检查并处理每个 CUDA API 调用的返回值是最佳实践。当没有错误时，返回值是 `cudaSuccess`。许多应用会实现一个实用的宏，如下面所示：

```
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```

这个宏使用了 `cudaGetErrorString` API，它返回一个可读的字符串来描述特定 `cudaError_t` 值的含义。使用上面的宏时，应用程序可以用 `CUDA_CHECK(expression)` 来调用 CUDA runtime API，如下面所示：

```
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

如果这些调用中的任何一个检测到错误，它就会使用这个宏打印到 `stderr`。这个宏在较小的项目中很常见，但在较大的应用中可以适配成日志系统或其他错误处理机制。

> **注**
>
> 需要注意的是，任何 CUDA API 调用返回的错误状态也可能表示先前发起的异步操作出现了错误。关于这一点的更详细内容，请参见章节 *Asynchronous Error Handling*。

#### 2.1.7.1. Error State

CUDA runtime 为每个 host 线程维护一个 `cudaError_t` 状态。该值默认为 `cudaSuccess`，并在发生错误时被覆盖。`cudaGetLastError` 返回当前错误状态，然后将其重置为 `cudaSuccess`。另一种方式 `cudaPeekLastError` 返回错误状态但不重置它。

使用三重尖括号语法启动的 kernel 不会返回 `cudaError_t`。在 kernel 启动后立即检查错误状态是一个良好实践，用于检测 kernel 启动时的直接错误或在 kernel 启动之前的异步错误。在 kernel 启动后立即检查错误状态时返回 `cudaSuccess` 并不意味着 kernel 已成功执行或甚至开始执行。它仅验证传递给 runtime 的 kernel 启动参数和 execution configuration 没有触发任何错误，并且错误状态不是启动 kernel 之前的先前错误或异步错误。

#### 2.1.7.2. Asynchronous Errors

CUDA kernel 启动和许多 runtime APIs 是异步的。异步 CUDA runtime APIs 会在章节 *Asynchronous Execution* 中详细讨论。CUDA 错误状态在发生错误时会被设置并覆盖。这意味着在异步操作执行期间发生的错误只有在下次检查错误状态时才会报告。如前所述，这可能是调用 `cudaGetLastError`、`cudaPeekLastError`，或者任何返回 `cudaError_t` 的 CUDA API。

当 CUDA runtime API 函数返回错误时，错误状态不会被清除。这意味着来自异步错误的错误码（例如 kernel 的无效内存访问）将被每个 CUDA runtime API 返回，直到错误状态通过调用 `cudaGetLastError` 被清除。

```c++
vecAdd<<<blocks, threads>>>(devA, devB, devC);
// 检查 kernel 启动后的错误状态
CUDA_CHECK(cudaGetLastError());
// 等待 kernel 执行完成
// CUDA_CHECK 将报告在 kernel 执行过程中发生的错误
CUDA_CHECK(cudaDeviceSynchronize());
```

> **注**
>  `cudaError_t` 值 `cudaErrorNotReady`，它可能由 `cudaStreamQuery` 和 `cudaEventQuery` 返回，但不被视为错误，并且不会通过 `cudaPeekAtLastError` 或 `cudaGetLastError` 报告。

#### 2.1.7.3. CUDA_LOG_FILE

另一种识别 CUDA 错误的好方法是使用 `CUDA_LOG_FILE` 环境变量。当设置此环境变量时，CUDA 驱动程序会将遇到的错误消息写入指定路径的文件中。比如，考虑下面这段错误的 CUDA 代码，它尝试启动一个线程块，其大小超出了任何架构支持的最大值。

```c++
__global__ void k() { }
int main() {
    k<<<8192, 4096>>>(); // 无效的块大小
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
```

构建并运行时，kernel 启动后的检查将使用第 2.1.7 节中示例的宏来检测并报告错误。

```bash
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
```

然而，当应用程序在设置 `CUDA_LOG_FILE` 为文本文件时运行，文件中包含更多有关错误的信息。

```bash
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754888][CUDA][E] One or more of block dimensions of (4096,1,1) exceeded the maximum supported by the architecture.
[12:46:23.854][137216133754888][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel.
```

将 `CUDA_LOG_FILE` 设置为 `stdout` 或 `stderr` 将打印到标准输出和标准错误。

通过使用 `CUDA_LOG_FILE` 环境变量，可以捕获和识别 CUDA 错误，即使应用程序未实现对 CUDA 返回值的错误检查。此方法对调试非常有帮助，因为环境变量允许应用程序在运行时处理和恢复 CUDA 错误。**CUDA 的错误日志管理功能还允许注册一个回调函数，驱动程序将在发生错误时调用此函数。**这可用于捕获和处理运行时的错误，并将 CUDA 错误无缝集成到现有的日志系统中。

第 4.8 节展示了更多关于 CUDA 错误日志管理的示例。错误日志管理和 `CUDA_LOG_FILE` 可在 NVIDIA Driver 版本 r570 及更高版本中使用。

### 2.1.8. Device and Host Functions

`__global__` 修饰符用于指示 kernel 的入口点。也就是说，这是一个将被用于 GPU 并行执行的函数。通常，kernel 是从主机启动的，但也可以通过动态并行性（dynamic parallelism）在另一个 kernel 内启动 kernel。

`__device__` 修饰符表示一个函数应该被编译为 GPU 代码，并且可以从其他 `__device__` 或 `__global__` 函数调用。一个函数（包括类成员函数、函数指针和 lambda 表达式）可以同时被指定为 `__device__` 和 `__host__`，如下所示的例子。

### 2.1.9. Variable Specifiers

CUDA 修饰符可以用于静态变量声明，以控制变量的存储位置。

- `__device__` 指定一个变量存储在 **Global Memory**（全局内存）
- `__constant__` 指定一个变量存储在 **Constant Memory**（常量内存）
- `__managed__` 指定一个变量存储在 **Unified Memory**（统一内存）
- `__shared__` 指定一个变量存储在 **Shared Memory**（共享内存）

当一个变量在 `__device__` 或 `__global__` 函数内部声明时，如果可能，它会分配到寄存器，如果需要则分配到本地内存。任何没有修饰符的变量（如果它在 `__device__` 或 `__global__` 函数外声明）会分配到系统内存。

#### 2.1.9.1. 编译时检测设备

当一个函数被指定为 `__host__ __device__` 时，编译器会生成 GPU 和 CPU 代码。这类函数可能希望使用预处理器来分别指定只适用于 GPU 或 CPU 代码的部分。**检查 `__CUDA_ARCH__`** 是否定义是实现这一点的常见方法，如下面的示例所示。

### 2.1.10. Thread Block Clusters

从 compute capability 9.0 开始，CUDA 编程模型引入了一个可选的层级，称为 thread block clusters，它由多个 thread blocks 组成。类似于一个 thread block 中的 threads 被保证在同一个 streaming multiprocessor 上协同调度，cluster 中的 thread blocks 也被保证在 GPU 中同一个 GPU Processing Cluster（GPC）上协同调度。

与 thread blocks 类似，clusters 也被组织成一维、二维或三维的 thread block cluster grid，如 Figure 5 所示。

一个 cluster 中的 thread blocks 数量可以由用户定义，在 CUDA 中支持的可移植 cluster size 上限为每个 cluster 最多 8 个 thread blocks。需要注意的是，在 GPU hardware 或 MIG 配置中过小、无法支持 8 个 multiprocessors 的情况下，最大 cluster size 会相应降低。对这些较小配置的识别，以及对支持超过 8 个 thread blocks 的更大配置的识别，都与具体架构相关，可以通过 `cudaOccupancyMaxPotentialClusterSize` API 进行查询。

cluster 中的所有 thread blocks 都被保证在同一个 GPU Processing Cluster（GPC）上同时执行，并且允许 cluster 中的 thread blocks 使用 cooperative groups API `cluster.sync()` 进行硬件支持的同步。cluster group 还提供了成员函数，可分别通过 `num_threads()` 和 `num_blocks()` API 查询 cluster group 中的 thread 数量和 block 数量。cluster group 中 thread 或 block 的 rank 可以分别通过 `dim_threads()` 和 `dim_blocks()` API 查询。

属于同一个 cluster 的 thread blocks 可以访问 distributed shared memory，它是 cluster 内所有 thread blocks 的 shared memory 的组合。cluster 中的 thread blocks 可以对 distributed shared memory 中的任意地址进行读取、写入以及原子操作。*Distributed Shared Memory* 一节给出了在 distributed shared memory 中执行 histogram 的示例。

> **Note**
>  在使用 cluster 支持启动的 kernel 中，`gridDim` 变量仍然表示 thread blocks 的数量，以保持兼容性。cluster 中 block 的 rank 可以通过 Cooperative Groups API 获取。

#### 2.1.10.1. 使用三重尖括号语法启动集群

thread block cluster 可以通过两种方式在 kernel 中启用：一种是使用编译期 kernel attribute `__cluster_dims__(X, Y, Z)`，另一种是使用 CUDA kernel launch API `cudaLaunchKernelEx`。下面的示例展示了如何使用编译期 kernel attribute 启动一个 cluster。使用 kernel attribute 时，cluster size 在编译期固定，随后 kernel 可以使用传统的 `<<< , >>>` 语法进行 launch。如果 kernel 使用的是编译期 cluster size，则在 launch kernel 时无法修改 cluster size。

```c++
// kernel 定义
// 编译期 cluster size：X 维度为 2，Y 和 Z 维度为 1
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{
}

int main()
{
    float *input, *output;
    // 使用编译期 cluster size 进行 kernel 启动
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // gridDim 不会受到 cluster launch 的影响
    // 仍然按 block 数量进行枚举
    // gridDim 必须是 cluster size 的整数倍
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```
