# 1.2 Programming Model

> 原文：CUDA Programming Guide v13.1  
> https://docs.nvidia.com/cuda/cuda-programming-guide/index.html





本章从高层次、与具体语言无关的角度介绍 CUDA programming model。这里引入的术语和概念适用于任何支持的 CUDA 编程语言。后续章节将用 C++ 来说明这些概念。

### 1.2.1. Heterogeneous Systems

CUDA programming model 假设一个 heterogeneous computing system，也就是既包括 GPUs 又包括 CPUs 的系统。CPU 及其直接连接的内存称为 host 和 host memory；GPU 及其直接连接的内存称为 device 和 device memory。在某些 SoC 系统中，它们可能集成在同一个封装内；在更大的系统里可能有多个 CPU 或多个 GPU。

CUDA 应用会在 GPU 上执行部分代码，但应用总是从 CPU 上开始执行。运行在 CPU 的代码称为 host code，它可以调用 CUDA API 在 host memory 和 device memory 之间复制数据、启动 GPU 上的代码、以及等待数据复制或 GPU 代码完成。CPU 和 GPU 可以同时执行代码，通常通过最大化 CPU 和 GPU 的利用率可以获得最佳性能。

运行在 GPU 上的代码称为 device code，出于历史原因，在 GPU 上执行的函数称为 kernel。启动 kernel 的执行称为 launching the kernel。kernel launch 可以理解为在 GPU 上启动大量线程并行执行该 kernel 代码。GPU 线程的行为类似于 CPU 上的线程，不过有一些对正确性和性能都很重要的差异，后面部分会介绍（参见 Section 3.2.2.1.1）。

### 1.2.2. GPU Hardware Model

**原文**
 Like any programming model, CUDA relies on a conceptual model of the underlying hardware. For the purposes of CUDA programming, the GPU can be considered to be a collection of Streaming Multiprocessors (SMs) which are organized into groups called Graphics Processing Clusters (GPCs). Each SM contains a local register file, a unified data cache, and a number of functional units that perform computations. The unified data cache provides the physical resources for shared memory and L1 cache.
 The allocation of the unified data cache to L1 and shared memory can be configured at runtime. The sizes of different types of memory and the number of functional units within an SM can vary across GPU architectures.
 Note
 The actual hardware layout of a GPU or the way it physically carries out the execution of the programming model may vary. These differences do not affect correctness of software written using the CUDA programming model.

**翻译**
和任何 programming model 一样，CUDA 依赖对底层硬件的一个概念性模型。就 CUDA 编程而言，可以把 GPU 看成由多个 Streaming Multiprocessors（SMs）组成的集合，这些 SMs 组织成称为 Graphics Processing Clusters（GPCs）的组。**每个 SM 包含一个本地的 register file、一个 unified data cache 以及一些执行计算的功能单元。unified data cache 提供了 shared memory 和 L1 cache 的物理资源。unified data cache 在 L1 和 shared memory 之间的分配可以在运行时配置。**不同类型内存的大小以及 SM 中功能单元的数量会随着 GPU 架构而变化。

> 注意
>
> GPU 的实际硬件布局或其物理执行 programming model 的方式可能不同。这些差异不会影响按 CUDA programming model 编写的软件的正确性。



![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-cpu-system-diagram.png)

*Figure 2：一个 GPU 有多个 SMs，每个 SM 包含多个功能单元。GPCs是 SMs 的集合。一个 GPU 是由多个 GPCs 组成并连接到 GPU 内存的。一个 CPU 通常有多个核心和一个连接到系统内存的内存控制器。CPU 和 GPU 通过如 PCIe 或 NVLINK 等互联连接。*

#### 1.2.2.1. Thread Blocks and Grids

当应用程序 launch 一个 kernel 时，它会启动许多 threads，通常是数百万个 threads。这些 threads 被组织成 blocks。一个 threads 组成的 block，顾名思义，被称为 thread block。thread blocks 被组织成一个 grid。grid 中所有的 thread blocks 具有相同的大小和维度。图3 显示了一个 thread block 网格的示意图。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)

*图 3 线程块的网格。每个箭头表示一个 thread（箭头数量并不代表实际 threads 的数量）*

thread blocks 和 grids 可以是 1 维、2 维或 3 维。这些维度可以简化将单个 thread 映射到工作单元或数据项的过程。

当一个 kernel 被 launch 时，它是使用一个特定的 execution configuration 启动的，该 configuration 指定了 grid 和 thread block 的维度。execution configuration 还可以包括可选参数，比如 cluster size、stream 和 SM 配置设置，这些将在后续部分介绍。

使用built-in variables，每个执行 kernel 的 thread 都可以确定它在所属 block 中的位置以及该 block 在所属 grid 中的位置。thread 还可以使用这些内建变量来确定 kernel 被 launch 的 thread blocks 和 grid 的维度。这使得每个 thread 在所有运行该 kernel 的 threads 中都有唯一身份。这个身份经常用于确定 thread 负责的数据或操作。

**一个 thread block 的所有 threads 都在单个 SM 上执行。**这允许 thread block 内的 threads 高效地相互通信和同步。thread block 内的所有 threads 都可以访问片上 shared memory，这可用于在 thread block 内的 threads 之间交换信息。

一个 grid 可能包含数百万个 thread blocks，而执行该 grid 的 GPU 可能只有几十个或几百个 SM。一个 thread block 的所有 threads 都由单个 SM 执行，并且在大多数情况下，在该 SM 上运行完成。**这些 thread blocks 之间的调度没有任何保证，因此一个 thread block 不能依赖其他 thread blocks 的结果，因为在该 thread block 完成之前其它 blocks 可能无法被调度。**图4显示了如何将 grid 中的 thread blocks 分配给 SM 的示例。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling.png)

*图 4 每个 SM 有一个或多个活跃的 thread blocks。在这个示例中，每个 SM 同时调度了三个 thread blocks。thread blocks 从 grid 分配到 SM 的顺序没有任何保证。*

CUDA programming model 使得任意大的 grids 都能在任意规模的 GPU 上运行，无论是只有一个 SM 还是有数千个 SM。为了实现这一点，CUDA programming model（在某些例外情况下）**要求不同 thread blocks 内的 threads 之间没有数据依赖**。也就是说，一个 thread 不应该依赖来自另一个相同 grid 内不同 thread block 的 thread 的结果或与其同步。grid 内不同的 thread blocks 在可用的 SMs 之间被调度，并可以以任意顺序执行。简而言之，CUDA programming model 要求能够以任意顺序、并行或串行执行 thread blocks。

##### 1.2.2.1.1. Thread Block Clusters

除了 thread blocks 之外，**compute capability 9.0 及以上的 GPUs** 有一个可选的分组级别，称为 clusters。clusters 是一组 thread blocks，像 thread blocks 和 grids 一样，可以按 1 维、2 维或 3 维布局。图5显示了一个组织成 clusters 的 thread blocks 网格。指定 clusters 并不会改变 grid 的维度或 thread block 在 grid 内的索引。![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-clusters.png)

*图 5 当指定 clusters 时，thread blocks 在 grid 中保持相同位置，但在所属 cluster 内也有一个位置。*

指定 clusters 会将**相邻的 thread blocks** 分组到 clusters 中，并在 cluster 级别提供一些额外的同步和通信机会。具体来说，cluster 中的所有 thread blocks 都在一个 GPC 中执行。图 6 显示了在指定 clusters 时，thread blocks 在 GPC 中如何调度到 SMs。由于**这些 thread blocks 被同时调度并在同一个 GPC 内执行**，因此属于同一 cluster 但来自不同 blocks 的 threads 可以使用 <u>Cooperative Groups</u> 提供的软件接口相互通信和同步。clusters 内的 threads 可以访问 cluster 所有 blocks 的 shared memory，这称为 <u>distributed shared memory</u>。cluster 的最大大小依赖于硬件并在不同设备之间有所不同。

图 6 显示了在 GPC 内，cluster 中的 thread blocks 如何被同时调度到 SMs 上。cluster 内的 thread blocks 在 grid 中始终是相邻的。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/thread-block-scheduling-with-clusters.png)

*图 6 当指定 clusters 时，cluster 内的 thread blocks 按其 cluster 形状排列在 grid 中。cluster 的 thread blocks 同时在单个 GPC 的 SMs 上调度。*

#### 1.2.2.2. Warps and SIMT

在一个 thread block 内，threads 被组织成每 32 个 threads 一组，称为 warp。一个 warp 以 Single‑Instruction Multiple‑Threads (SIMT) 模式执行 kernel 代码。在 SIMT 中，一个 warp 中所有 threads 都执行相同的 kernel 代码，但每个 thread 可能在代码中跟随不同的分支。也就是说，尽管程序的所有 threads 执行相同的代码，threads 不需要遵循相同的执行路径。

当 threads 被 warp 执行时，它们会被分配到 warp lane。warp lanes 的编号从 0 到 31，来自一个 thread block 的 threads 按照 Hardware Multithreading 中详细说明的可预测方式被分配到 warps。

warp 中的所有 threads 同时执行相同的指令。如果 warp 内有些 threads 在执行时遵循控制流分支而其他 threads 不遵循，则不遵循分支的 threads 会被屏蔽，而遵循分支的 threads 会被执行。例如，如果一个条件仅对 warp 中一半的 threads 为真，则 warp 的另一半在活跃 threads 执行这些指令时将被屏蔽。图 7 展示了这种情况。当 warp 中的不同 threads 跟随不同的代码路径时，这有时被称为 warp divergence。因此，当 warp 内的 threads 跟随相同的控制流路径时，GPU 的利用率最大化。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/active-warp-lanes.png)

*图 7 在这个例子中，仅具有偶数 thread index 的 threads 执行 if 语句体，其它 threads 在执行语句体时被屏蔽。*

在 SIMT 模式中，warp 内的所有 threads 以 lock step 的方式穿过 kernel。实际的硬件执行可能有所不同。有关这一区别重要性的更多信息，请参见 Independent Thread Execution 部分。不鼓励利用关于 warp 执行实际如何映射到真实硬件的知识。CUDA programming model 和 SIMT 指出 warp 中的所有 threads 一起穿过代码。只要遵循编程模型，硬件可能以对程序透明的方式优化被屏蔽的 lanes。如果程序违反此模型，这可能导致在不同 GPU 硬件上表现不同的未定义行为。

在编写 CUDA 代码时不必考虑 warps，但理解 warp 执行模型有助于理解诸如 <u>global memory coalescing</u> 和 <u>shared memory bank access patterns</u> 等概念。一些高级编程技术使用在 thread block 内对 warps 的特化来限制 thread divergence 并最大化 utilization。这种优化以及其他优化利用了 threads 在执行时被分组到 warps 的知识。

warp 执行的一个含义是，最好将 thread blocks 指定为具有总线程数为 32 的倍数。使用任意数量的 threads 是合法的，但当总数不是 32 的倍数时，thread block 的最后一个 warp 将在整个执行过程中有一些 lanes 未使用。这可能导致该 warp 的 functional units 利用率和 memory 访问 suboptimal。

> SIMT 常常与 Single Instruction Multiple Data (SIMD) 并行进行比较，但有一些重要的区别。在 SIMD 中，执行遵循单一控制流路径，而在 SIMT 中，每个 thread 允许跟随它自己的控制流路径。因此，SIMT 不像 SIMD 那样具有固定的数据宽度。有关 SIMT 更详细的讨论，请参见 SIMT Execution Model。

### 1.2.3. GPU Memory

在现代计算系统中，高效利用 memory 与最大化执行计算功能单元的利用率一样重要。异构系统有多个 memory spaces，并且 GPUs 除了 caches 外还包含各种类型的可编程片上 memory。以下各节将更详细地介绍这些 memory spaces。

#### 1.2.3.1. DRAM Memory in Heterogeneous Systems

GPUs 和 CPUs 都有直接连接的 DRAM 芯片。在有多个 GPU 的系统中，每个 GPU 都有自己的 memory。从 device code 的角度看，连接到 GPU 的 DRAM 被称为 **global memory**，因为它可被 GPU 中所有 SMs 访问。这个术语并不意味着它在系统中的所有位置都可访问。连接到 CPU 的 DRAM 被称为 system memory 或 host memory。

像 CPUs 一样，GPUs 使用 virtual memory addressing。在当前所有受支持的系统中，CPU 和 GPU 使用单一的 unified virtual memory space。这意味着系统中每个 GPU 的虚拟 memory 地址范围是唯一且不同于 CPU 和其它 GPU 的。对于特定的虚拟 memory address，可以确定该地址是在 GPU memory 还是 system memory，在多 GPU 系统中，还可以确定哪个 GPU memory 包含该地址。

有 CUDA APIs 用于分配 GPU memory、CPU memory，以及在 CPU 和 GPU 之间、GPU 内部或多 GPU 系统中不同 GPU 之间的 allocations 之间进行复制。在需要时可以显式控制数据的 locality。下面讨论的 Unified Memory 允许由 CUDA runtime 或系统硬件自动处理 memory 的 placement。

#### 1.2.3.2. On‑Chip Memory in GPUs

除了 global memory 外，每个 GPU 还有一些片上 memory。每个 SM 有自己的 **register file** 和 **shared memory**。这些 memories 是 SM 的一部分，可以被在 SM 中执行的 threads 极快访问，但不能被在其它 SM 上运行的 threads 访问。

**register file 通常由 compiler 分配，用于存储 thread local variables**。shared memory 可由 thread block 或 cluster 内的所有 threads 访问。shared memory 可用于在 thread block 的 threads 或 cluster 的 threads 之间交换数据。

SM 中的 register file 和 unified data cache 有有限的大小。SM 的 register file、unified data cache 的大小以及 unified data cache 如何配置以在 L1 和 shared memory 之间平衡，可在 Memory Information per Compute Capability 中找到。register file、shared memory space 和 **L1 cache** 在一个 thread block 的所有 threads 之间共享。

**要调度一个 thread block 到 SM，必须使每个 thread 所需的 registers 数乘以该 thread block 中的 threads 数小于或等于 SM 中可用的 registers。**如果 thread block 所需的 registers 数超过 register file 的大小，则该 kernel 无法 launch，必须减少 thread block 中的 threads 数，使 thread block 可 launch。

shared memory 的 allocations 是在 thread block 级别完成的。也就是说，与每个 thread 的 register allocations 不同，shared memory 的 allocations 是整个 thread block 公共的。

##### 1.2.3.2.1. Caches

除了可编程 memories，GPUs 还有 L1 和 L2 caches。每个 SM 有一个 L1 cache，它是 unified data cache 的一部分。更大的 L2 cache 被所有 SMs 共享。这可以在图 2 的 GPU 块图中看到。每个 SM 还有一个单独的 **constant cache**，用于缓存在 global memory 中声明在 kernel 生命周期内保持 constant 的值。compiler 也可以将 kernel 参数放入 constant memory。这可以避免 kernel 参数出现在 L1 data cache，从而来提高 kernel 性能。

#### 1.2.3.3. Unified Memory

当应用程序在 GPU 或 CPU 上显式分配 memory 时，该 memory 只能由在该设备上运行的代码访问。也就是说，CPU memory 只能由 CPU code 访问，GPU memory 只能由在 GPU 上运行的 kernels 访问。CUDA APIs 用于在 CPU 和 GPU 之间显式复制数据，以便在正确的时间将数据复制到正确的 memory。

一个叫 unified memory 的 CUDA 特性允许应用程序进行可由 CPU 或 GPU 访问的 memory allocations。CUDA runtime 或底层硬件在需要时访问或将数据重新定位到正确的位置。即使使用 unified memory，要获得最佳性能仍需尽量减少 memory 的迁移，并尽可能从直接连接到该 memory 的处理器访问数据。

系统的硬件特性决定了 memory spaces 之间如何实现访问和数据交换。Section Unified Memory 介绍了 unified memory systems 的不同类别。Section Unified Memory 包含有关 unified memory 在各种情况下使用和行为的更多细节。

**[1]** 在某些情况下，当使用像 CUDA Dynamic Parallelism 这样的特性时，thread block 可能会被挂起到 memory。这意味着 SM 的状态存储被到 GPU memory 上一块由系统管理的区域中，而SM则被释放出来去执行其它 thread blocks。这类似于 CPUs 上的 context swapping。这种情况不是常见。

**[2]** 一个例外情况是 mapped memory，它是通过启用GPU直接访问的属性分配的CPU memory。然而，mapped access 是通过 PCIe 或 NVLINK 连接进行的。GPU 无法通过并行性隐藏较高的 latency 和较低的 bandwidth，因此 mapped memory 不能作为 unified memory 或将数据放置到适当的 memory space 的高效替代方案。
