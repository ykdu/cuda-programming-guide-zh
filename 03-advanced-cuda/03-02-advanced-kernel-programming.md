# 3.2 Advanced Kernel Programming

本章首先将深入介绍 NVIDIA GPU 的 hardware model，然后介绍一些在 CUDA kernel code 中可用的更高级特性，这些特性旨在提升 kernel performance。本章将引入与 thread scopes、asynchronous execution 以及相关 synchronization primitives 有关的一些概念。这些概念性的讨论为 kernel code 中可用的一些高级性能特性提供了必要的基础。

其中部分特性的详细说明包含在本编程指南后续部分中专门介绍这些特性的章节里。

* 本章介绍的 advanced synchronization primitives，在 Section 4.9 和 Section 4.10 中有完整说明。
* Asynchronous data copies（包括 tensor memory accelerator，TMA）在本章中引入，并在 Section 4.11 中有完整说明。

### 3.2.1. Using PTX

Parallel Thread Execution（PTX）是 CUDA 用来抽象 hardware ISAs 的 virtual machine instruction set architecture（ISA），已在 Section 1.3.3 中介绍。直接使用 PTX 编写代码是一种非常高级的优化手段，对大多数开发者来说并非必需，应当被视为最后的工具选择。尽管如此，在某些情况下，直接编写 PTX 所提供的细粒度控制能力，能够在特定应用中带来性能提升。所有可用的 PTX instructions 都记录在 PTX ISA 文档中。

`cuda::ptx` namespace

在代码中直接使用 PTX 的一种方式是使用 libcu++ 提供的 `cuda::ptx` namespace。该 namespace 提供了一组与 PTX instructions 直接映射的 C++ functions，从而简化了在 C++ 应用中使用这些指令的方式。更多信息请参考 cuda::ptx namespace 的相关文档。

**Inline PTX**

在代码中引入 PTX 的另一种方式是使用 inline PTX。该方法在对应的文档中有详细描述。这种方式与在 CPU 上编写 assembly code 非常相似。

### 3.2.2. Hardware Implementation

Streaming multiprocessor（SM，参见 GPU Hardware Model）被设计为能够并发执行数百个 threads。为了管理如此大量的 threads，它采用了一种称为 Single-Instruction, Multiple-Thread（SIMT）的并行计算模型，该模型在 SIMT Execution Model 中进行了描述。指令以流水线方式执行，在单个 thread 内利用 instruction-level parallelism，同时通过 simultaneous hardware multithreading 实现大规模的 thread-level parallelism，相关细节在 Hardware Multithreading 中进行了说明。与 CPU cores 不同，SM 按顺序发射指令，不进行 branch prediction 或 speculative execution。

SIMT Execution Model 和 Hardware Multithreading 两个章节描述了所有设备通用的 SM 架构特性。Section Compute Capabilities 则给出了不同 compute capabilities 设备的具体细节。

NVIDIA GPU 架构采用 little-endian 表示方式。

#### 3.2.2.1. SIMT Execution Model

每个 SM 以 32 个并行 threads 为一组创建、管理、调度和执行 threads，这组名为 warp。组成一个 warp 的各个 threads 在相同的程序地址一同开始，但每个 thread 有自己的指令地址计数器和寄存器状态，因此它们可以自由地分支和独立执行。warp 这一术语来源于织布，是最早的 parallel thread 技术。一个 half-warp 是 warp 的前半部分或后半部分。一个 quarter-warp 是 warp 的第一、第二、第三或第四个四分之一部分。

一个 warp 每次执行一条共同的指令，因此当一个 warp 的所有 32 个 threads 对执行路径一致时可以实现最高效率。如果 warp 的 threads 通过数据依赖的条件分支出现分歧，则 warp 会对所有被采用的分支路径分别执行，并禁用不在该路径上的 threads。分支分歧仅发生在 warp 内；不同的 warps 无论执行的是共同还是不相交的代码路径，都独立执行。

SIMT 架构类似于 SIMD（Single Instruction, Multiple Data）向量组织，因为单条指令控制多个处理单元。一个关键区别是 SIMD 向量组织将 SIMD 宽度暴露给软件，而 SIMT 指令指定单个线程的执行和分支行为。与 SIMD 向量机器相比，SIMT 使程序员能够为独立的标量 threads 编写 thread-level parallel 代码，以及为协同 threads 编写数据并行代码。出于正确性目的，程序员基本可以忽略 SIMT 行为；然而，若代码很少产生 warp 内 threads 分歧，则可以实现显著的性能提升。在实践中，这类似于 cache line 的作用：在为正确性设计时可以安全地忽略 cache line 大小，但在为峰值性能设计时则必须在代码结构中考虑 cache line。另一方面，向量架构要求软件将加载合并为向量并手动管理分歧。

##### 3.2.2.1.1. Independent Thread Scheduling

对于 compute capability 低于 7.0 的 GPUs，warp 使用一个在 warp 内所有 32 个 threads 之间共享的单一程序计数器，并通过一个活动掩码指定 warp 的活动 threads。因此，在分歧区域或不同执行状态下来自同一个 warp 的 threads 无法彼此发信号或交换数据，而且那些依赖锁或互斥量进行细粒度数据共享的算法可能导致死锁，这取决于争用的 threads 属于哪个 warp。

在 compute capability 7.0 及更高的 GPUs 中，independent thread scheduling 允许 threads 之间完全并发，而不管 warp 如何。使用 independent thread scheduling 时，GPU 为每个 thread 维护执行状态，包括程序计数器和调用栈，并能够在 thread 级别粒度上让出执行，无论是为了更好地利用执行资源，还是让一个 thread 等待另一个 thread 产生数据。一个调度优化器决定如何将来自同一个 warp 的活跃 threads 组合到 SIMT 单元中。这保留了前代 NVIDIA GPUs 中 SIMT 执行的高吞吐量，同时具备更大的灵活性：threads 现在可以在 sub-warp 级别粒度发生分歧和重新汇合。

Independent thread scheduling 可能破坏依赖于以前 GPU 架构的隐式 warp 同步行为的代码。warp-synchronous 代码假设同一个 warp 中的 threads 在每个指令上锁步执行，但 threads 在 sub-warp 级别粒度的分歧和重新汇合能力使这种假设无效。这可能导致参与执行的 threads 与预期不同。任何为 CC 7.0 之前的 GPUs 开发的 warp-synchronous 代码（例如无需同步的 intra-warp reductions）都应重新检查以确保兼容性。开发者应使用 `__syncwarp()` 明确同步这些代码，以确保在所有 GPU 世代中行为正确。

> Note
>
> 在一个 warp 中，正在参与当前指令执行的 threads 被称为 *active* threads，而未参与当前指令的 threads 被称为 *inactive*（disabled）。threads 可能由于多种原因处于 inactive 状态，包括比同一 warp 中的其它 threads 更早退出、走了与 warp 当前正在执行的分支路径不同的分支路径，或者作为某个 thread 数量不是 warp size 整数倍的 block 中的最后一批 threads。
>
> 如果一个由 warp 执行的 non-atomic 指令从该 warp 中的多个 threads 向 global 或 shared memory 的同一位置写入，那么该位置发生的串行写入次数可能会随着 device 的 compute capability 不同而变化。然而，对于所有 compute capabilities，最终由哪个 thread 执行最后一次写入都是未定义的。
>
> 如果一个由 warp 执行的 atomic 指令从该 warp 中的多个 threads 对 global memory 的同一位置进行 read、modify 和 write，那么对该位置的每一次 read/modify/write 都会发生，并且它们都会被串行化，但这些操作发生的顺序是未定义的。

#### 3.2.2.2. 硬件多线程

当一个 SM 被分配一个或多个 thread blocks 去执行时，它将这些 blocks 分成若干 warps，并且每个 warp 都由一个 warp scheduler 调度去执行。一个 block 被划分成 warps 的方式始终相同；每个 warp 包含连续、递增的 thread IDs，其中第一个 warp 包含 thread 0。Thread Hierarchy 介绍了 thread IDs 与 block 中 thread 索引的对应关系。

一个 block 中 warp 的总数定义如下：
$$
\mathrm{ceil}\left(\frac{T}{W_{\text{size}}}, 1\right)
$$

  * T 是每个 block 中的 threads 数量，
  * Wsize 是 warp size，它等于 32，
  * ceil(x, y) 表示 x 向上取整到最接近的 y 的倍数。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/warps-in-a-block.png)

图 19 一个 thread block 被划分成 32 个 threads 的 warps。#

每个由 SM 处理的 warp 的执行上下文（程序计数器、寄存器等）在整个 warp 生命周期中都保持在芯片内。因此，在**不同 warps 之间切换不会产生开销**。在每个指令发出周期内，warp scheduler 会选择一个准备好执行下一条指令的 warp（该 warp 的 active threads），并将指令发给这些 threads。

每个 SM 拥有一组 32 位的寄存器，这些**寄存器在 warps 之间划分**，还有一个在 thread blocks 之间划分的 shared memory。对于某个 kernel，能够在 SM 上同时驻留和处理的 blocks 和 warps 的数量取决于该 kernel 使用的寄存器和 shared memory 的数量，以及 SM 上可用的寄存器和 shared memory 的数量。此外，每个 SM 还有最大驻留 blocks 和 warps 数。这些限制，以及 SM 上可用的寄存器和 shared memory 数量，都取决于设备的 compute capability，并在 Compute Capabilities 部分中给出。如果每个 SM 上没有足够的资源来处理至少一个 block，则该 kernel 将无法 launch。为一个 block 分配的寄存器和 shared memory 的总数可以通过 Occupancy 部分中记录的几种方法来确定。

#### 3.2.2.3. Asynchronous Execution Features 异步执行特性

近年来的 NVIDIA GPU 在硬件层面逐步引入了一类新的异步执行能力，其核心目的并不是简单地增加并行度，而是为了在 GPU 内部更有效地重叠数据搬运、计算以及同步操作。这类能力关注的不是 CPU 与 GPU 之间的异步关系，而是 GPU 自身内部的执行方式：在同一个 thread block 内，某些由线程发起的操作可以不再与该线程的执行进度强绑定，而是由硬件独立推进，从而让线程在发起操作之后继续执行后续指令。这一点需要特别区分于 CUDA 中常见的异步 API（例如异步 kernel launch 或异步内存拷贝），后者描述的是 kernel、内存操作或 CPU 与 GPU 之间的异步关系，而这里讨论的是完全发生在设备端、thread block 内部的异步执行模型。

从计算能力 8.0（Ampere 架构）开始，NVIDIA 在硬件中引入了对这种设备端异步执行的直接支持。最具代表性的是硬件加速的 global memory 到 shared memory 的异步数据拷贝，以及与之配套的异步 barrier。与传统的 load/store 指令不同，这类异步拷贝并不是由线程逐条执行的访存指令序列，而是由专门的硬件机制在后台推进。线程只负责发起拷贝请求，随后便可以继续执行其他指令，而无需等待数据真正到达 shared memory。异步 barrier 则为这种执行方式提供了必要的同步手段，使线程能够在合适的时机等待一组异步操作完成。

在计算能力 9.0（Hopper 架构）中，这一异步执行模型得到了进一步扩展。Hopper 引入了 Tensor Memory Accelerator（TMA）单元，使得 GPU 可以在 global memory 与 shared memory 之间高效地搬运大块数据甚至多维张量，而无需线程显式编写复杂的拷贝循环。同时，Hopper 还支持异步的事务型 barrier 以及异步的矩阵乘加操作。这意味着在这一代架构中，不仅数据搬运可以与线程执行解耦，某些计算操作本身也能够以异步方式由硬件独立推进。

为了让程序能够使用这些硬件能力，CUDA 提供了一组可以在 device code 中调用的 API，并围绕它们定义了一套异步编程模型。这套模型明确规定了异步操作的行为语义，以及它们与 CUDA 线程之间的关系。所谓异步操作，是指由某个 CUDA 线程发起，但其执行过程并不由该线程逐步完成，而是好像由另一个独立的执行实体在推进。文档中将这一执行实体称为“异步线程（async thread）”。需要注意的是，这里的异步线程并不是一个真实存在、可被调度的 CUDA 线程，也不是 warp 的成员，而是一种用于描述硬件独立执行上下文的抽象概念。

在一个正确的程序中，一个或多个 CUDA 线程最终必须与异步操作进行同步，以确保其完成。发起异步操作的线程本身并不一定参与这种同步过程，也就是说，某个线程可以负责启动异步数据拷贝，而由其他线程在稍后的 barrier 或 pipeline 处等待该操作完成。尽管异步操作的执行与线程解耦，但异步线程始终与发起它的 CUDA 线程相关联，其生命周期、作用域以及可见性都受到该线程的约束。

异步操作完成与否的判断依赖于同步对象，这些同步对象可能是 barrier，也可能是 pipeline。它们用于向线程表明某一组异步操作是否已经结束。CUDA 在后续章节中会对这些同步原语本身进行更深入的说明，并通过具体的异步数据拷贝示例展示它们在实际编程中的使用方式。

##### 3.2.2.3.1. Async Thread 和 Async Proxy

异步操作在访问内存时，其行为方式与普通的加载和存储指令并不完全相同。为了区分这些不同的内存访问路径，CUDA 在编程模型中引入了三个概念：**异步线程（async thread）**、**通用代理（generic proxy）\**以及\**异步代理（async proxy）**。在常规情况下，普通的 load 和 store 指令都通过通用代理来访问内存，这也是传统 CUDA 程序默认遵循的内存访问与排序规则。而某些异步指令则不完全遵循这一路径，它们的执行语义需要通过“异步线程”和不同类型的代理来建模。

部分异步指令，例如 LDGSTS 以及 STAS/REDAS 这类指令，被建模为：**由一个异步线程执行，但仍然运行在通用代理之中**。这意味着，尽管这些操作的执行是异步推进的，但它们在内存访问路径上仍然属于“普通”的那一类。与此不同，另一些异步指令——例如使用 TMA 的批量异步拷贝，以及某些 Tensor Core 相关操作（如 tcgen05.*、wgmma.mma_async.*）——则被建模为：**由一个异步线程执行，并且运行在异步代理之中**。这一区分，直接决定了这些操作与普通 load/store 在内存顺序和一致性方面的关系。

当一个异步操作被发起时，CUDA 会为它关联一个异步线程。这个异步线程在概念上是独立的，它不同于发起该操作的 CUDA 线程。也就是说，CUDA 线程只是“启动”了异步操作，而真正推动该操作完成的是这个异步线程。接下来，内存顺序的行为取决于这个异步线程是运行在通用代理中，还是运行在异步代理中。

对于**运行在通用代理中的异步线程**，CUDA 提供了一种“部分有序”的保证。具体来说，在同一地址上，位于异步操作之前的普通 load 和 store，一定会在该异步操作之前完成，因此它们的顺序是被保证的。但对于位于异步操作之后的普通 load 和 store，CUDA 并不保证它们与异步操作之间仍然保持顺序关系。这意味着，如果在异步操作尚未完成之前，后续的普通 load 或 store 访问了同一地址，就有可能与异步操作产生竞态，直到该异步线程完成为止。

而对于**运行在异步代理中的异步线程**，内存顺序的约束则更加宽松。在这种情况下，无论是位于异步操作之前还是之后的普通 load 和 store，只要它们访问的是同一地址，都不再被保证与异步操作保持顺序。换句话说，异步代理中的操作与通用代理中的普通内存访问之间，默认是彼此“无序”的。为了在这两种代理之间建立正确的内存顺序关系，程序必须显式使用 **proxy fence**。只有通过 proxy fence，才能在通用代理和异步代理之间建立必要的同步，从而确保内存访问的正确性。

在使用 Tensor Memory Accelerator（TMA）执行异步拷贝时，这一点尤为重要。CUDA 文档在后续章节中通过具体示例展示了如何使用 proxy fence，在异步代理与通用代理之间进行同步，以确保异步拷贝过程中程序的内存一致性和执行正确性。

有关这些概念的更多细节，请参阅 PTX ISA documentation。

### 3.2.3. Thread Scopes

CUDA threads 形成了一个 Thread Hierarchy，并且利用这个层次结构对于编写既正确又高效的 CUDA kernels 是很关键的。在这个层次结构内部，memory operations 的可见性和同步范围是不同的。为了说明这种不均匀性，CUDA programming model 引入了 thread scopes 的概念。一个 thread scope 定义了哪些 threads 能够观察某个线程的 loads 和 stores，并指定哪些 threads 可以通过诸如 atomic operations 和 barriers 之类的 synchronization primitives 相互同步。每个 scope 在 memory hierarchy 中都有一个相应的 coherency 点。

Thread scopes 在 CUDA PTX 中有定义，并且也在 libcu++ 库中作为扩展提供。下面的表格定义了可用的 thread scopes：

| CUDA C++ Thread Scope       | CUDA PTX Thread Scope | 描述                                                         | 在 Memory Hierarchy 中的 Coherency 点 |
| --------------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------- |
| `cuda::thread_scope_thread` |                       | memory operations 仅对本地线程可见。                         | -                                     |
| `cuda::thread_scope_block`  | `.cta`                | memory operations 对同一个 thread block 中的其他 threads 可见。 | L1                                    |
|                             | `.cluster`            | memory operations 对同一 thread block cluster 中的其他 threads 可见。 | L2                                    |
| `cuda::thread_scope_device` | `.gpu`                | memory operations 对同一个 GPU device 中的其他 threads 可见。 | L2                                    |
| `cuda::thread_scope_system` | `.sys`                | memory operations 对同一 system（CPU、其他 GPUs）中的其他 threads 可见。 | L2 + 连接的 caches                    |

后续章节 *Advanced Synchronization Primitives* 和 *Asynchronous Data Copies* 中会展示如何使用 thread scopes。

### 3.2.4. Advanced Synchronization Primitives

本节介绍三类 synchronization primitives：

- Scoped Atomics，它将 C++ memory ordering 与 CUDA thread scopes 结合起来，使 threads 能够在 block、cluster、device 或 system scope 上安全地进行通信（参见 Thread Scopes）。

- Asynchronous Barriers，它将 synchronization 拆分为 arrival 和 wait 两个阶段，并且可以用于跟踪 asynchronous operations 的执行进度。

- Pipelines，它通过对工作进行分阶段处理，并协调多缓冲的 producer–consumer 模式，通常用于将 compute 与 asynchronous data copies 进行重叠。

#### 3.2.4.1. Scoped Atomics

第 5.4.5 节概述了 CUDA 中可用的 atomic functions。本节将重点介绍 scoped atomics，这类 atomics 支持 C++ 标准的 atomic memory semantics，可通过 libcu++ library 或 compiler 的 built-in functions 使用。Scoped atomics 提供了一组工具，用于在 CUDA thread hierarchy 中合适的层级上实现高效的 synchronization，从而在复杂的 parallel algorithms 中同时保证 correctness 和 performance。

##### 3.2.4.1.1 Thread Scope and Memory Ordering

Scoped atomics 结合了两个关键概念：

* Thread Scope：定义了哪些 threads 可以观察 atomic 操作的效果（参见 Thread Scopes）。
* Memory Ordering：定义了相对于其它内存操作的顺序约束（参见 **C++ standard atomic memory semantics**）。

CUDA C++ `cuda::atomic`

```c++
#include <cuda/atomic>
__global__ void block_scoped_counter() {
    // 仅在这个 block 内可见的共享 atomic 计数器
    __shared__ cuda::atomic<int, cuda::thread_scope_block> counter;

    // 初始化计数器（应仅由一个 thread 执行）
    if (threadIdx.x == 0) {
        counter.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // block 中所有 threads 原子自增
    int old_value = counter.fetch_add(1, cuda::memory_order_relaxed);
    // 使用 old_value...
}
```

内建 Atomic 函数版本

```c++
__global__ void block_scoped_counter() {
    // 仅在这个 block 内可见的共享计数器
    __shared__ int counter;

    // 初始化计数器（应仅由一个 thread 执行）
    if (threadIdx.x == 0) {
        __nv_atomic_store_n(&counter, 0,
                            __NV_ATOMIC_RELAXED,
                            __NV_THREAD_SCOPE_BLOCK);
    }
    __syncthreads();
    // block 中所有 threads 原子自增
    int old_value = __nv_atomic_fetch_add(&counter, 1,
                                          __NV_ATOMIC_RELAXED,
                                          __NV_THREAD_SCOPE_BLOCK);

    // 使用 old_value...
}
```

这个示例实现了一个 block 作用域的 atomic 计数器，演示了 scoped atomics 的基本概念：

* 共享变量：利用 `__shared__` memory 在 block 内所有 threads 之间共享一个计数器。
* Atomic 类型声明：`cuda::atomic<int, cuda::thread_scope_block>` 创建了一个具有 block 级可见性的 atomic 整数。
* 单次初始化：仅由 thread 0 初始化计数器，以防止设置期间出现竞争。
* Block 同步：`__syncthreads()` 确保所有 threads 在继续之前看到已初始化的计数器。
* Atomic 自增：每个 thread 原子地增加计数器并获得原来的值。

这里选择 `cuda::memory_order_relaxed`，是因为我们只需要 atomicity（不可分割的 read-modify-write），而不需要在不同 memory locations 之间施加 ordering 约束。由于这是一个简单的计数操作，increment 的执行顺序对正确性没有影响。

对于生产者-消费者模式，acquire-release 语义确保了正确的顺序：

CUDA C++ `cuda::atomic`

```c++
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ cuda::atomic<bool, cuda::thread_scope_block> ready;
    if (threadIdx.x == 0) {
        // Producer：写入 data 然后标记 ready
        data = 42;
        ready.store(true, cuda::memory_order_release);  // Release 保证数据写入可见
    } else {
        // Consumer：等待 ready 信号然后读取 data
        while (!ready.load(cuda::memory_order_acquire)) {  // Acquire 保证读取时看到写入
            // 轮询等待
        }
        int value = data;
        // 处理 value...
    }
}
```

内建 Atomic 函数版本

```c++
__global__ void producer_consumer() {
    __shared__ int data;
    __shared__ bool ready; // 仅 ready 标志需要 atomic 操作
    if (threadIdx.x == 0) {
        // Producer：写入 data 然后标记 ready
        data = 42;
        __nv_atomic_store_n(&ready, true,
                            __NV_ATOMIC_RELEASE,
                            __NV_THREAD_SCOPE_BLOCK);  // Release 保证数据写入可见
    } else {
        // Consumer：等待 ready 信号然后读取 data
        while (!__nv_atomic_load_n(&ready,
                                   __NV_ATOMIC_ACQUIRE,
                                   __NV_THREAD_SCOPE_BLOCK)) {  // Acquire 保证读取时看到写入
            // 轮询等待
        }
        int value = data;
        // 处理 value...
    }
}
```

##### 3.2.4.1.2 Performance Considerations

* 使用尽可能窄的 scope：block 作用域的 atomics 比 system 作用域的 atomics 更快。
* 偏好较弱的 ordering：仅在正确性必要时使用更强的 ordering。
* 考虑内存位置：shared memory 中的 atomics 比 global memory 中的 atomics 更快。

#### 3.2.4.2. 异步 barriers

异步 barrier 与典型的单阶段 barrier（`__syncthreads()`）不同之处在于：线程通知它已到达 barrier（“arrive”）的操作，与等待其它线程到达 barrier（“wait”）的操作是分开的。通过这种分离，一个线程可以在等待期间执行与 barrier 无关的额外工作，从而提高执行效率。异步 barriers 可以用于在 CUDA threads 中实现 producer-consumer 模式，或者在内存层次结构中启用异步数据复制，其中复制操作在完成时向 barrier 发出信号（“arrive on”）。

异步 barriers 在 compute capability 7.0 或更高的设备上可用。 compute capability 8.0 或更高的设备在 shared memory 中为异步 barriers 提供硬件加速，并显著提高了同步粒度，允许对 block 内任意 CUDA threads 子集进行硬件加速同步。早期架构仅支持在整个 warp (`__syncwarp()`) 或整个 block (`__syncthreads()`) 级别进行加速同步。

CUDA 编程模型通过 `cuda::std::barrier` 提供异步 barriers，这是一种符合 ISO C++ 的 barrier，可在 `libcu++` 库中使用。除了实现 `std::barrier` 之外，该库还提供了 CUDA 特定的扩展，可以选择 barrier 的 thread scope 以提升性能，并公开了更底层的 API。一个 `cuda::barrier` 可以与 `cuda::ptx` 互操作，方法是使用友元函数 `cuda::device::barrier_native_handle()` 来获取 barrier 的本机句柄，并将其传递给 `cuda::ptx` 函数。CUDA 还为 shared memory 中的异步 barriers 在 thread-block 范围内提供了 primitives API。

下面的表格概述了在不同 thread scope 下可用于同步的异步 barriers：

> Thread Scope | Memory Location | Arrive on Barrier | Wait on Barrier | Hardware-accelerated | CUDA APIs
> -------------|-----------------|------------------|-----------------|---------------------|-----------
> block        | local shared memory | 允许        | 允许            | 是（8.0+）         | `cuda::barrier`, `cuda::ptx`, primitives
> cluster      | local shared memory | 允许        | 允许            | 是（9.0+）         | `cuda::barrier`, `cuda::ptx`
> cluster      | remote shared memory | 允许       | 不允许          | 是（9.0+）         | `cuda::barrier`, `cuda::ptx`
> device       | global memory | 允许           | 允许            | 否                 | `cuda::barrier`
> system       | global/unified memory | 允许      | 允许            | 否                 | `cuda::barrier`

**时间分离的同步**

如果没有异步 arrive-wait barriers，使用 Cooperative Groups 时，线程 block 内的同步可通过 `__syncthreads()` 或 `block.sync()` 实现。

```c++
#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* arrive 之前的代码 */
         // 等待所有线程到达这里。
        block.sync();

        /* wait 之后的代码 */
    }
}
```
线程会在同步点 (`block.sync()`) 阻塞，直到所有线程到达该同步点。此外，在同步点之前发生的内存更新在同步点之后对 block 内所有线程都是可见的。

该模式包含三个阶段：

- 同步之前的代码执行将会在同步后读取的内存更新。
- 同步点。
- 同步之后的代码，在这里可以看到同步之前的内存更新。

使用异步 barriers 时，时间分离的同步模式如下：

CUDA C++ `cuda::barrier`

```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // 用预期到达数初始化 barrier。
    init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* arrive 之前的代码 */
    // 这个线程到达。到达不会阻塞线程。
    barrier_t::arrival_token token = bar.arrive();

    compute(data, i);

    // 等待所有参与 barrier 的线程完成 bar.arrive()。
    bar.wait(std::move(token));

    /* wait 之后的代码 */
  }
}
```

CUDA C++ `cuda::ptx`

```c++
#include <cuda/ptx>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  __shared__ uint64_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // 用预期到达数初始化 barrier。
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* arrive 之前的代码 */
    // 这个线程到达。到达不会阻塞线程。
    uint64_t token = cuda::ptx::mbarrier_arrive(&bar);

    compute(data, i);

    // 等待所有参与 mbarrier_arrive() 的线程完成。
    while(!cuda::ptx::mbarrier_try_wait(&bar, token)) {}

    /* wait 之后的代码 */
  }
}
```

CUDA C primitives

```c++
#include <cuda_awbarrier_primitives.h>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  __shared__ __mbarrier_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // 用预期到达数初始化 barrier。
    __mbarrier_init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* arrive 之前的代码 */
    // 这个线程到达。到达不会阻塞线程。
    __mbarrier_token_t token = __mbarrier_arrive(&bar);

    compute(data, i);

    // 等待所有参与 __mbarrier_arrive() 的线程完成。
    while(!__mbarrier_try_wait(&bar, token, 1000)) {}

    /* wait 之后的代码 */
  }
}
```

在这种模式中，同步点被分成到达点 (`bar.arrive()`) 和等待点 (`bar.wait(std::move(token))`)。线程在第一次调用 `bar.arrive()` 时开始参与 `cuda::barrier`。当线程调用 `bar.wait(std::move(token))` 时，它会阻塞直到参与的线程完成了预期次数的 `bar.arrive()`，这个次数是传递给 init() 的预期到达数量。在参与线程调用 `bar.arrive()` 之前发生的内存更新保证在它们调用 `bar.wait(std::move(token))` 之后对这些参与线程可见。注意，调用 `bar.arrive()` 不会阻塞线程，它可以继续执行与内存更新无关的其他工作。

arrive 和 wait 模式包含五个阶段：

- 在 arrive **之前**的代码执行内存更新，这些更新会在 wait **之后**被读取。
- arrive 点，带有隐式的 memory fence（即等价于
  `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)`）。
- arrive 与 wait **之间**的代码。
- wait 点。
- 在 wait **之后**的代码，此时可以看到在 arrive **之前**执行的内存更新。

有关如何使用异步 barriers 的完整指南，请参见 Asynchronous Barriers。

#### 3.2.4.3. Pipelines

CUDA programming model 提供了 **pipeline synchronization object** 作为一种协调机制，用于将 asynchronous memory copies 按多个阶段串联起来，从而方便实现双缓冲或多缓冲的 producer-consumer 模式。pipeline 本质上是一个带有 head 和 tail 的双端队列，它按先进先出（FIFO）的顺序处理工作。producer 线程将工作提交到 pipeline 的 head，而 consumer 线程从 pipeline 的 tail 拿取工作进行处理。

Pipelines 通过 libcu++ 库中的 `cuda::pipeline` API 暴露出来，同时也通过一个 primitives API 暴露。下面的表格描述了这两种 API 的主要功能。



| **`cuda::pipeline` API** | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `producer_acquire`       | 获取 pipeline 内部队列中一个可用的 stage。                   |
| `producer_commit`        | 提交在当前已获取 stage 上、在调用 `producer_acquire` 之后发出的 asynchronous 操作。 |
| `consumer_wait`          | 等待 pipeline 中最早的 stage 上的 asynchronous 操作完成。    |
| `consumer_release`       | 释放 pipeline 中最早的 stage，使其返回到 pipeline 对象以供重用。这个被释放的 stage 之后可以被 producer 再次获取。 |

| **Primitives API**         | 说明                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `__pipeline_memcpy_async`  | 请求一个从 global memory 到 shared memory 的内存拷贝，该请求将被提交以便异步执行。 |
| `__pipeline_commit`        | 对当前 pipeline stage 上、在调用之前发出的 asynchronous 操作进行提交。 |
| `__pipeline_wait_prior(N)` | 等待 pipeline 中除最后 N 次提交之外的所有 asynchronous 操作完成。 |

`cuda::pipeline` API 提供了更丰富的接口和更少的限制，而 primitives API 仅支持跟踪从 global memory 到 shared memory 的异步拷贝，该拷贝需满足特定的大小和对齐要求。primitives API 提供的功能等价于一个作用域为 `cuda::thread_scope_thread` 的 `cuda::pipeline` 对象。

有关详细的使用模式和示例，请参见 Pipelines 部分。

### 3.2.5. 异步数据拷贝

在内存层次结构中高效地移动数据是实现 GPU 计算高性能的基础。传统的同步内存操作会迫使 threads 在数据传输期间空闲等待。GPU 通过并行性本质上可以隐藏内存延迟；也就是说，当内存操作完成时，SM 会切换去执行另一个 warp。即便通过并行性来隐藏延迟，这种延迟仍然可能成为 memory bandwidth 利用率和计算资源效率的瓶颈。为了解决这些瓶颈，现代 GPU 架构提供了硬件加速的异步 data copy 机制，这些机制允许内存传输在 threads 继续执行其他工作时独立进行。

异步数据拷贝通过将启动内存传输和等待其完成分离，使计算与数据移动能够重叠。这样 threads 就可以在内存延迟期间做有用的工作，从而提高整体吞吐量和资源利用率。

> 注意
>
> 本节的概念和原则与前面章节关于异步执行的讨论具有相似性，但前一章讨论的是 kernel 启动或诸如 `cudaMemcpyAsync` 这样由 host API 调用的 memory transfer 的异步行为。这可以看作是应用程序不同部分之间的 asynchrony。
>
> 本节所描述的异步是指在单次 kernel launch 的执行过程中，在 GPU 的 DRAM（即 global memory）和在 SM 内的 memory（如 shared memory 或 tensor memory）之间的数据传输而不阻塞 GPU threads。这是一种发生在单次 kernel launch 内部的 asynchrony。

为了理解异步拷贝如何提高性能，检查一个常见的 GPU 计算模式是有帮助的。CUDA 应用程序通常采用一种 copy-and-compute 模式，该模式：

- 从 global memory 获取数据，
- 将数据存储到 shared memory，
- 再在 shared memory 数据上执行计算，并可能将结果写回 global memory。

在这种模式中，copy 阶段通常表达为 `shared[local_idx] = global[global_idx]`。编译器将这个从 global 到 shared memory 的复制展开为先从 global memory 读入寄存器，再从寄存器写入 shared memory。

当这种模式发生在迭代算法中时，每个 thread block 在 `shared[local_idx] = global[global_idx]` 赋值后需要同步，以确保所有对 shared memory 的写入完成，然后计算阶段才能开始。thread block 在 compute 阶段后也需要再次同步，以防在所有 threads 完成计算之前覆盖 shared memory。下面的代码片段演示了这种模式：

```c
#include <cooperative_groups.h>

__device__ void compute(int* global_out, int const* shared_in) {
    // 使用 shared memory 中当前 batch 的所有值进行计算
    // 并将这个 thread 的结果写回 global memory
}

__global__ void without_async_copy(int* global_out, int const* shared_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // 假设输入大小适合 batch_sz * grid.size

  extern __shared__ int shared[]; // block.size() * sizeof(int)

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // 计算当前 batch 在 global memory 中的索引
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];

    // 等待所有拷贝完成
    block.sync();

    // 计算并将结果写回 global memory
    compute(global_out + block_batch_idx, shared);

    // 等待 shared memory 上的计算完成
    block.sync();
  }
}
```

使用异步数据拷贝时，从 global memory 到 shared memory 的数据移动可以以异步方式完成，从而在等待数据到达的同时更有效地利用 SM。

```c
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in) {
    // 使用 shared memory 中当前 batch 的所有值进行计算
    // 并将这个 thread 的结果写回 global memory
}

__global__ void with_async_copy(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size());

  extern __shared__ int shared[]; // block.size() * sizeof(int)

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;

    // 整个 thread group 协作地将整个 batch 拷贝到 shared memory
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, block.size());

    // 在等待期间对不同数据执行计算

    // 等待所有拷贝完成
    cooperative_groups::wait(block);

    // 计算并将结果写回 global memory
    compute(global_out + block_batch_idx, shared);

    // 等待 shared memory 上的计算完成
    block.sync();
  }
}
```

`cooperative_groups::memcpy_async` 函数将 `block.size()` 个元素从 global memory 拷贝到 `shared`。这个操作就好像由另一个 thread 执行，并在拷贝完成后与当前 thread 调用 `cooperative_groups::wait` 同步。在拷贝操作完成之前，修改 global 数据或读取/写入 shared 数据会引起数据竞争。

这个例子说明了所有异步拷贝操作背后的基本概念：它们将内存传输的启动和完成分离，使 threads 能在数据在后台移动时执行其他工作。CUDA programming model 提供了多种 API 来访问这些功能，包括 Cooperative Groups 和 libcu++ 库中提供的 `memcpy_async` 函数，以及更底层的 `cuda::ptx` 和 primitives API。这些 API 具有类似语义：它们将对象从 source 复制到 destination，就好像由另一个 thread 执行一样，并且在拷贝完成后可通过不同的完成机制进行同步。

现代 GPU 架构为异步数据移动提供了多种硬件机制：

- LDGSTS（compute capability 8.0 及以上）允许从 global 到 shared memory 进行高效的小规模异步传输。
- tensor memory accelerator（TMA，compute capability 9.0 及以上）扩展了这些能力，提供针对大规模多维数据传输优化的 bulk-asynchronous 拷贝操作。
- STAS 指令（compute capability 9.0 及以上）使得可以从寄存器到 distributed shared memory 内的小规模异步传输。

这些机制支持不同的数据路径、传输大小和对齐要求，使开发者可以为特定的数据访问模式选择最合适的方法。以下表格给出了 GPU 内部异步拷贝支持的数据路径概览：

表 5：支持的异步拷贝及其可能的 source 与 destination memory space。空单元表示该 source–destination 组合不受支持。

| **Direction**   |                 | **Copy Mechanism**    |                            |
| --------------- | --------------- | --------------------- | -------------------------- |
| **Source**      | **Destination** | **Asynchronous Copy** | **Bulk-Asynchronous Copy** |
| global          | global          |                       |                            |
| shared::cta     | global          |                       | 支持（TMA，9.0+）          |
| global          | shared::cta     | 支持（LDGSTS，8.0+）  | 支持（TMA，9.0+）          |
| global          | shared::cluster |                       | 支持（TMA，9.0+）          |
| shared::cluster | shared::cta     |                       | 支持（TMA，9.0+）          |
| shared::cta     | shared::cta     |                       |                            |
| registers       | shared::cluster | 支持（STAS，9.0+）    |                            |

**Using LDGSTS**、**Using the Tensor Memory Accelerator (TMA)** 和 **Using STAS** 章节将进一步详细说明每种机制。

### 3.2.6. 配置 L1 / Shared Memory 的划分比例

如前文 *L1 data cache* 所述，在一个 SM 内部，L1 cache 与 shared memory 并非两套完全独立的硬件资源，而是共享同一块物理存储资源，通常称为 **unified data cache**。在多数架构上，如果某个 kernel 很少使用甚至不使用 shared memory，那么这块 unified data cache 可以被配置为尽可能多地用于 L1 cache，从而提升缓存命中率。





用于 shared memory 的 unified data cache 容量可以按 **kernel 粒度**进行配置。应用程序可以在 kernel 启动之前，通过 `cudaFuncSetAttribute` 指定该 kernel 偏好的 shared memory carveout：

```c++
cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
```

其中，`carveout` 用于表达该 kernel 对 shared memory 与 L1 cache 划分比例的偏好。

`carveout` 可以设置为一个整数百分比，表示期望 shared memory 占用该架构所支持的 **最大 shared memory 容量**的比例。除此之外，CUDA 还提供了三个便捷的枚举取值：

- `cudaSharedmemCarveoutDefault`：使用默认配置
- `cudaSharedmemCarveoutMaxL1`：尽可能多地分配给 L1 cache，shared memory 最小
- `cudaSharedmemCarveoutMaxShared`：尽可能多地分配给 shared memory，L1 cache 最小

不同架构所支持的最大 shared memory 容量以及可用的 carveout 档位并不相同，具体取值请参考 *Shared Memory Capacity per Compute Capability*。

当指定的百分比不能精确映射到某个受支持的 shared memory 容量时，运行时会选择 **下一个更大的可用容量**。例如，对于 compute capability 12.0 的设备，其最大 shared memory 容量为 100 KB，且仅支持 0、8、16、32、64 和 100 KB 这些档位；如果将 carveout 设置为 50%，最终得到的 shared memory 容量将是 64 KB，而不是 50 KB。

传递给 `cudaFuncSetAttribute` 的函数必须使用 `__global__` 修饰符声明。需要注意的是，`cudaFuncSetAttribute` 所设置的 carveout 仅作为运行时的偏好（preference）。驱动程序会在可能的情况下采用该配置，但如果为了正确执行 kernel 或避免性能问题（例如过度抖动），驱动程序可以选择不同的划分方式。

> **Note**
>
> CUDA 还提供了另一个 API：`cudaFuncSetCacheConfig`，同样可以用于调整某个 kernel 的 L1 cache 与 shared memory 的相对比例。然而，这个接口在运行时语义和调度行为上存在显著差异。但该 API 为 kernel launch 设置了对 shared/L1 平衡的硬性要求。这种行为在交替使用不同配置的 kernel 场景下，可能导致不必要的 kernel 启动串行化。相比之下，`cudaFuncSetAttribute` 提供的是 按 kernel 描述的偏好信息，并明确允许驱动程序根据整体执行情况选择是否实际切换硬件划分，从而更容易避免频繁的配置切换（thrashing）。因此，在大多数场景下，推荐优先使用 `cudaFuncSetAttribute` 来表达 L1 与 shared memory 的偏好。

对于每个 block 需要使用超过 48 KB shared memory 的 kernel，其行为具有明显的架构相关性。这类 kernel 必须使用 **动态 shared memory**，而不能使用静态大小的 shared memory 数组，并且需要通过 `cudaFuncSetAttribute` 显式声明其最大动态 shared memory 使用量。

```c
// Device 代码
__global__ void MyKernel(...)
{
  extern __shared__ float buffer[];
  ...
}

// Host 代码
int maxbytes = 98304; // 96 KB
cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
MyKernel <<<gridDim, blockDim, maxbytes>>>(...);
```

通过这种方式，应用程序明确告知运行时该 kernel 可能使用的最大 shared memory 容量，驱动程序据此进行调度和资源分配。
