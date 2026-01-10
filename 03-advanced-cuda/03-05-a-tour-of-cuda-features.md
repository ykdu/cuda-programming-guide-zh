# 3.5 A Tour of CUDA Features

本编程指南的第 1–3 章已经对 CUDA 和 GPU 编程进行了介绍，涵盖了概念层面的基础内容以及一些简单的代码示例。指南第 4 部分中对具体 CUDA 特性的描述，默认读者已经掌握了第 1–3 章中所涉及的相关概念。

CUDA 提供了许多面向不同问题的特性，但并非所有特性都适用于每一种使用场景。本章的作用是介绍这些特性，说明它们各自的设计用途，以及它们可能帮助解决的问题。这些特性会按照其主要解决的问题类型进行粗略分类。有些特性（例如 CUDA graphs）可能同时适用于多个类别。

第 4 章将对这些 CUDA 特性进行更加完整和详细的说明。

### 3.5.1. 提升 Kernel 性能

本节中介绍的特性，都是为了帮助 kernel 开发者尽可能提升 kernel 的执行性能。

#### 3.5.1.1. Asynchronous Barriers

Asynchronous barriers 已在第 3.2.4.2 节中介绍过，它们为 threads 之间的 synchronize 提供了更加细致的控制方式。Asynchronous barriers 将 barrier 的 arrival 和 wait 两个阶段分离开来，这使得在等待其他 threads 到达 barrier 的过程中，应用可以继续执行那些不依赖该 barrier 的工作。Asynchronous barriers 可以针对不同的 thread scope 进行指定。关于 asynchronous barriers 的完整细节将在第 4.9 节中给出。

#### 3.5.1.2. Asynchronous Data Copies 与 Tensor Memory Accelerator（TMA）

在 CUDA kernel code 的语境下，asynchronous data copies 指的是在执行计算的同时，在 shared memory 与 GPU DRAM 之间移动数据的能力。这一概念不应与 CPU 与 GPU 之间的 asynchronous memory copies 混淆。该特性依赖 asynchronous barriers 来实现。第 4.11 节将对 asynchronous copies 的使用方式进行详细说明。

#### 3.5.1.3. Pipelines

Pipelines 是一种用于对工作进行分阶段组织，并协调多缓冲区 producer–consumer 模式的机制，通常用于将 compute 与 asynchronous data copies 进行重叠执行。第 4.10 节提供了在 CUDA 中使用 pipelines 的详细说明和示例。

#### 3.5.1.4. 通过 Cluster Launch Control 实现 Work Stealing

Work stealing 是一种用于在 workload 不均衡时维持资源利用率的技术，当某些 worker 已完成自身工作后，可以从其他 worker 处“窃取”任务。Cluster launch control 是在 compute capability 10.0（Blackwell）中引入的一项特性，它使 kernel 能够直接控制尚在执行流程中的 block scheduling，从而在运行时适配不均衡的 workload。**一个 thread block 可以取消另一个尚未开始执行的 thread block 或 cluster 的 launch**，接管其 index，并立即开始执行被窃取的工作。这种 work-stealing 的执行流程能够在数据分布不规则或运行时存在变化的情况下保持 SM 忙碌，减少空闲时间，在不完全依赖 hardware scheduler 的前提下实现更细粒度的 load balancing。

第 4.12 节将介绍如何使用这一特性。

### 3.5.2. Improving Latencies

本节中介绍的各项特性都围绕着一个共同目标，即降低某一类延迟，尽管不同特性所关注的具体延迟类型并不相同。总体而言，这些特性主要关注的是 kernel launch 层面或更高层次的延迟。本节不涉及 kernel 内部的 GPU memory access latency。

#### 3.5.2.1. Green Contexts

Green contexts，也称为 execution contexts，是一种 CUDA 特性，它允许程序创建只在 GPU 的一部分 SM 上执行工作的 CUDA context。默认情况下，kernel launch 中的 thread blocks 会被调度到 GPU 中任何能够满足该 kernel 资源需求的 SM 上。能够执行某个 thread block 的 SM 会受到多种因素的影响，包括但不限于 shared memory 使用量、register 使用量、cluster 的使用情况，以及 thread block 中 threads 的总数量。

Execution contexts 允许将 kernel launch 到一个专门创建的 context 中，从而进一步限制可用于执行该 kernel 的 SM 数量。需要注意的是，当程序创建了一个使用某一组 SM 的 green context 后，**GPU 上的其他 context 将不会把 thread blocks 调度到分配给该 green context 的 SM 上**。这其中也包括 primary context，即 CUDA runtime 默认使用的 context。通过这种方式，这些 SM 可以被保留给高优先级或对延迟敏感的 workload。

第 4.6 节对 green contexts 的使用方式进行了完整说明。Green contexts 在 CUDA 13.1 及之后版本的 CUDA runtime 中可用。

#### 3.5.2.2. Stream-Ordered Memory Allocation

Stream-ordered memory allocator 允许程序将 GPU memory 的 allocation 和 free 操作按顺序插入到某个 CUDA stream 中执行。与会立即执行的 cudaMalloc 和 cudaFree 不同，cudaMallocAsync 和 cudaFreeAsync 会将一次 memory allocation 或 free 操作插入到 CUDA stream 中。第 4.3 节对这些 API 的所有细节进行了说明。

#### 3.5.2.3. CUDA Graphs

CUDA graphs 允许应用指定一系列 CUDA operations（例如 kernel launch 或 memory copy）以及这些 operations 之间的依赖关系，从而使它们能够在 GPU 上高效执行。类似的行为也可以通过 CUDA streams 实现，事实上，创建 graph 的一种机制称为 stream capture，它可以将 stream 上的 operations 记录到一个 CUDA graph 中。Graph 也可以通过 CUDA graphs API 来创建。

当一个 graph 被创建后，它可以被 instantiation 并执行多次，这对于描述会被重复执行的 workload 非常有用。Graph 通过减少调用 CUDA operations 时的 CPU launch 开销来带来一定的性能收益，同时也使得那些只有在整个 workload 预先已知时才能进行的优化成为可能。

第 4.2 节对如何使用 CUDA graphs 进行了说明和演示。

#### 3.5.2.4. Programmatic Dependent Launch

Programmatic dependent launch 是一种 CUDA 特性，它允许一个 dependent kernel（即依赖于先前 kernel 输出的 kernel）在其所依赖的 primary kernel 尚未完成时就开始执行。Dependent kernel 可以先执行初始化代码以及与 primary kernel 无关的工作，直到需要 primary kernel 产生的数据时才阻塞在那里。Primary kernel 可以在 dependent kernel 所需的数据就绪时发出信号，从而释放 dependent kernel 继续执行。这种机制使 kernel 之间能够产生一定程度的重叠执行，有助于在尽量降低关键数据路径延迟的同时保持较高的 GPU 利用率。第 4.5 节对 programmatic dependent launch 进行了说明。

#### 3.5.2.5. Lazy Loading

Lazy loading 是一种用于控制 JIT compiler 在 application startup 阶段行为的特性。如果一个 application 包含大量需要从 PTX JIT 编译为 cubin 的 kernels，并且在 startup 阶段对所有 kernels 进行 JIT 编译，则可能导致较长的启动时间。默认行为是 module 只有在被实际使用时才会被编译。该行为可以通过 environment variables 进行修改，相关内容在第 4.7 节中进行了说明。

### 3.5.3. 功能性特性

本节中描述的特性具有一个共同点，即它们旨在为 CUDA 提供额外的能力或功能。

#### 3.5.3.1. Extended GPU Memory

Extended GPU memory 是一种在 NVLink-C2C 连接的系统中可用的特性，它允许 GPU 高效访问系统中的全部 memory。第 4.17 节对 EGM 进行了详细说明。

#### 3.5.3.2. Dynamic Parallelism

CUDA application 通常从运行在 CPU 上的代码中 launch kernel。同时，也可以从运行在 GPU 上的 kernel 中创建新的 kernel invocation。这一特性被称为 CUDA dynamic parallelism。第 4.18 节对如何从 GPU 上运行的代码中创建新的 GPU kernel launch 进行了详细说明。

### 3.5.4. CUDA 互操作性

#### 3.5.4.1. CUDA 与其他 API 的互操作

除了 CUDA 之外，还存在其他在 GPU 上运行代码的机制。应用级 GPU 最初是为了加速计算机图形而构建的，并拥有自己的一套 API，例如 Direct3D 和 Vulkan。应用程序可能希望使用某种图形 API 来进行 3D 渲染，同时使用 CUDA 执行计算。CUDA 提供了相关机制，使得存储在 GPU 上的数据可以在 CUDA context 与图形 API 所使用的 GPU context 之间进行交换。例如，应用可以使用 CUDA 执行一次仿真计算，然后再通过 3D API 对计算结果进行可视化。这一过程是通过使部分 buffer 同时对 CUDA 和图形 API 可读和 / 或可写来实现的。

用于与图形 API 共享 buffer 的这些机制，同样**也被用于与通信机制共享 buffer，从而支持在多节点环境中实现快速、直接的 GPU-to-GPU 通信**。

第 4.19 节介绍了 CUDA 如何与其他 GPU API 进行互操作，以及如何在 CUDA 与其他 API 之间共享数据，并针对多种不同的 API 提供了具体示例。

#### 3.5.4.2. Interprocess Communication

对于规模非常大的计算任务，通常会同时使用多块 GPU，以便利用更多的 memory 和 compute 资源协同处理同一个问题。在单一系统内，或者在集群计算术语中称为一个 node 内，可以在单个 host process 中使用多块 GPU。相关内容已在第 3.4 节中进行介绍。

在实际应用中，也常常会使用多个独立的 host process，这些 process 可能运行在同一台计算机上，也可能分布在多台计算机上。当多个 process 协同工作时，它们之间的通信被称为 interprocess communication。CUDA interprocess communication（CUDA IPC）提供了在不同 process 之间共享 GPU buffer 的机制。第 4.15 节对 CUDA IPC 如何用于协调不同 host process 之间的通信进行了说明和示例演示。

### 3.5.5. 细粒度控制

#### 3.5.5.1. Virtual Memory Management

如第 2.4.1 节所述，系统中的所有 GPU 以及 CPU memory 共享同一个统一的 virtual address space。大多数 application 可以直接使用 CUDA 提供的默认 memory management，而无需对其行为进行修改。然而，对于有更高需求的场景，CUDA driver API 提供了对该 virtual memory space 布局的高级且细致的控制能力。这类能力主要适用于在 GPU 之间（无论是在单一系统内还是跨多个系统）共享 buffer 时，对其行为进行精确控制。

第 4.16 节介绍了 CUDA driver API 所提供的这些控制能力，它们的工作方式，以及在什么情况下 developer 可能会发现它们具有优势。

#### 3.5.5.2. Driver Entry Point Access

Driver entry point access 指的是从 CUDA 11.3 开始提供的一项能力，它允许获取 CUDA Driver API 和 CUDA Runtime API 的 function pointer。该机制还允许 developer 获取 driver function 的特定变体的 function pointer，并访问比当前 CUDA toolkit 所提供版本更新的 driver function。第 4.20 节对 driver entry point access 进行了说明。

#### 3.5.5.3. Error Log Management

Error log management 提供了一组用于处理和记录 CUDA API error 的工具。通过设置一个 environment variable `CUDA_LOG_FILE`，可以将 CUDA error 直接输出到 stderr、stdout，或写入一个文件。Error log management 还允许 application 注册一个 callback，当 CUDA 遇到 error 时会触发该 callback。第 4.8 节对 error log management 提供了更多细节说明。
