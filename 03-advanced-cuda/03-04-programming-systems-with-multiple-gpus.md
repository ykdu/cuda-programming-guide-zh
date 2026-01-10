# 3.4 Programming Systems with Multiple GPUs

多 GPU 编程允许应用程序利用多块 GPU 所提供的**更高总算力、更大总内存容量以及更高的内存带宽**，从而处理单块 GPU 无法胜任的问题规模，并获得更高的整体性能。

CUDA 通过一整套从软件到硬件的机制来支持多 GPU 编程，主要包括：

- **Host 线程的 CUDA context 管理**
   支持在 host 侧创建和管理多个 CUDA context，用于同时驱动多块 GPU。
- **系统内所有处理器的统一内存寻址（Unified Memory Addressing）**
   使 CPU 与多块 GPU 能够在统一的虚拟地址空间中访问内存。
- **GPU 之间的点对点（peer-to-peer）批量内存拷贝**
   支持 GPU 直接进行大块数据传输，而无需经过 host 内存。
- **GPU 之间的细粒度点对点 load/store 内存访问**
   允许一个 GPU 以 load/store 的方式直接访问另一块 GPU 的 device memory。
- **更高层的抽象与配套系统软件**
   包括 CUDA 进程间通信（IPC）、基于 NCCL 的并行归约操作，以及通过 NVLink 和 / 或 GPU-Direct RDMA 实现的通信机制，这些能力通常通过 NVSHMEM、MPI 等 API 提供。

------

从最基础的角度看，多 GPU 编程要求应用程序能够：

- 同时管理多个处于活动状态的 CUDA context；
- 将数据合理地分发到各个 GPU；
- 在各个 GPU 上启动 kernel 来完成计算；
- 在 GPU 之间或 GPU 与 host 之间进行通信，并最终汇总或处理计算结果。

具体应如何实现这些步骤，取决于应用本身的算法结构、可利用的并行度，以及现有代码更适合哪一种多 GPU 映射方式。常见的多 GPU 编程模式包括：

- **单个 host 线程同时驱动多块 GPU**
- **多个 host 线程，每个线程驱动一块 GPU**
- **多个单线程的 host 进程，每个进程驱动一块 GPU**
- **包含多个线程的多个 host 进程，每个线程或进程驱动一块 GPU**
- **多节点集群系统**：各节点通过 NVLink 互连，GPU 由运行在不同操作系统实例中的线程和进程驱动

------

在上述所有多设备工作分配方式中，GPU 之间都可以通过 **device memory 之间的内存拷贝或 peer 访问**进行通信。

为了实现**高性能、低延迟的 GPU 通信**，应用通常需要：

- 查询并启用 GPU 之间的 peer-to-peer memory access；
- 利用 NVLink，在 GPU 之间实现高带宽的数据传输，以及更细粒度的 load/store 操作。

------

CUDA 的**统一虚拟寻址（Unified Virtual Addressing）\**机制，使得在\**同一 host 进程内**的多 GPU 通信变得更加直接。应用只需进行少量额外的查询和配置，就可以启用高性能的 peer-to-peer 内存访问和传输（例如通过 NVLink）。

对于**由不同 host 进程管理的多块 GPU**，CUDA 通过 **进程间通信（IPC）** 和 **虚拟内存管理（VMM）API** 提供支持。

- IPC 的基本概念以及节点内的 CUDA IPC API，在 *Interprocess Communication* 章节中进行了介绍。
- 高级虚拟内存管理（VMM）API 同时支持节点内和跨节点的 IPC，可在 Linux 和 Windows 操作系统上使用，并且允许以**单个内存分配为粒度**来精确控制哪些内存缓冲区可以被共享，相关内容在 *Virtual Memory Management* 章节中有详细说明。

------

CUDA 本身提供了构建 **GPU 组内集合操作（collective operations）**所需的基础 API，这些集合操作的参与者可以包括 GPU，必要时也可以包含 host。但 CUDA 并不直接提供高层次的多 GPU collective API。

真正面向多 GPU 的集合通信功能，是由更高层的 CUDA 通信库来提供的，例如 **NCCL** 和 **NVSHMEM**。

### 3.4.1. 多设备的 Context 与执行管理

应用程序要使用多块 GPU，首先需要完成一系列基础准备工作。这些工作包括：枚举系统中可用的 GPU 设备，根据设备的硬件属性、CPU 亲和性以及 GPU 之间的互连关系选择合适的设备，并为应用将要使用的每一块 GPU 创建对应的 CUDA context。只有在这些步骤完成之后，应用才能在多 GPU 环境中正确地分发任务并执行计算。

#### 3.4.1.1. 设备枚举

应用可以通过 CUDA runtime API 查询系统中 CUDA 设备的数量，并逐一获取每个设备的属性信息。下面的代码示例展示了如何获取设备数量、遍历所有设备，并查询每个设备的 compute capability。

```c
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

通过设备枚举，应用可以了解系统中有哪些 GPU 以及它们的基本能力，为后续根据架构特性或性能需求进行设备选择提供依据。

#### 3.4.1.2. 设备选择

在 runtime API 中，**每个 host 线程在任意时刻都会关联一个“当前设备”**。host 线程可以通过调用 `cudaSetDevice()` 来指定其后续操作所针对的 GPU。所有与 device 相关的操作，包括 device memory 的分配、kernel 的启动，以及 stream 和 event 的创建，都会作用在当前设备上。如果 host 线程从未调用过 `cudaSetDevice()`，则默认使用 device 0。

下面的代码示例说明了当前设备的设置如何影响内存分配和 kernel 执行：

```c
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // 将 device 0 设为当前设备
float* p0;
cudaMalloc(&p0, size);       // 在 device 0 上分配内存
MyKernel<<<1000, 128>>>(p0); // 在 device 0 上启动 kernel

cudaSetDevice(1);            // 将 device 1 设为当前设备
float* p1;
cudaMalloc(&p1, size);       // 在 device 1 上分配内存
MyKernel<<<1000, 128>>>(p1); // 在 device 1 上启动 kernel
```

通过在同一 host 线程中切换当前设备，应用可以顺序地向不同 GPU 提交各自的内存操作和 kernel 执行请求。

#### 3.4.1.3. 多设备下的 Stream、Event 与内存拷贝行为

在多 GPU 编程中，stream 和 event 都是与特定 device 相关联的对象，这种设备归属关系直接影响哪些操作是合法的。

当 kernel 被 launch 到一个 stream 时，该 stream 必须与当前设备关联，否则 kernel launch 将失败。下面的代码示例展示了这一限制：

```c
cudaSetDevice(0);               // 当前设备设为 device 0
cudaStream_t s0;
cudaStreamCreate(&s0);          // 在 device 0 上创建 stream s0
MyKernel<<<100, 64, 0, s0>>>(); // 在 device 0 的 s0 上启动 kernel

cudaSetDevice(1);               // 当前设备切换为 device 1
cudaStream_t s1;
cudaStreamCreate(&s1);          // 在 device 1 上创建 stream s1
MyKernel<<<100, 64, 0, s1>>>(); // 在 device 1 的 s1 上启动 kernel

// 该 kernel launch 会失败，因为 s0 属于 device 0，而当前设备是 device 1
MyKernel<<<100, 64, 0, s0>>>(); 
```

与 kernel launch 不同，**内存拷贝操作在 stream 与 device 的绑定上限制较少**。即使 memory copy 被提交到一个并非当前设备所关联的 stream，该拷贝操作依然可以成功执行。

Event 相关 API 在多设备场景下的行为则更加细致。如果 `cudaEventRecord()` 的 event 与 stream 关联到不同的设备，调用将失败；如果 `cudaEventElapsedTime()` 的两个 event 分别属于不同的设备，调用同样会失败。而 `cudaEventSynchronize()` 和 `cudaEventQuery()` 即使作用于不属于当前设备的 event，也可以正常执行。此外，`cudaStreamWaitEvent()` 允许 stream 和 event 关联到不同的设备，因此可以用来在多块 GPU 之间建立同步关系。

需要注意的是，每一块 GPU 都拥有各自独立的默认 stream。因此，一个设备的默认 stream 中提交的命令，与另一个设备的默认 stream 中提交的命令之间，不存在顺序保证，也不保证互斥执行，它们可能并发执行，也可能以任意顺序执行。

### 3.4.1. 多设备的 Context 与执行管理

应用程序要使用多块 GPU，首先需要完成一系列基础准备工作。这些工作包括：枚举系统中可用的 GPU 设备，根据设备的硬件属性、CPU 亲和性以及 GPU 之间的互连关系选择合适的设备，并为应用将要使用的每一块 GPU 创建对应的 CUDA context。只有在这些步骤完成之后，应用才能在多 GPU 环境中正确地分发任务并执行计算。

#### 3.4.1.1. 设备枚举

应用可以通过 CUDA runtime API 查询系统中 CUDA 设备的数量，并逐一获取每个设备的属性信息。下面的代码示例展示了如何获取设备数量、遍历所有设备，并查询每个设备的 compute capability。

```
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
```

通过设备枚举，应用可以了解系统中有哪些 GPU 以及它们的基本能力，为后续根据架构特性或性能需求进行设备选择提供依据。

#### 3.4.1.2. 设备选择

在 runtime API 中，每个 host 线程在任意时刻都会关联一个“当前设备”。host 线程可以通过调用 `cudaSetDevice()` 来指定其后续操作所针对的 GPU。所有与 device 相关的操作，包括 device memory 的分配、kernel 的启动，以及 stream 和 event 的创建，都会作用在当前设备上。如果 host 线程从未调用过 `cudaSetDevice()`，则默认使用 device 0。

下面的代码示例说明了当前设备的设置如何影响内存分配和 kernel 执行：

```
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // 将 device 0 设为当前设备
float* p0;
cudaMalloc(&p0, size);       // 在 device 0 上分配内存
MyKernel<<<1000, 128>>>(p0); // 在 device 0 上启动 kernel

cudaSetDevice(1);            // 将 device 1 设为当前设备
float* p1;
cudaMalloc(&p1, size);       // 在 device 1 上分配内存
MyKernel<<<1000, 128>>>(p1); // 在 device 1 上启动 kernel
```

通过在同一 host 线程中切换当前设备，应用可以顺序地向不同 GPU 提交各自的内存操作和 kernel 执行请求。

#### 3.4.1.3. 多设备下的 Stream、Event 与内存拷贝行为

在多 GPU 编程中，stream 和 event 都是与特定 device 相关联的对象，这种设备归属关系直接影响哪些操作是合法的。

当 kernel 被 launch 到一个 stream 时，该 stream 必须与当前设备关联，否则 kernel launch 将失败。下面的代码示例展示了这一限制：

```
cudaSetDevice(0);               // 当前设备设为 device 0
cudaStream_t s0;
cudaStreamCreate(&s0);          // 在 device 0 上创建 stream s0
MyKernel<<<100, 64, 0, s0>>>(); // 在 device 0 的 s0 上启动 kernel

cudaSetDevice(1);               // 当前设备切换为 device 1
cudaStream_t s1;
cudaStreamCreate(&s1);          // 在 device 1 上创建 stream s1
MyKernel<<<100, 64, 0, s1>>>(); // 在 device 1 的 s1 上启动 kernel

// 该 kernel launch 会失败，因为 s0 属于 device 0，而当前设备是 device 1
MyKernel<<<100, 64, 0, s0>>>(); 
```

与 kernel launch 不同，内存拷贝操作在 stream 与 device 的绑定上限制较少。即使 memory copy 被提交到一个并非当前设备所关联的 stream，该拷贝操作依然可以成功执行。

Event 相关 API 在多设备场景下的行为则更加细致。如果 `cudaEventRecord()` 的 event 与 stream 关联到不同的设备，调用将失败；如果 `cudaEventElapsedTime()` 的两个 event 分别属于不同的设备，调用同样会失败。而 `cudaEventSynchronize()` 和 `cudaEventQuery()` 即使作用于不属于当前设备的 event，也可以正常执行。此外，`cudaStreamWaitEvent()` 允许 stream 和 event 关联到不同的设备，因此可以用来在多块 GPU 之间建立同步关系。

需要注意的是，每一块 GPU 都拥有各自独立的默认 stream。因此，一个设备的默认 stream 中提交的命令，与另一个设备的默认 stream 中提交的命令之间，不存在顺序保证，也不保证互斥执行，它们可能并发执行，也可能以任意顺序执行。

### 3.4.2. 多设备的 Peer-to-Peer 传输与内存访问

#### 3.4.2.1. Peer-to-Peer 内存传输

当系统支持 peer-to-peer 内存访问时，CUDA 可以在 GPU 之间直接执行内存拷贝，并利用专用的 copy engine 以及 NVLink 硬件通道来获得更高的传输性能。

在具备条件的情况下，可以直接使用 `cudaMemcpy`，并指定拷贝类型为 `cudaMemcpyDeviceToDevice` 或 `cudaMemcpyDefault`。如果无法通过这种方式完成拷贝，则需要显式使用 peer-to-peer 拷贝接口，例如 `cudaMemcpyPeer()`、`cudaMemcpyPeerAsync()`、`cudaMemcpy3DPeer()` 或 `cudaMemcpy3DPeerAsync()`。下面的代码示例展示了一个典型的跨设备内存拷贝流程：

```c
cudaSetDevice(0);                   // 将 device 0 设为当前设备
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // 在 device 0 上分配内存

cudaSetDevice(1);                   // 将 device 1 设为当前设备
float* p1;
cudaMalloc(&p1, size);              // 在 device 1 上分配内存

cudaSetDevice(0);                   // 切回 device 0
MyKernel<<<1000, 128>>>(p0);        // 在 device 0 上执行 kernel

cudaSetDevice(1);                   // 切换到 device 1
cudaMemcpyPeer(p1, 1, p0, 0, size); // 将 p0 从 device 0 拷贝到 device 1
MyKernel<<<1000, 128>>>(p1);        // 在 device 1 上执行 kernel
```

当在隐式的 NULL stream 中执行跨设备内存拷贝时，该拷贝操作具有严格的同步语义：拷贝不会开始，直到此前已经提交到任一设备的所有命令都执行完成；同时，在拷贝完成之前，之后提交到任一设备的命令也不会开始执行。换句话说，这种拷贝在两个设备之间形成了一次全局同步点。

与普通 stream 的行为一致，如果使用异步方式执行跨设备内存拷贝，该拷贝操作可以与其他 stream 中的 kernel 或拷贝操作重叠执行。

如果两个设备之间已经启用了 peer-to-peer 内存访问（例如在 *Peer-to-Peer Memory Access* 一节中所述），那么设备之间的内存拷贝就不再需要经过 host 内存中转，从而能够显著提升拷贝性能。

#### 3.4.2.2. Peer-to-Peer 内存访问

是否能够在 GPU 之间直接进行内存访问，取决于系统的硬件拓扑，尤其是 PCIe 和 / 或 NVLink 的连接方式。在支持的系统中，一个 GPU 上执行的 kernel 可以直接解引用指向另一块 GPU device memory 的指针。对于给定的一对设备，如果 `cudaDeviceCanAccessPeer()` 返回 true，则说明这两个设备支持 peer-to-peer 内存访问。

在支持的前提下，应用需要通过调用 `cudaDeviceEnablePeerAccess()` 显式启用两个设备之间的 peer-to-peer 内存访问。下面的代码示例展示了这一过程。在未启用 NVSwitch 的系统中，每个设备在整个系统范围内最多支持八个 peer 连接。

```c
cudaSetDevice(0);                   // 将 device 0 设为当前设备
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // 在 device 0 上分配内存
MyKernel<<<1000, 128>>>(p0);        // 在 device 0 上执行 kernel

cudaSetDevice(1);                   // 切换到 device 1
cudaDeviceEnablePeerAccess(0, 0);   // 启用与 device 0 的 peer 访问

// 在 device 1 上启动 kernel
// 该 kernel 可以直接访问位于 device 0、地址为 p0 的内存
MyKernel<<<1000, 128>>>(p0);
```

在启用 peer-to-peer 内存访问后，两个设备会共享同一个统一虚拟地址空间（参见 *Unified Virtual Address Space*），因此同一个指针值可以在两个设备上用来访问同一块内存。

> 注意
>
> `cudaDeviceEnablePeerAccess()` 的作用范围是全局的。一旦启用，对 peer 设备上**此前已经分配的内存以及之后新分配的内存**都会生效。由于 runtime 需要在内存分配时立即确保这些分配对当前设备以及所有已建立 peer 关系的设备可见，因此启用 peer 访问会给 device memory 分配操作带来额外的运行时开销，并且该开销会随着 peer 设备数量的增加而成倍增长。
>
> 一种更具扩展性的替代方案是使用 CUDA 的虚拟内存管理（Virtual Memory Management，VMM）API，在内存分配阶段按需显式创建可被 peer 访问的内存区域。通过在分配时明确指定哪些内存需要支持 peer 访问，可以避免对不需要共享的内存引入额外的分配开销，同时也能够更清晰地限定哪些数据结构可以被跨设备访问，从而提升调试体验和整体可靠性（参见 *Virtual Memory Management*）。

#### 3.4.2.3. Peer-to-Peer 内存一致性

当计算被分布到多块 GPU 上、并且多个 device 上的 grid 同时执行时，必须通过显式的同步操作来保证内存访问的顺序性和正确性。跨设备进行同步的线程，其同步作用域属于 `thread_scope_system`，也就是说，这类同步需要在整个系统范围内生效。同样地，涉及多设备的内存操作也处于 `thread_scope_system` 这一内存同步域中。

在特定受限的使用场景下，CUDA 的 `ref::atomic-functions` 允许在 peer 设备的内存上执行 read-modify-write 操作。该能力仅适用于这样一种情况：**目标内存对象在任意时刻只会被一块 GPU 发起访问**。也就是说，虽然该内存位于另一块 GPU 的 device memory 中，但不会同时被多个 GPU 并发读写。

关于跨设备原子操作在可用性和一致性方面的具体要求与限制，需参考 CUDA 内存模型中对 atomicity 的相关说明。

#### 3.4.2.4. 多设备下的Managed Memory

在支持 peer-to-peer 的多 GPU 系统中，可以使用managed memory。关于多设备并发访问managed memory的详细约束条件，以及用于控制managed memory在特定 GPU 上进行独占访问的相关 API，均在 *Multi-GPU* 章节中进行了说明。

#### 3.4.2.5. Host IOMMU 硬件、PCI Access Control Services 与虚拟机

在 Linux 系统上，CUDA 和显示驱动在 **裸机（bare-metal）环境** 下并不支持启用了 IOMMU 的 PCIe peer-to-peer 内存传输。如果在 Linux 裸机系统中启用了 IOMMU，可能会导致 device memory 被悄然破坏而不易察觉。因此，在 Linux 裸机环境中运行 CUDA 时，必须禁用 IOMMU。

与此相对，在**虚拟机透传（PCIe pass through）**的场景下，CUDA 和显示驱动是支持 IOMMU 的。在这种情况下，应启用 IOMMU，并使用 VFIO 驱动来完成 PCIe 设备的透传配置。

在 Windows 系统上，不存在上述与 IOMMU 相关的限制。

有关相关背景，还可以参考 *Allocating DMA Buffers on 64-bit Platforms* 一节。

此外，在支持 IOMMU 的系统中，还可以启用 PCI Access Control Services（ACS）。PCI ACS 会将所有 PCI 点对点流量强制重定向通过 CPU 的 root complex，从而破坏原本设备之间的直连路径。这种行为会显著降低系统的整体双向带宽（bisection bandwidth），并带来明显的性能损失。
