# 4.15 Interprocess Communication

由不同 host 进程管理的多个 GPU 之间的通信，可以通过使用进程间通信（IPC）API 以及可进行 IPC 共享的 memory buffer 来实现，其方式是创建可在进程之间传递的 handle，随后通过这些 handle 获取指向 peer GPU 上 device memory 的进程本地 device pointer。

由某个 host thread 创建的任何 device memory pointer 或 event handle，都可以被同一进程中的其他 thread 直接引用。然而，device pointer 或 event handle 在创建它们的进程之外无效，因此不能被属于其他进程的 thread 直接引用。要在进程之间访问 device memory 和 CUDA events，应用程序必须使用 CUDA Interprocess Communication（IPC）或 Virtual Memory Management API 创建可在进程之间共享的 handle，这些 handle 可以通过标准的 host 操作系统 IPC 机制（例如进程间共享内存或文件）进行交换。一旦这些可移植的 handle 在进程之间交换完成，就必须使用 CUDA IPC 或 VMM API 从这些 handle 中获取进程本地的 device pointer。之后，这些进程本地 device pointer 就可以像在单个进程中一样使用。

在单节点、单操作系统实例内进行 IPC 所采用的可移植 handle 方法，同样适用于多节点 NVLink 互连集群中 GPU 之间的 peer-to-peer 通信。在多节点场景中，参与通信的 GPU 由运行在各个集群节点上独立操作系统实例中的进程管理，因此需要在操作系统实例之上增加额外抽象。多节点 peer 通信通过创建并交换所谓的 “fabric” handle 来实现，然后在参与的进程和对应多节点 rank 的操作系统实例中获取进程本地 device pointer。

有关用于建立和交换进程可移植 handle，以及节点和操作系统实例可移植 handle（用于获取 GPU 通信所需的进程本地 device pointer）的具体 API，请参阅下文（单节点 CUDA IPC）以及 ref:virtual-memory-management。

> note
>
> 在使用 CUDA IPC API 和 Virtual Memory Management（VM）API 进行 IPC 时，各自具有不同的优点和限制。
>
> CUDA IPC API 目前仅支持 Linux 平台。
>
> CUDA Virtual Memory Management API 允许在 memory allocation 时对 peer 可访问性和共享进行逐 allocation 控制，但需要使用 CUDA Driver API。

### 4.15.1. 使用传统 Interprocess Communication API 的 IPC

为了在进程之间共享 device memory pointer 和 event，应用程序必须使用 CUDA Interprocess Communication API，相关细节可参考 reference manual。IPC API 允许应用程序通过 `cudaIpcGetMemHandle()` 为给定的 device memory pointer 获取 IPC handle。CUDA IPC handle 可以通过标准的 host 操作系统 IPC 机制（例如进程间共享内存或文件）传递给其他进程。`cudaIpcOpenMemHandle()` 使用该 IPC handle 来获取一个可在其他进程中使用的有效 device pointer。event handle 也可以通过类似的入口函数进行共享。

IPC API 的一个使用示例是：一个主进程生成一批输入数据，使这些数据可被多个次级进程使用，而无需重新生成或复制数据。

> Note
>
> IPC API 仅支持 Linux。
>
> 请注意，IPC API 不支持 `cudaMallocManaged` 分配。
>
> 使用 CUDA IPC 进行通信的应用程序，应当使用相同的 CUDA driver 和 runtime 进行编译、链接和运行。
>
> 出于性能原因，由 `cudaMalloc()` 创建的 allocation 可能来自更大内存块的子分配。在这种情况下，CUDA IPC API 会共享整个底层内存块，这可能导致其他子分配也被共享，从而在进程之间产生潜在的信息泄露。为避免这种情况，建议仅共享大小为 2MiB 对齐的 allocation。
>
> 在 L4T 和嵌入式 Linux Tegra 设备上，仅支持 IPC event 共享 API，且要求 compute capability 为 7.x 或更高。IPC memory 共享 API 不支持 Tegra 平台。

### 4.15.2. 使用 Virtual Memory Management API 的 IPC

CUDA Virtual Memory Management API 允许创建可进行 IPC 共享的 memory allocation，并通过操作系统特定的 IPC handle 数据结构支持多个操作系统。
