# 4.16 Virtual Memory Management

在 CUDA 编程模型中，内存分配调用（例如 cudaMalloc()）会返回一个位于 GPU memory 中的地址。这个地址可以用于任何 CUDA API，也可以在 device kernel 内部使用。开发者可以通过 cudaEnablePeerAccess 来启用 peer device 对该内存分配的访问，这样不同 device 上的 kernel 就可以访问同一份数据。但这样做的结果是，所有过去和未来的用户内存分配都会被映射到目标 peer device 上，这会让用户在不知情的情况下为所有 cudaMalloc 分配付出额外的运行时映射成本。在大多数场景中，应用只需要和另一个 device 共享少量内存分配，通常没有必要把所有分配都映射到所有 device 上。此外，这种方式在扩展到 multi-node 场景时本身也会变得非常困难。

CUDA 提供了一套 virtual memory management（VMM）API，让开发者可以对这一过程进行显式的、低层级的控制。

Virtual memory allocation 是一个由 operating system 和 Memory Management Unit（MMU）共同管理的复杂过程，主要分为两个阶段。第一步，OS 会为程序保留一段连续的 virtual address range，但此时并不会分配任何 physical memory。第二步，当程序第一次尝试使用这段内存时，OS 才会提交这些 virtual address，并按需为 virtual pages 分配对应的 physical storage。

CUDA 的 VMM API 将类似的概念引入了 GPU memory management，允许开发者先显式地保留一段 virtual address range，然后在之后再把它映射到 physical GPU memory。通过 VMM，应用可以明确选择哪些内存分配可以被其他 device 访问。

VMM API 让复杂应用可以在多个 GPU（以及 CPU cores）之间更高效地管理内存。通过手动控制 memory reservation、mapping 以及 access permissions，VMM API 支持诸如 fine-grained data sharing、zero-copy transfers 和自定义 memory allocator 等高级技术。CUDA VMM API 为用户提供了对 GPU memory 管理的精细化控制能力。

开发者可以从 VMM API 中获得多方面的收益：

- 对 virtual memory 和 physical memory 的精细化控制，允许将非连续的 physical memory chunk 映射到连续的 virtual address space，这有助于减少 GPU memory 碎片，并提升内存利用率，尤其适用于 deep neural network training 这类大规模 workload。
- 通过将 virtual address space 的保留与 physical memory 的实际分配解耦，实现更高效的内存分配与释放。开发者可以先保留较大的 virtual memory 区域，再按需映射 physical memory，而无需进行昂贵的 memory copy 或重新分配，从而提升动态数据结构和可变大小内存分配的性能。
- 支持动态扩展 GPU memory allocation，而不需要复制并重新分配全部数据，类似于 CPU memory 管理中的 realloc 或 std::vector，这使 GPU memory 的使用方式更加灵活高效。
- 通过提供低层级 API，帮助开发者构建更复杂的 memory allocator 和 cache management system，从而提升开发效率和应用性能，例如在 large language model 中动态管理 key-value cache，改善吞吐和延迟。
- 在分布式 multi-GPU 场景下，CUDA VMM API 也具有很高的价值，它支持在多个 GPU 之间高效地共享和访问内存。通过将 virtual address 与 physical memory 解耦，API 允许开发者创建统一的 virtual address space，并将数据动态映射到不同 GPU 上，从而优化内存使用并减少数据传输开销。例如，NVIDIA 的 NCCL 和 NVShmem 等库就大量使用了 VMM。

总的来说，CUDA VMM API 为开发者提供了一套超越传统 malloc 类抽象的高级工具，用于实现精细、高效、灵活且可扩展的 GPU memory 管理，这对高性能和大内存应用至关重要。

> [!注意]
> 本节中介绍的这一整套 API 需要系统支持 UVA。参见 Virtual Memory Management APIs。

### 4.16.1 预备知识

#### 4.16.1.1 定义

**Fabric Memory**：Fabric memory 指的是可以通过高速互连 fabric（例如 NVIDIA 的 NVLink 和 NVSwitch）进行访问的内存。这种 fabric 在多个 GPU 或 node 之间提供内存一致性和高带宽通信层，使它们能够高效地共享内存，就好像这些内存连接在一个统一的 fabric 上，而不是分别孤立在各个 device 上。

CUDA 12.4 及之后的版本引入了 VMM allocation handle 类型 CU_MEM_HANDLE_TYPE_FABRIC。在受支持的平台上，并且在 NVIDIA IMEX daemon 正在运行的前提下，这种 allocation handle 不仅支持通过任意通信机制（例如 MPI）在单个 node 内共享内存，也支持跨 node 共享。这使得 multi-node NVLink system 中的 GPU 即使位于不同 node，也能够映射同一 NVLink fabric 中其他 GPU 的内存。

**Memory Handles**：在 VMM 中，handle 是用于表示 physical memory allocation 的不透明标识符。这些 handle 是低层级 CUDA VMM API 中内存管理的核心，它们使得对 physical memory object 的控制更加灵活，并支持将其映射到 virtual address space 中。每个 handle 都唯一对应一个 physical memory allocation。handle 作为内存资源的抽象引用存在，不会暴露直接的 pointer。通过 handle，可以进行跨 process 或 device 的内存导出和导入操作，从而实现内存共享和虚拟化。

**IMEX Channels**：IMEX 是 internode memory exchange 的缩写，是 NVIDIA 跨 node GPU-to-GPU 通信解决方案的一部分。IMEX channel 是 GPU driver 提供的一项特性，用于在 IMEX domain 内的 multi-user 或 multi-node 环境中实现基于用户的内存隔离。IMEX channel 本身承担着安全性和隔离机制的作用。

IMEX channel 与 fabric handle 直接相关，并且在 multi-node GPU 通信中必须启用。当一个 GPU 分配了内存并希望让不同 node 上的 GPU 访问这块内存时，首先需要导出该内存对应的 handle。在导出过程中，会通过 IMEX channel 生成一个安全的 fabric handle，只有具备正确 channel 访问权限的远端 process 才能导入该 handle。

**Unicast Memory Access**：在 VMM API 的语境下，unicast memory access 指的是由特定 device 或 process 对 physical memory 进行受控、直接的映射和访问，并且对应一个唯一的 virtual address range。与将访问权限广播给多个 device 不同，unicast memory access 表示只向某一个 GPU device 明确授予对某个已保留 virtual address range 的读写权限，该地址范围映射到一个 physical memory allocation。

**Multicast Memory Access**：在 VMM API 的语境下，multicast memory access 指的是通过 multicast 机制，将同一个 physical memory allocation 或 memory region 同时映射到多个 device 的 virtual address space 中的能力。这使数据可以在多个 GPU 之间以 one-to-many 的方式高效共享，减少重复的数据传输，并提升通信效率。NVIDIA 的 CUDA VMM API 支持创建 multicast object，用于将来自多个 device 的 physical memory allocation 绑定在一起。

#### 4.16.1.2 支持性查询

在尝试使用相关特性之前，应用应当先查询这些特性是否受支持，因为其可用性可能会因 GPU architecture、driver version 以及所使用的具体 software library 而有所不同。下面的内容将详细说明如何通过编程方式检查所需特性的支持情况。

**VMM Support**：在尝试使用 VMM API 之前，应用必须确保目标 device 支持 CUDA virtual memory management。下面的代码示例展示了如何查询 device 是否支持 VMM。

```c++
int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(
    &deviceSupportsVmm,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
    device
);
if (deviceSupportsVmm != 0) {
    // device 支持 Virtual Memory Management
}
```

**Fabric Memory Support**：在尝试使用 fabric memory 之前，应用必须确认目标 device 支持 fabric memory。下面的代码示例展示了如何查询是否支持 fabric memory：

```c++
int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(
    &deviceSupportsFabricMem,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
    device
);
if (deviceSupportsFabricMem != 0) {
    // device 支持 Fabric Memory
}
```

除了使用 CU_MEM_HANDLE_TYPE_FABRIC 作为 handle type，并且在交换 sharable handle 时不需要依赖 OS 原生的 inter-process communication 机制之外，fabric memory 的使用方式与其他 allocation handle type 并没有区别。

**IMEX Channels Support** 在 IMEX domain 内，IMEX channel 用于在 multi-user 环境中实现安全的内存共享。NVIDIA driver 通过创建一个字符设备 nvidia-caps-imex-channels 来实现这一机制。要使用基于 fabric handle 的共享，用户需要确认以下两点：

- 首先，应用必须确认该设备在 /proc/devices 中存在；

  ```shell
  # cat /proc/devices | grep nvidia
  195 nvidia
  195 nvidiactl
  234 nvidia-caps-imex-channels
  509 nvidia-nvswitch
  
  nvidia-caps-imex-channels 设备应当具有一个 major number（例如 234）。
  ```

- 其次，两个 CUDA process（一个 exporter，一个 importer）要想共享内存，必须都能够访问同一个 IMEX channel 文件。这些文件（例如 /dev/nvidia-caps-imex-channels/channel0）是代表单个 IMEX channel 的节点。system administrator 需要提前创建这些文件，例如使用 mknod() 命令。

  ```shell
  # mknod ∕dev∕nvidia-caps-imex-channels∕channelN c <major_number> 0
  
  该命令会使用从 /proc/devices 中获取的 major number 来创建 channelN。
  ```

> [!注意]
> 在默认情况下，如果指定了 NVreg_CreateImexChannel0 模块参数，driver 可以自动创建 channel0。

**Multicast Object Support**：在尝试使用 multicast object 之前，应用必须确认目标 device 支持该特性。下面的代码示例展示了如何查询是否支持 multicast object：

```c++
int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(
    &deviceSupportsMultiCast,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
    device
);
if (deviceSupportsMultiCast != 0) {
    // device 支持 Multicast Objects
}
```

### 4.16.2 API 概览

VMM API 为开发者提供了对 virtual memory management 的精细化控制。由于 VMM 是一套非常低层级的 API，因此需要直接使用 CUDA Driver API。该 API 具有很强的通用性，既可以用于 single-node 环境，也可以用于 multi-node 环境。

为了高效地使用 VMM，开发者需要对内存管理中的几个关键概念有扎实的理解：

- 了解 operating system 的 virtual memory 基础原理，包括它如何管理 page 和 address space
- 理解 memory hierarchy 以及硬件层面的特性
- 熟悉 inter-process communication（IPC）方式，例如 socket 或 message passing
- 具备关于 memory access rights 安全性的基础认知

VMM API 的工作流程由一系列内存管理步骤组成，其核心目标是在不同 device 或 process 之间共享内存。首先，开发者需要在 source device 上分配 physical memory。为了实现共享，VMM API 使用 handle 来向 target device 或 process 传递与该内存相关的必要信息。用户必须导出一个用于共享的 handle，该 handle 可以是 OS-specific handle，也可以是 fabric-specific handle。OS-specific handle 仅限于 single-node 场景下的 inter-process communication，而 fabric-specific handle 的适用范围更广，既可用于 single-node，也可用于 multi-node 环境。需要注意的是，使用 fabric-specific handle 需要启用 IMEX channel。

当 handle 被导出之后，需要通过 inter-process communication 协议将其共享给接收方 process，具体采用哪种方式由开发者自行决定。接收方 process 随后使用 VMM API 来导入该 handle。在 handle 成功完成导出、共享和导入之后，source 和 target process 都必须保留 virtual address space，用于映射已分配的 physical memory。最后一步是为各个 device 设置 memory access rights，确保访问权限被正确配置。上述完整流程（包括两种 handle 方式）在配套的示意图中有更详细的说明。

 

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/vmm-overview-diagram.png)

*Figure 52 VMM使用概览*
*该图概述了使用 VMM 所需的一系列步骤。流程首先从对运行环境的评估开始。基于这一评估，用户需要做出一个关键的初始决策：是使用 fabric memory handle，还是使用 OS-specific handle。根据最初选择的 handle 类型，后续会执行一套不同的步骤。不过，最终的内存管理操作——具体来说是对已分配内存进行 mapping、reserve 以及设置 access rights——与所选择的 handle 类型无关，都是相同的。*

### 4.16.3 单播内存共享 （Unicast Memory Sharing）

GPU memory 的共享既可以发生在一台包含多个 GPU 的机器上，也可以发生在由多台机器组成的网络中。整个流程遵循以下几个步骤：

- Allocate and Export：CUDA 程序在某个 GPU 上分配内存，并获取该内存对应的 sharable handle。
- Share and Import：通过 IPC、MPI、NCCL 等方式将 handle 发送给 node 内的其他程序，在接收方 GPU 上，由 CUDA driver 导入该 handle 并创建必要的内存对象。
- Reserve and Map：driver 在程序的 Virtual Address（VA）与 GPU 的 Physical Address（PA）以及 network Fabric Address（FA）之间建立映射关系。
- Access Rights：为该内存分配设置访问权限。
- Releasing the Memory：在程序结束执行时释放所有相关的内存分配。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/unicast-memory-sharing.png)

*Figure 53*
*Unicast Memory Sharing 示例*

#### 4.16.3.1 分配与导出（Allocate and Export）

Allocating Physical Memory 使用 virtual memory management API 进行内存分配的第一步，是创建一块 physical memory chunk，作为后续内存分配的实际支撑。要分配 physical memory，应用必须使用 cuMemCreate API。该函数创建的内存分配此时还不具备任何 device 或 host 的映射关系。函数参数 CUmemGenericAllocationHandle 用于描述要分配内存的属性，例如内存的分配位置、该分配是否需要共享给其他 process（或 graphics API），以及所分配内存的 physical 属性。用户必须确保所请求的内存分配大小满足相应的 granularity 对齐要求。关于内存分配的 granularity 要求，可以通过 cuMemGetAllocationGranularity 进行查询。

```c++
CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    // 以下二选一：
    CUmemAllocationHandleType handleType =
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR; // OS-Specific Handle (Linux)
    //CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC; // Fabric Handle
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleType = handleType;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);
    return allocHandle;
}
```

> [!注意]
> 通过 cuMemCreate 分配的内存是由其返回的 CUmemGenericAllocationHandle 来引用的。需要注意的是，这个引用并不是一个 pointer，此时对应的内存仍然不可访问。

> [!注意]
> 可以通过 cuMemGetAllocationPropertiesFromHandle 来查询 allocation handle 的属性。

**Exporting Memory Handle** CUDA virtual memory management API 提供了一种新的 interprocess communication 机制，通过 handle 来交换与内存分配和 physical address space 相关的必要信息。用户可以导出用于 OS-specific IPC 的 handle，或者用于 fabric-specific IPC 的 handle。OS-specific IPC handle 只能用于 single-node 场景，而 fabric-specific handle 既可以用于 single-node，也可以用于 multi-node 场景。

OS-Specific Handle (Linux)

```c++
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
int fd;
cuMemExportToShareableHandle(&fd, handle, handleType, 0);
```

Fabric Handle

```c++
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
CUmemGenericAllocationHandle handle = allocatePhysicalMemory(0, 1<<21);
CUmemFabricHandle fh;
cuMemExportToShareableHandle(&fh, handle, handleType, 0);
```

> [!注意]
> OS-specific handle 要求所有相关的 process 都必须属于同一个 OS。

> [!注意]
> Fabric-specific handle 需要由 sysadmin 启用 IMEX channel。

#### 4.16.3.2 共享与导入（Share and Import）

**Sharing Memory Handle** 当 handle 被导出之后，必须通过 inter-process communication 协议将其共享给接收方的一个或多个 process。开发者可以自由选择共享 handle 的方式，具体采用哪种 IPC 方法取决于应用的设计和运行环境。常见的方法包括 OS-specific 的 inter-process socket 以及分布式的 message passing。使用 OS-specific IPC 可以获得较高的传输性能，但仅限于同一台机器上的 process，且不具备可移植性。fabric-specific IPC 更加简单，也更具可移植性，但它依赖 system-level 的支持。所选择的方法必须能够安全、可靠地将 handle 数据传输给目标 process，使其能够导入内存并建立有效的映射。IPC 方法选择上的灵活性，使 VMM API 能够集成到各种 system architecture 中，从 single-node 应用到分布式的 multi-node 场景。在下面的代码示例中，将分别给出使用 socket 编程和 MPI 来共享与接收 handle 的示例。

```C++
// Send: OS-Specific IPC (Linux)
int ipcSendShareableHandle(int socket, int fd, pid_t process) {
    struct msghdr msg;
    struct iovec iov[1];

    union {
        struct cmsghdr cm;
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);

    struct cmsghdr *cmptr;
    ssize_t readResult;
    struct sockaddr_un cliaddr;
    socklen_t len = sizeof(cliaddr);

    // Construct client address to send this SHareable handle to
    memset(&cliaddr, 0, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    char temp[20];
    sprintf(temp, "%s%u", "/tmp/", process);
    strcpy(cliaddr.sun_path, temp);
    len = sizeof(cliaddr);

    // Send corresponding shareable handle to the client
    int sendfd = fd;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    msg.msg_name = (void *)&cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base = (void *)"";
    iov[0].iov_len = 1;
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    ssize_t sendResult = sendmsg(socket, &msg, 0);
    if (sendResult <= 0) {
        perror("IPC failure: Sending data over socket failed");
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}
```

```c++
// Receive: OS-Specific IPC (Linux)
int ipcRecvShareableHandle(int socket, int* fd) {
    struct msghdr msg = {0};
    struct iovec iov[1];
    struct cmsghdr cm;

    // Union to guarantee alignment requirements for control array
    union {
        struct cmsghdr cm;
        // This will not work on QNX as QNX CMSG_SPACE calls __cmsg_alignbytes
        // And __cmsg_alignbytes is a runtime function instead of compile-time macros
        // char control[CMSG_SPACE(sizeof(int))]
        char* control;
    } control_un;

    size_t sizeof_control = CMSG_SPACE(sizeof(int)) * sizeof(char);
    control_un.control = (char*) malloc(sizeof_control);
    struct cmsghdr *cmptr;
    ssize_t n;
    int receivedfd;
    char dummy_buffer[1];
    ssize_t sendResult;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof_control;

    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    if ((n = recvmsg(socket, &msg, 0)) <= 0) {
        perror("IPC failure: Receiving data over socket failed");
        free(control_un.control);
        return -1;
    }

    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
        if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        free(control_un.control);
        return -1;
        }

        memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
        *fd = receivedfd;
    } else {
        free(control_un.control);
        return -1;
    }

    free(control_un.control);
    return 0;
}
```

```c++
// Send: Fabric IPC
MPI_Send(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
```

```c++
// Receive: Fabric IPC
MPI_Recv(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 1, 0, MPI_COMM_WORLD);
```

**Importing Memory Handle** 同样地，用户既可以导入用于 OS-specific IPC 的 handle，也可以导入用于 fabric-specific IPC 的 handle。OS-specific IPC handle 只能用于 single-node 场景，而 fabric-specific handle 可以用于 single-node 或 multi-node 场景。

```c++
// OS-Specific Handle (Linux)
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
cuMemImportFromShareableHandle(handle, (void*) &fd, handleType);
```

```c++
// Fabric Handle
CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;
cuMemImportFromShareableHandle(handle, (void*) &fh, handleType);
```

#### 4.16.3.3 保留与映射（Reserve and Map）

**Reserving a Virtual Address Range**

 由于在 VMM 中 address 与 memory 是相互区分的概念，应用必须预先划出一段 address range，用来容纳通过 cuMemCreate 创建的内存分配。所保留的 address range 大小，至少要等于用户计划放入其中的所有 physical memory allocation 的大小总和。

应用可以通过向 cuMemAddressReserve 传入合适的参数来保留一段 virtual address range。获得的这段 address range 并不会关联任何 device 或 host 的 physical memory。该 virtual address range 可以映射到系统中任意 device 的 memory chunk 上，从而为应用提供一段连续的 VA range，而其底层可以由来自不同 device 的内存进行支撑和映射。应用在使用完成后，应当通过 cuMemAddressFree 将这段 virtual address range 归还给 CUDA。用户必须确保在调用 cuMemAddressFree 之前，整个 VA range 已经被完全 unmapped。从概念上看，这些函数与 Linux 中的 mmap / munmap，或 Windows 中的 VirtualAlloc / VirtualFree 是类似的。下面的代码示例展示了该函数的使用方式：

```c++
CUdeviceptr ptr;
// `ptr` 保存所保留的 virtual address range 的起始地址
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0
                                                          // 表示使用默认对齐方式
```

**Mapping Memory**

 前面两步中分配的 physical memory 与划出的 virtual address space，体现了 VMM API 引入的 memory 与 address 的区分。要让已分配的内存变得可用，用户必须将该内存映射到 address space 中。

通过 cuMemAddressReserve 获得的 address range，以及通过 cuMemCreate 或 cuMemImportFromShareableHandle 获得的 physical allocation，必须通过 cuMemMap 关联到一起。只要预留了足够的 address space，用户就可以将来自多个 device 的 allocation 关联到连续的 virtual address range 中。为了将 physical allocation 与 address range 解耦，用户需要使用 cuMemUnmap 对映射进行解除。

只要预先划出了足够的 address space，用户就可以将来自多个 device 的 allocation 关联到连续的 virtual address range 中。为了将 physical allocation 与 address range 解耦，用户需要使用 cuMemUnmap 对映射的地址进行解除。用户可以在同一个 address range 上反复进行 memory 的 map 和 unmap 操作，只要确保不会在已经处于 mapped 状态的 VA range reservation 上再次尝试创建映射即可。下面的代码示例展示了该函数的使用方式：

```c++
CUdeviceptr ptr;
// `ptr`: address in the address range previously reserved by cuMemAddressReserve.
// `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate.
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);
```

#### 4.16.3.4 访问权限（Access Rights）

CUDA 的 virtual memory management API 允许应用通过访问控制机制，对其 VA range 进行显式保护。仅使用 cuMemMap 将 allocation 映射到 address range 的某个区域，并不会让该地址变得可访问，如果 CUDA kernel 访问该地址，将会导致程序崩溃。用户必须在 source device 和访问该内存的 device 上，通过 cuMemSetAccess 函数显式设置访问控制。这样可以允许或限制特定 device 对某个已映射 address range 的访问。下面的代码示例展示了该函数的使用方式：

```c++
void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Make the address accessible
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}
```

VMM 提供的访问控制机制，使用户能够明确指定哪些 allocation 可以与系统中的其他 peer device 共享。如前所述，cudaEnablePeerAccess 会强制将所有此前以及之后通过 cudaMalloc 创建的 allocation 映射到目标 peer device 上。这在很多情况下使用起来比较方便，因为用户无需关心系统中每个 allocation 到每个 device 的映射状态。但这种做法会带来性能方面的影响。通过在 allocation 粒度上进行访问控制，VMM 能够以最小的开销实现 peer mapping。

vectorAddMMAP sample 可以作为使用 Virtual Memory Management API 的示例参考。

#### 4.16.3.5 释放内存（Releasing the Memory）

为了释放已分配的内存和 address space，source process 和 target process 都应当按顺序调用 cuMemUnmap、cuMemRelease 和 cuMemAddressFree。cuMemUnmap 会将之前映射到 address range 的内存区域解除映射，从而把 physical memory 与已保留的 virtual address space 分离。接着，cuMemRelease 会释放之前创建的 physical memory，并将其归还给系统。最后，cuMemAddressFree 会释放之前保留的 virtual address range，使其可以被后续再次使用。按照这一特定顺序操作，可以确保 physical memory 和 virtual address space 都被干净、完整地释放。

```c++
cuMemUnmap(ptr, size);
cuMemRelease(handle);
cuMemAddressFree(ptr, size);
```

> [!注意]
> 在 OS-specific 的场景下，导出的 handle 必须通过 fclose 进行关闭。该步骤不适用于 fabric-based 的场景。

### 4.16.4 多播内存共享（Multicast Memory Sharing）

Multicast Object Management API 为应用提供了一种创建 multicast object 的方式，并且可以与前面介绍的 Virtual Memory Management API 结合使用，使应用能够在通过 NVSwitch 连接、且支持 NVLink 的 GPU 上利用 NVLink SHARP。NVLink SHARP 允许 CUDA 应用利用 fabric 内计算能力，加速通过 NVSwitch 连接的 GPU 之间的 broadcast 和 reduction 等操作。为了实现这一点，多个通过 NVLink 连接的 GPU 会组成一个 multicast team，team 中的每个 GPU 都会使用 physical memory 为同一个 multicast object 提供支撑。因此，一个由 N 个 GPU 组成的 multicast team 会拥有 N 份 multicast object 的 physical replica，每一份都本地存在于对应的参与 GPU 上。使用 multicast object 映射的 multimem PTX instruction 会同时作用于该 multicast object 的所有 replica。

要使用 multicast object，应用需要完成以下步骤：

- 查询是否支持 multicast；
- 使用 cuMulticastCreate 创建一个 multicast handle；
- 将该 multicast handle 共享给所有控制着参与 multicast team 的 GPU 的 process，这一步可以像前面所述那样，通过 cuMemExportToShareableHandle 实现；
- 使用 cuMulticastAddDevice 将所有需要参与 multicast team 的 GPU 添加进去；
- 对于每个参与的 GPU，将前面通过 cuMemCreate 分配的 physical memory 绑定到该 multicast handle 上。在对任何 device 进行内存绑定之前，必须先把所有 device 都加入 multicast team。
- 按照前面介绍的 unicast mapping 的方式，保留 address range、映射 multicast handle，并设置 access rights。可以同时对同一块 physical memory 建立 unicast 和 multicast 映射。关于如何保证同一块 physical memory 的多重映射之间的一致性，请参见前面的 Virtual Aliasing Support 章节；
- 使用 multicast mapping 配合 multimem PTX instruction。

Multi GPU Programming Models GitHub 仓库中的 multi_node_p2p 示例，提供了一个完整示例，展示了如何结合 fabric memory 和 multicast object 来利用 NVLink SHARP。需要注意的是，该示例主要面向像 NCCL 或 NVSHMEM 这类 library 的开发者，它展示了类似 NVSHMEM 这样的高层编程模型在（multi-node）NVLink domain 内部是如何工作的。对于应用开发者而言，通常应当优先使用更高层的 MPI、NCCL 或 NVSHMEM 接口，而不是直接使用这一 API。

#### 4.16.4.1 分配 Multicast Object（Allocating Multicast Objects）

可以通过 cuMulticastCreate 来创建 multicast object：

```c++
CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // 或者在 single-node 场景下
                                                    // 使用 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp,
                              CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // 确保 size 满足该分配的 granularity 对齐要求
    size_t padded_size = ROUND_UP(size, granularity);

    mcProp.size = padded_size;
    // 创建 Multicast Object，此时尚未关联任何 device
    // 也尚未绑定任何 physical memory
    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);
    return mcHandle;
}
```

#### 4.16.4.2 向 Multicast Object 添加 Device

可以通过 cuMulticastAddDevice 将 device 添加到 multicast team 中：

```c++
cuMulticastAddDevice(&mcHandle, device);
```

在对任何 device 进行内存绑定之前，所有控制着参与 multicast team 的 device 的 process，都必须先完成这一步。

#### 4.16.4.3 将内存绑定到 Multicast Object（Bind Memory to Multicast Objects）

在创建 multicast object 并且将所有参与的 device 都加入 multicast object 之后，需要为每个 device 使用 cuMemCreate 分配的 physical memory 来为该 multicast object 提供支撑：

```c++
cuMulticastBindMem(mcHandle, mcOffset,
                   memHandle, memOffset,
                   size, 0 /*flags*/);
```

#### 4.16.4.4 使用 Multicast 映射（Use Multicast Mappings）

要在 CUDA C++ 中使用 multicast mapping，需要结合 inline PTX 来使用 multimem PTX instruction。

```c++
__global__ void all_reduce_norm_barrier_kernel(float* l2_norm,
                                               float* partial_l2_norm_mc,
                                               unsigned int* arrival_counter_uc,
                                               unsigned int* arrival_counter_mc,
                                               const unsigned int expected_count) {
    assert( 1 == blockDim.x * blockDim.y * blockDim.z *
                    gridDim.x * gridDim.y * gridDim.z );
    float l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900
    // 对所有 replica 执行 atomic reduction
    // 从概念上可以理解为一次 __threadfence_system()
    atomicAdd_system(arrival_counter_mc, 1);
    cuda::ptx::multimem_red(cuda::ptx::release_t,
                            cuda::ptx::scope_sys_t,
                            cuda::ptx::op_add_t,
                            arrival_counter_mc,
                            1);

    // 在 Multicast（mc）和 Unicast（uc）对同一块内存
    // `arrival_counter_uc` 和 `arrival_counter_mc`
    // 的访问之间，需要插入一个 fence：
    // - fence.proxy 指令用于在可能通过不同 proxy
    //   发生的内存访问之间建立顺序关系
    // - .proxykind 修饰符中的 .alias 表示这些访问
    //   是通过不同的虚拟别名地址访问同一内存位置
    // 参考：
    // https://docs.nvidia.com/cuda/parallel-thread-execution/
    // #parallel-synchronization-and-communication-instructions-membar
    cuda::ptx::fence_proxy_alias();

    // 在 UC 映射上以 acquire 语义进行自旋等待，
    // 直到所有 peer 都在本轮迭代中到达
    // 注意：所有 rank 在该 kernel 之后还需要
    // 再次到达一个 barrier，这样可以避免
    // 某个 rank 较慢时，被下一轮迭代提前解锁
    cuda::atomic_ref<unsigned int,
                     cuda::thread_scope_system> ac(arrival_counter_uc);
    while (expected_count > ac.load(cuda::memory_order_acquire));

    // 从所有 replica 执行 atomic load reduction
    // 该操作不提供顺序保证，因此可以使用 relaxed
    asm volatile("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];"
                 : "=f"(l2_norm_sum)
                 : "l"(partial_l2_norm_mc)
                 : "memory");
#else
#error "ERROR: multimem instruction 需要 compute capability 9.0 或更高"
#endif
    *l2_norm = std::sqrt(l2_norm_sum);
}
```

### 4.16.5 高级配置

#### 4.16.5.1. 内存类型

VMM 还提供了一种机制，让应用程序可以分配某些设备可能支持的特殊类型内存。使用 `cuMemCreate` 时，应用程序可以通过 `CUmemAllocationProp::allocFlags` 指定内存类型的要求，以选择特定的内存功能。应用程序必须确保所请求的内存类型是设备支持的。

#### 4.16.5.2. 可压缩内存

可压缩内存可用于加速对具有无结构稀疏性和其他可压缩数据模式的数据的访问。压缩可以根据数据情况节省 DRAM 带宽、L2 读取带宽和 L2 容量。想要在支持计算数据压缩的设备上分配可压缩内存的应用程序，可以将 `CUmemAllocationProp::allocFlags::compressionType` 设置为 `CU_MEM_ALLOCATION_COMP_GENERIC`。用户必须使用 `CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED` 查询设备是否支持 Compute Data Compression。下面的代码片段展示了如何用 `cuDeviceGetAttribute` 查询对可压缩内存的支持。

```c++
int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);
```

在支持计算数据压缩的设备上，用户必须在分配时选择启用，如下面所示。

```c++
prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
```

由于硬件资源有限等多种原因，该内存分配可能没有压缩属性。要验证这些标志是否生效，用户可以使用 `cuMemGetAllocationPropertiesFromHandle` 查询已分配内存的属性。

```c++
CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // Obtained compressible memory allocation
}
```

#### 4.16.5.3. Virtual Aliasing Support / 虚拟别名支持

虚拟内存管理 API 提供了一种方法，可以通过多次调用 `cuMemMap` 并使用不同的虚拟地址，为同一内存分配创建多个虚拟内存映射或“代理”。这称为虚拟别名。除非 PTX ISA 明确说明，否则对分配的一个代理的写操作，在写入的设备操作（如 grid 启动、memcpy、memset 等）完成之前，被认为与同一内存的其他代理不一致且不具备内存一致性。

例如，假设设备指针 A 和 B 是对同一内存分配的虚拟别名，则下面的代码段被视为未定义行为。

```c++
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  printf("%d\n", *B);    // 未定义行为！*B 可能是之前的值，也可能是介于中间的某个值。
}
```

如果两个 kernel 的执行按照单调顺序（通过 stream 或 event 控制），下面的行为是定义好的。

```c++
__global__ void foo1(char *A) {
  *A = 0x1;
}

__global__ void foo2(char *B) {
  printf("%d\n", *B);    // 假设在启动 foo2 之前等待 foo1 执行完成，
                        // 那么 *B == *A == 0x1
}

cudaMemcpyAsync(B, input, size, stream1);    // 在操作边界处允许别名，
                                             // 因此允许 foo1 访问 A
foo1<<<1,1,0,stream1>>>(A);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);   // foo2 的两次启动以及
                                             // cudaMemcpy（它们都只读）
                                             // 都会等待 foo1（写操作）
                                             // 执行完成后再继续
```

如果需要在同一个 kernel 中通过不同的 “proxies” 访问同一块 allocation，可以在这两次访问之间使用 fence.proxy.alias。这样一来，上面的示例就可以通过内联 PTX 汇编变成合法的：

```c++
__global__ void foo(char *A, char *B) {
  *A = 0x1;
  cuda::ptx::fence_proxy_alias();
  printf("%d\n", *B);    // *B == *A == 0x1
}
```

#### 4.16.5.4. IPC 的 OS-Specific Handle 细节（OS-Specific Handle Details for IPC）

通过 `cuMemCreate`，用户可以在分配时就表明某个特定的 allocation 是为 inter-process communication 或 graphics interop 预留的。应用程序可以通过将 `CUmemAllocationProp::requestedHandleTypes` 设置为平台相关的字段来做到这一点。在 Windows 上，当 `CUmemAllocationProp::requestedHandleTypes` 被设置为 `CU_MEM_HANDLE_TYPE_WIN32` 时，应用程序还必须在 `CUmemAllocationProp::win32HandleMetaData` 中指定一个 LPSECURITYATTRIBUTES 属性。这个 security attribute 用来定义哪些被导出的 allocation 可以被传递给其他进程。

用户必须在尝试导出通过 `cuMemCreate` 分配的 memory 之前，先确认系统是否支持所请求的 handle type。下面的代码片段展示了如何以平台相关的方式查询对 handle type 的支持情况。

```c++
int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif
```

用户应当按照下面所示，正确地设置 `CUmemAllocationProp::requestedHandleTypes`：

```c++
#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
#endif
```

