# 4.17 Extended GPU Memory

### 4.17 扩展 GPU 内存（Extended GPU Memory）

扩展 GPU 内存（EGM）特性利用高带宽的 NVLink-C2C，使 GPU 能够高效地访问所有系统内存，无论是在单节点系统还是多节点系统中。EGM 适用于集成式 CPU-GPU 的 NVIDIA 系统，通过允许物理内存分配，从而使任何 GPU 线程都能够访问这些内存资源。EGM 确保所有 GPU 都可以通过 GPU-GPU NVLink 或 NVLink-C2C 的速度访问这些资源。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/egm-c2c-intro.png)

在这种配置下，本地内存访问通过高带宽的 NVLink-C2C 完成。对于远程内存访问，使用 GPU NVLink，并且在某些情况下，也可以使用 NVLink-C2C。通过 EGM，GPU 线程能够访问所有可用的内存资源，包括连接到 CPU 的内存和 HBM3 内存，这些都可以通过 NVSwitch 互连访问。

### 4.17.1. 前言

在深入探讨 EGM 功能的 API 变化之前，我们将介绍当前支持的拓扑结构、标识符分配、虚拟内存管理的前提条件以及 EGM 的 CUDA 类型。

#### 4.17.1.1. EGM 平台：系统拓扑结构

目前，EGM 可以在多个平台中启用：
 (1) **单节点，单 GPU**：由 Arm 架构的 CPU、CPU 附加内存和 GPU 组成。CPU 和 GPU 之间有一个高带宽的 C2C（芯片到芯片）互连。
 (2) **单节点，多 GPU**：由每个具有附加内存的 ARM 架构 CPU 和通过 NVLink 基于网络连接的多个 GPU 组成。
 (3) **多节点，多 GPU**：由两个或更多单节点系统组成，每个系统如同（1）或（2）所述，通过 NVLink 基于网络连接。

> **注释**
>  使用 `cgroups` 限制可用设备将会阻止通过 EGM 路由，从而导致性能问题。请改为使用 `CUDA_VISIBLE_DEVICES`。

### 4.17.1.2. 套接字标识符：它们是什么？如何访问它们？

NUMA（非统一内存访问）是一种在多处理器计算机系统中使用的内存架构，允许内存分为多个节点。每个节点都有自己的处理器和内存。在这样的系统中，NUMA 将系统分为多个节点并为每个节点分配一个唯一的标识符（`numaId`）。

EGM 使用由操作系统分配的 NUMA 节点标识符。需要注意的是，这个标识符不同于设备的序号，并且与最近的主机节点相关联。除了现有的方法外，用户还可以通过调用 `cuDeviceGetAttribute` 并使用 `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` 属性类型来获取主机节点的标识符（`numaId`），如下所示：

```c
int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);
```

### 4.17.1.3. 分配器与 EGM 支持

将系统内存映射为 EGM 不会导致性能问题。事实上，访问作为 EGM 映射的远程套接字的系统内存将更快。因为，EGM 流量保证通过 NVLink 路由。目前，`cuMemCreate` 和 `cudaMemPoolCreate` 分配器支持与适当的定位类型和 NUMA 标识符一起使用。

### 4.17.1.4. 对现有 API 的内存管理扩展

目前，EGM 内存可以通过虚拟内存（`cuMemCreate`）或流顺序内存（`cudaMemPoolCreate`）分配器进行映射。用户负责分配物理内存并将其映射到所有套接字的虚拟内存地址空间。

> **注释**
>  多节点、多 GPU 平台需要进程间通信。因此，我们鼓励读者参见第 4.15 章。

> **注释**
>  我们鼓励读者阅读 CUDA 编程指南中的第 4.16 章和第 4.3 章，以便更好地理解。

新的 CUDA 属性类型已添加到 API 中，以允许这些方法使用类似 NUMA 的节点标识符来理解分配位置：

| CUDA类型                         | 与之配合使用                               |
| -------------------------------- | ------------------------------------------ |
| `CU_MEM_LOCATION_TYPE_HOST_NUMA` | `CUMemAllocationProp` for `cuMemCreate`    |
| `cudaMemLocationTypeHostNuma`    | `cudaMemPoolProps` for `cudaMemPoolCreate` |

> **注释**
>  请参阅 CUDA 驱动程序 API 和 CUDA 运行时数据类型，以了解有关 NUMA 特定 CUDA 类型的更多信息。

### 4.17.2. 使用 EGM 接口

#### 4.17.2.1. 单节点，单 GPU

现有的 CUDA host 分配器以及系统分配的内存均可用于受益于高带宽的 C2C。对用户来说，local 访问就是今天的 host 分配。

> **注释**
>  有关内存分配器和页大小的更多信息，请参阅调优指南。

#### 4.17.2.2. 单节点，多 GPU

在多 GPU 系统中，用户需要提供主机信息以进行放置。如前所述，表达这些信息的自然方式是使用 NUMA 节点 ID，而 EGM 正是遵循这种方法。因此，使用 `cuDeviceGetAttribute` 函数，用户应能够得知最近的 NUMA 节点 ID。（参见 **套接字标识符：它们是什么？如何访问它们？**）接着，用户可以使用 VMM（虚拟内存管理）API 或 CUDA 内存池分配器来分配和管理 EGM 内存。

#### 4.17.2.1. 使用 VMM API

使用虚拟内存管理 API 进行内存分配的第一步是创建一个物理内存块来为分配提供后备。有关更多细节，请参见 CUDA 编程指南中的 **虚拟内存管理** 部分。在 EGM 分配中，用户必须显式提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型，并提供 `numaId` 作为位置标识符。在 EGM 中，分配必须对平台的适当粒度进行对齐。以下代码片段展示了如何使用 `cuMemCreate` 分配物理内存：

```c
CUMemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t_t padded_size = ROUND_UP(size, granularity);
CUMemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

在物理内存分配后，我们需要保留地址空间并将其映射到指针。这些过程没有 EGM 特定的变化：

```c
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

最后，用户必须显式保护映射虚拟地址范围。否则，访问映射空间会导致崩溃。与内存分配类似，用户需要提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型，并提供 `numaId` 作为位置标识符。以下代码片段为主机节点和 GPU 创建访问描述符，以便给映射内存的两者提供读写访问权限：

```
CUMemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = currentDev;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, accessDesc, 2);
```

------

##### 4.17.2.2.2. 使用 CUDA 内存池

为了定义 EGM，用户可以在一个节点上创建一个内存池并授予对等设备访问权限。在这种情况下，用户必须显式定义 `cudaMemLocationTypeHostNuma` 作为位置类型，并提供 `numaId` 作为位置标识符。以下代码片段展示了如何创建一个内存池 `cudaMemPoolCreate`：

```c
cudaSetDevice(homeDevice);
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;
cudaMemPoolCreate(&memPool, &props);
```

此外，对于直接连接的对等访问，也可以使用现有的对等访问 API，`cudaMemPoolSetAccess`。以下代码片段展示了如何为 `accessingDevice` 设置访问：

```c
cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = accessingDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);
```

创建内存池并授予访问权限后，用户可以将创建的内存池分配给 `residentDevice` 并使用 `cudaMallocAsync` 开始分配内存：

```c
cudaDeviceSetMemPool(residentDevice, memPool);
cudaMallocAsync(&ptr, size, memPool, stream);
```

> **注释**
>  EGM 使用 2MB 页。因此，用户在访问非常大的分配时，可能会遇到更多的 TLB 缺失。

#### 4.17.2.3. 多节点，多 GPU

除了内存分配，远程对等访问没有 EGM 特定的修改，并且遵循 CUDA 进程间（IPC）协议。有关 IPC 的更多详细信息，请参见 CUDA 编程指南。

用户应使用 `cuMemCreate` 分配内存，并且用户需要显式提供 `CU_MEM_LOCATION_TYPE_HOST_NUMA` 作为位置类型，`numaId` 作为位置标识符。此外，`CU_MEM_HANDLE_TYPE_FABRIC` 应该作为所请求的句柄类型进行定义。以下代码片段展示了在 Node A 上分配物理内存：

```c
CUMemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUMemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);
```

使用 `cuMemCreate` 创建分配句柄后，用户可以通过 `cuMemExportToShareableHandle` 将该句柄导出到其他节点 Node B：

```c
cuMemExportToShareableHandle(&fabricHandle, allocHandle,
                             CU_MEM_HANDLE_TYPE_FABRIC, 0);
// 此时，fabricHandle 应该通过 TCP/IP 发送到 Node B。
```

在 Node B 上，句柄可以通过 `cuMemImportFromShareableHandle` 导入，并作为任何其他 fabric 句柄进行处理：

```c
// 此时，fabricHandle 应该通过 TCP/IP 从 Node A 接收。
CUMemGenericAllocationHandle allocHandle;
cuMemImportFromShareableHandle(&allocHandle, &fabricHandle,
                               CU_MEM_HANDLE_TYPE_FABRIC);
```

当在 Node B 导入句柄后，用户可以按照常规方式保留地址空间并将其映射到本地：

```c
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                               MEM_ALLOC_GRANULARITY_MINIMUM);

size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);
```

最后一步，用户应为 Node B 上的每个本地 GPU 提供适当的访问权限。以下代码片段为给定八个本地 GPU 读写权限的示例：

```c
// 给所有 8 个本地 GPU 访问位于 Node A 上的导出 EGM 内存。
CUMemAccessDesc accessDesc[8];
for (int i = 0; i < 8; i++) {
  accessDesc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc[i].location.id = i;
  accessDesc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(dptr, size, accessDesc, 8);
```
