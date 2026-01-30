# 4.3 Stream-Ordered Memory Allocator

### 4.3.1. Introduction

使用 `cudaMalloc` 和 `cudaFree` 进行内存分配管理会导致 GPU 在所有正在执行的 CUDA streams 之间发生同步。Stream-ordered memory allocator 允许应用程序将内存的分配与释放，与其他被 launch 到 CUDA stream 中的工作（例如 kernel launch 和异步 copy）进行顺序关联。通过利用 stream 的顺序语义来复用内存分配，可以提升应用程序的内存使用效率。该 allocator 还允许应用程序控制其内存缓存行为。当设置合适的释放阈值时，这种缓存机制可以在应用程序愿意接受更大内存占用的情况下，避免代价高昂的 OS 调用。此外，该 allocator 还支持在进程之间方便且安全地共享分配的内存。

Stream-Ordered Memory Allocator：

> - 一定程度上帮助用户实现对自定义内存管理抽象的需求，同时让那些确实需要自定义高性能内存管理的应用程序更容易实现。
> - 使多个库能够共享由 driver 管理的公共 memory pool，从而减少多余的内存消耗。
> - 允许 driver 基于其对 allocator 及其他 stream 管理 API 的理解执行优化。

> Note
>
> 自 CUDA 11.3 起，Nsight Compute 和 Next-Gen CUDA debugger 已支持并识别该 allocator。

------

### 4.3.2. Memory Management

`cudaMallocAsync` 和 `cudaFreeAsync` 是支持 stream-ordered 内存管理的 API。`cudaMallocAsync` 用于返回一块分配的内存，而 `cudaFreeAsync` 用于释放一块已分配的内存。这两个 API 都接受一个 stream 参数，用来定义该分配在何时开始可用以及何时停止可用。这些函数使内存操作能够绑定到特定的 CUDA streams 上，从而在不阻塞 host 或其他 streams 的情况下执行。通过避免 `cudaMalloc` 和 `cudaFree` 可能带来的高成本同步，可以提升应用程序性能。

这些 API 还可以通过 memory pools 实现进一步的性能优化。memory pool 通过管理和复用大块内存，实现更高效的分配和释放。memory pools 有助于减少开销并防止内存碎片，在频繁进行内存分配操作的场景下能够提升性能。

#### 4.3.2.1. Allocating Memory

`cudaMallocAsync` 函数会在 GPU 上触发一次与特定 CUDA stream 关联的异步内存分配。`cudaMallocAsync` 允许内存分配在不阻塞 host 或其他 streams 的情况下进行，从而消除对昂贵同步操作的需求。

> Note
>
> 在确定分配将驻留在哪个 device 时，`cudaMallocAsync` 会忽略当前的 device/context。相反，它会根据指定的 memory pool 或提供的 stream 来确定合适的 device。

下面的示例展示了一个基本的使用模式：内存被分配，在同一个 stream 中使用，然后再释放回该 stream。

```c
void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// 使用该分配的内存执行工作
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// 可以在不对 cpu 和 GPU 进行同步的情况下指定异步释放
cudaFreeAsync(ptr, cudaStreamPerThread);
```

> Note
>
> 当从一个与执行 allocation 的 stream 不同的 stream 访问该 allocation 时，用户必须保证该访问发生在 allocation 操作之后，否则行为是未定义的。

#### 4.3.2.2. 释放内存

`cudaFreeAsync()` 以 stream-ordered 方式异步释放 device memory，这意味着内存释放操作被安排到指定的 CUDA stream 上，并且不会阻塞 host 或其他 streams。

用户必须保证 free 操作发生在 allocation 操作之后，并且在该内存的任何使用之后。一旦 free 操作开始后再使用这块内存，将导致未定义行为。

应该使用 events 和/或 stream 同步操作来保证其他 streams 对这块内存的访问完成后，再开始 free 操作，如下面的示例所示。

```c
cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);
// stream2 必须等待 allocation 完成后才能访问
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);
// stream3 必须等待 stream2 完成对该 allocation 的访问后，才能释放
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);
```



通过 `cudaMalloc()` 分配的内存也可以通过 `cudaFreeAsync()` 来释放。如上所述，在 free 操作开始之前，所有对该内存的访问必须完成。

```
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);
```



同样，通过 `cudaMallocAsync` 分配的内存也可以通过 `cudaFree()` 来释放。当通过 `cudaFree()` API 释放这种 allocation 时，driver 会假定对该 allocation 的所有访问已经完成，并且不会执行额外的同步。用户可以使用 `cudaStreamQuery` / `cudaStreamSynchronize` / `cudaEventQuery` / `cudaEventSynchronize` / `cudaDeviceSynchronize` 来保证相关的异步工作已经完成，并且 GPU 不会再尝试访问该 allocation。

```c
cudaMallocAsync(&ptr, size, stream);
kernel<<<..., stream>>>(ptr, ...);
// 需要进行 synchronize，以避免过早释放内存
cudaStreamSynchronize(stream);
cudaFree(ptr);
```

### 4.3.3. Memory Pools

Memory pools 封装了 virtual address 和 physical memory 资源，这些资源会根据 pool 的 attributes 和 properties 进行分配与管理。memory pool 的核心特性在于它所管理的 memory 类型及其所在位置。

所有对 `cudaMallocAsync` 的调用都会从 memory pool 中获取资源。如果没有显式指定 memory pool，`cudaMallocAsync` 会使用所提供 stream 所属 device 的当前 memory pool。device 的当前 memory pool 可以通过 `cudaDeviceSetMempool` 设置，并通过 `cudaDeviceGetMempool` 查询。每个 device 都有一个 default memory pool；如果未调用 `cudaDeviceSetMempool`，则该 default pool 处于激活状态。

API `cudaMallocFromPoolAsync` 以及 c++ 版本的 `cudaMallocAsync` 重载允许用户在不将其设为当前 pool 的情况下，为一次 allocation 指定所使用的 pool。API `cudaDeviceGetDefaultMempool` 和 `cudaMemPoolCreate` 返回 memory pool 的 handle。`cudaMemPoolSetAttribute` 与 `cudaMemPoolGetAttribute` 用于控制 memory pool 的 attributes。

> Note
> 某个 device 当前的 mempool 仅对该 device 本地有效。因此，在未指定 memory pool 的情况下进行 allocation，始终会得到一个位于该 stream 所属 device 上的 allocation。

#### 4.3.3.1. Default/Implicit Pools

可以通过调用 `cudaDeviceGetDefaultMempool` 获取某个 device 的 default memory pool。从 device 的 default memory pool 分配得到的是不可迁移的 device allocation，并且位于该 device 上。这些 allocation 始终可以从该 device 访问。default memory pool 的可访问性可以通过 `cudaMemPoolSetAccess` 修改，并通过 `cudaMemPoolGetAccess` 查询。由于 default pools 不需要显式 create，因此也被称为 implicit pools。device 的 default memory pool 不支持 IPC。

#### 4.3.3.2. Explicit Pools

`cudaMemPoolCreate` 用于创建 explicit pool。这使得应用程序可以为其 allocation 请求超出 default/implicit pools 所提供范围之外的属性。这些属性包括 IPC 能力、最大 pool 大小、在支持的平台上驻留于特定 CPU NUMA node 的 allocation 等。

```c
// 在 device 0 上创建一个类似 implicit pool 的 pool
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPoolCreate(&memPool, &poolProps));
```

下面的代码示例演示了如何在一个有效的 CPU NUMA node 上创建一个支持 IPC 的 memory pool。

```c
// 创建一个驻留在 CPU NUMA node 上、并支持通过文件描述符进行 IPC 共享的 pool
int cpu_numa_id = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

cudaMemPoolCreate(&ipcMemPool, &poolProps));
```

#### 4.3.3.3. Device Accessibility for Multi-GPU Support

与通过 virtual memory management APIs 控制 allocation 可访问性不同，memory pool 的 allocation 可访问性不遵循 `cudaDeviceEnablePeerAccess` 或 `cuCtxEnablePeerAccess`。对于 memory pool，API `cudaMemPoolSetAccess` 用于修改哪些 devices 可以访问某个 pool 中的 allocation。默认情况下，allocation 仅能从其所在的 device 访问，并且这种访问权限不能被撤销。
 若要启用其他 device 的访问权限，访问方 device 必须与 memory pool 所在 device 具备 peer capability。可以通过 `cudaDeviceCanAccessPeer` 进行验证。如果未检查 peer capability，则设置访问权限可能会因 `cudaErrorInvalidDevice` 而失败。不过，如果在该 pool 中尚未进行任何 allocation，即使 devices 不具备 peer capability，`cudaMemPoolSetAccess` 调用也可能成功。在这种情况下，下一次从该 pool 进行 allocation 将会失败。
 需要注意的是，`cudaMemPoolSetAccess` 会影响该 memory pool 中的所有 allocation，而不仅仅是之后的 allocation。同样，`cudaMemPoolGetAccess` 返回的可访问性信息也适用于该 pool 中的所有 allocation，而不仅仅是未来的 allocation。不建议频繁更改某个 pool 针对特定 GPU 的可访问性设置。也就是说，一旦某个 pool 被设置为可从某个 GPU 访问，那么在该 pool 的整个生命周期内，都应保持对该 GPU 可访问。

```c
// snippet showing usage of cudaMemPoolSetAccess:
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
              int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
              residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    // Make the address accessible
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}
```

#### 4.3.3.4. 为 IPC 启用 Memory Pools

Memory pools 可以启用于进程间通信（IPC），从而在不同进程之间实现对 GPU memory 的便捷、高效且安全的共享。CUDA 的 IPC memory pools 提供与 CUDA virtual memory management APIs 相同的安全特性。

使用 memory pools 在进程之间共享内存需要两个步骤：首先，进程需要共享对该 pool 的访问权限；然后，再共享该 pool 中的具体分配。第一步用于建立并强制执行安全策略。第二步用于协调各个进程中使用的 virtual addresses，以及在 importing 进程中何时需要保证映射有效。

##### 4.3.3.4.1. 创建和共享 IPC Memory Pools

共享对 pool 的访问包括：使用 `cudaMemPoolExportToShareableHandle()` 获取该 pool 的 OS 原生句柄，通过 OS 原生的 IPC 机制将该句柄传输到 importing 进程，然后使用 `cudaMemPoolImportFromShareableHandle()` API 创建一个导入的 memory pool。要使 `cudaMemPoolExportToShareableHandle` 调用成功，memory pool 在创建时必须在 pool 属性结构中指定所请求的句柄类型。

请参考 samples 了解用于在进程之间传输 OS 原生句柄的合适 IPC 机制。其余流程如下方代码片段所示。

```c
// 在 exporting 进程中
// 在 device 0 上创建一个可导出的、支持 IPC 的 pool
cudaMemPoolProps poolProps = {};
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// 将 handleTypes 设置为非零值会使该 pool 可导出（支持 IPC）
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps);

// 基于 FD 的句柄是整数类型
int fdHandle = 0;

// 获取该 pool 的 OS 原生句柄
// 注意这里传入的是句柄内存的指针
cudaMemPoolExportToShareableHandle(&fdHandle,
                                   memPool,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                   0);

// 必须使用适当的 OS 特定 API
// 将该句柄发送给 importing 进程
// 在 importing 进程中
int fdHandle;
// 需要通过合适的 OS 特定 API
// 从 exporting 进程获取该句柄
// 从 shareable handle 创建一个导入的 pool
// 注意这里句柄是按值传递
cudaMemPoolImportFromShareableHandle(&importedMemPool,
                                     (void*)fdHandle,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                     0);
```

##### 4.3.3.4.2. 在 Importing 进程中设置访问权限

导入的 memory pools 初始时仅能从其所属的 device 访问。导入的 memory pool 不会继承 exporting 进程所设置的任何可访问性。importing 进程必须使用 `cudaMemPoolSetAccess` 为其计划访问该内存的任意 GPU 启用访问权限。

如果导入的 memory pool 属于一个 importing 进程不可见的 device，用户必须使用 `cudaMemPoolSetAccess` API，为将使用这些分配的 GPU 启用访问。（参见 Device Accessibility for Multi-GPU Support）

##### 4.3.3.4.3. 从导出的 Pool 创建和共享分配

一旦 pool 被共享，在 exporting 进程中通过 `cudaMallocAsync()` 从该 pool 分配的内存，可以被已导入该 pool 的其他进程共享。由于 pool 的安全策略是在 pool 级别建立和验证的，OS 无需为特定 pool 分配额外维护安全信息。换句话说，用于导入 pool 分配的 opaque `cudaMemPoolPtrExportData` 可以通过任意机制发送给 importing 进程。

虽然分配可以在不与分配所在 stream 做任何同步的情况下进行导出和导入，但 importing 进程在访问该分配时必须遵循与 exporting 进程相同的规则。具体来说，必须在分配所在 stream 中的分配操作执行完成之后，才能访问该分配。下面两个代码片段展示了如何使用 `cudaMemPoolExportPointer()` 和 `cudaMemPoolImportPointer()` 共享分配，并结合 IPC event 保证在 importing 进程中访问分配之前，该分配已准备就绪。

```c
// 在 exporting 进程中准备一个分配
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// 用于进程间协调的 ipc event
// cudaEventInterprocess 标志使该 event 成为 ipc event
// 出于性能原因设置 cudaEventDisableTiming

cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess);

// 从 exporting mem pool 分配
cudaMallocAsync(&ptr, size, exportMemPool, stream);

// 当分配准备就绪时用于共享的 event
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// 使用任意机制将 IPC event 和 pointer 导出数据
// 共享给 importing 进程，这里我们复制到共享内存
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// 通知消费者数据已准备好
// 导入一个分配
cudaMemPoolPtrExportData *importData = &shmem->ptrData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle = &shmem->readyIpcEventHandle;

// 需要通过任意机制从 exporting 进程
// 获取 ipc event 句柄和导出数据
// 这里我们使用 shmem，只需确保共享内存已填充

cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// 导入分配，该操作不会阻塞等待分配就绪
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// 在 importing 进程中使用该分配之前
// 需要等待 allocating stream 中之前的操作完成
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);
```

在释放分配时，必须先在 importing 进程中释放该分配，然后才能在 exporting 进程中释放。下面代码片段演示了如何使用 CUDA IPC events，在两个进程中的 `cudaFreeAsync` 操作之间提供所需的同步。显然，在 importing 进程侧执行 free 操作后，将限制该进程对该分配的访问。值得注意的是，也可以使用 `cudaFree` 在两个进程中释放该分配，并且可以使用其他 stream 同步 API 来替代 CUDA IPC events。

```c
// 必须先在 importing 进程中 free，再在 exporting 进程中 free
kernel<<<..., stream>>>(ptr, ...);

// importing 进程中的最后一次访问
cudaFreeAsync(ptr, stream);

// 在 importing 进程中 free 之后不允许再访问
cudaEventRecord(finishedIpcEvent, stream);
// Exporting 进程
// exporting 进程需要将其 free 操作
// 与 importing 进程 free 的 stream 顺序进行协调
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// importing 进程中的 free 不会阻止 exporting 进程
// 使用该分配
cudaFreeAsync(ptrInExportingProcess, stream);
```

##### 4.3.3.4.4. IPC Export Pool 限制

IPC pools 当前不支持将物理内存块释放回 OS。因此，`cudaMemPoolTrimTo` API 不起作用，`cudaMemPoolAttrReleaseThreshold` 也实际上被忽略。该行为由 driver 控制，而不是 runtime，将来可能在 driver 更新中发生变化。

##### 4.3.3.4.5. IPC Import Pool 限制

不允许从 import pool 进行分配；具体来说，import pools 不能被设为当前 pool，也不能用于 `cudaMallocFromPoolAsync` API。因此，分配重用策略属性对这些 pool 没有意义。

IPC import pools 与 IPC export pools 一样，目前都不支持将物理内存块释放回 OS。

资源使用统计属性查询仅反映导入到该进程中的分配及其关联的物理内存。

### 4.3.4. 最佳实践与调优

#### 4.3.4.1. 查询支持情况

应用程序可以通过调用 `cudaDeviceGetAttribute()`（参见 developer blog），并查询 device 属性 `cudaDevAttrMemoryPoolsSupported`，来判断某个 device 是否支持 stream-ordered memory allocator。

IPC memory pool 的支持情况可以通过 device 属性 `cudaDevAttrMemoryPoolSupportedHandleTypes` 查询。该属性在 CUDA 11.3 中新增；在更早版本的 driver 上查询该属性时会返回 `cudaErrorInvalidValue`。

```c
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
        cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // 'device' 支持 Stream-Ordered Memory Allocator
}

if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
        cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
    // 指定 device 上的 pool 可以使用基于 posix file descriptor 的 IPC 创建
}
```

在查询之前先进行 driver 版本检查，可以避免在尚未定义该属性的 driver 上触发 `cudaErrorInvalidValue` 错误。也可以通过调用 `cudaGetLastError` 清除该错误，而不是提前避免它。

#### 4.3.4.2. Physical Page 缓存行为

默认情况下，allocator 会尽量减少某个 pool 所持有的 physical memory。为了减少向 OS 申请和释放 physical memory 的调用次数，应用程序需要为每个 pool 配置一个 memory footprint。可以通过设置 release threshold 属性（`cudaMemPoolAttrReleaseThreshold`）来实现。

release threshold 表示在尝试将 memory 释放回 OS 之前，pool 应当保留的字节数。当 memory pool 持有的 memory 超过 release threshold 时，allocator 会在下一次 stream、event 或 device synchronize 调用时尝试将多余的 memory 释放回 OS。将 release threshold 设置为 UINT64_MAX，可以防止 driver 在每次 synchronize 之后都尝试收缩 pool。

```c
Cuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
```

如果应用程序将 `cudaMemPoolAttrReleaseThreshold` 设置得足够高，从而基本禁用 memory pool 的自动收缩机制，则可能希望显式收缩 memory pool 的 footprint。可以使用 `cudaMemPoolTrimTo` 实现这一点。在对 memory pool 执行 trim 操作时，`minBytesToKeep` 参数允许应用程序保留指定数量的 memory，例如后续执行阶段预期所需的内存量。

```c
Cuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

// 需要大量来自 stream-ordered allocator 的 memory 的应用阶段
for (i=0; i<10; i++) {
    for (j=0; j<10; j++) {
        cudaMallocAsync(&ptrs[j], size[j], stream);
    }
    kernel<<<..., stream>>>(ptrs,...);
    for (j=0; j<10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// 下一阶段不再需要这么多 memory。
// 先同步，以便 trim 操作知道这些 allocation 已经不再使用。
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(mempool, 0);

// 现在，其他进程或 allocation 机制可以使用此次 trim 操作释放的 physical memory。
```

#### 4.3.4.3. 资源使用统计

查询 pool 的 `cudaMemPoolAttrReservedMemCurrent` 属性可以得到该 pool 当前占用的 physical GPU memory 总量。查询 `cudaMemPoolAttrUsedMemCurrent` 则会返回当前从该 pool 中分配出去、且尚未可复用的 memory 总大小。

`cudaMemPoolAttr*MemHigh` 属性是水位标记，用于记录自上次重置以来对应 `cudaMemPoolAttr*MemCurrent` 属性达到的最大值。可以通过调用 `cudaMemPoolSetAttribute` API 将其重置为当前值。

```c
// 示例辅助函数：批量获取使用统计信息
struct usageStatistics {
    Cuint64_t reserved;
    Cuint64_t reservedHigh;
    Cuint64_t used;
    Cuint64_t usedHigh;
};

void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
{
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
}

// 重置水位标记后，其值将变为当前值。
void resetStatistics(cudaMemoryPool_t memPool)
{
    Cuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```

#### 4.3.4.4. Memory Reuse 策略

为了响应 allocation 请求，driver 会优先尝试复用之前通过 `cudaFreeAsync()` 释放的 memory，而不是立即向 OS 申请新的 memory。例如，在某个 stream 中释放的 memory，可以在同一 stream 的后续 allocation 请求中立即复用。当一个 stream 与 CPU 完成 synchronize 后，该 stream 中此前释放的 memory 就可以被任意 stream 中的 allocation 复用。Reuse 策略既适用于默认 memory pool，也适用于显式创建的 memory pool。

stream-ordered allocator 提供了一些可控的 allocation 策略。通过 pool 属性 `cudaMemPoolReuseFollowEventDependencies`、`cudaMemPoolReuseAllowOpportunistic` 和 `cudaMemPoolReuseAllowInternalDependencies` 可以控制这些策略，下面将分别介绍。这些策略可以通过调用 `cudaMemPoolSetAttribute` 启用或禁用。升级到更新版本的 CUDA driver 可能会改变、增强、扩展或重新排列 reuse 策略的枚举方式。

##### 4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies

在尝试分配更多 physical GPU memory 之前，allocator 会检查由 CUDA events 建立的依赖信息，并尝试从其他 stream 中已释放的 memory 进行分配。

```c
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event, originalStream);

// 在另一个 stream 中等待该 event（该 event 捕获了 free 操作）
// 使 allocator 在启用 cudaMemPoolReuseFollowEventDependencies 时
// 能够复用这块 memory 来满足另一个 stream 中的新 allocation 请求。
cudaStreamWaitEvent(otherStream, event);
cudaMallocAsync(&ptr2, size, otherStream);
```

##### 4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic

当启用 `cudaMemPoolReuseAllowOpportunistic` 策略时，allocator 会检查已释放的 allocation，判断 free 操作在 stream 顺序语义上是否已经完成，例如 stream 是否已经执行过 free 所在的位置。当该策略被禁用时，allocator 仍然会复用那些在 stream 与 CPU synchronize 后可用的 memory。禁用该策略不会阻止 `cudaMemPoolReuseFollowEventDependencies` 生效。

```c
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// 经过一段时间后，kernel 执行完成
wait(10);

// 当启用 cudaMemPoolReuseAllowOpportunistic 时，
// 可以根据 originalStream 的执行进度
// 使用之前的 allocation 来满足该 allocation 请求
cudaMallocAsync(&ptr2, size, otherStream);
```

##### 4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies

如果无法从 OS 分配并映射更多 physical memory，driver 会查找那些其可用性依赖于其他 stream 未完成进度的 memory。如果找到这样的 memory，driver 会在正在进行 allocation 的 stream 中插入所需的依赖关系，并复用该 memory。

```c
cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// 当启用 cudaMemPoolReuseAllowInternalDependencies 时，
// 且 driver 无法分配更多 physical memory，
// driver 会在进行 allocation 的 stream 中
// 实际插入一个类似 cudaStreamWaitEvent 的操作，
// 以确保 'otherStream' 中未来的 work
// 发生在 originalStream 中允许访问原 allocation 的 work 之后。
cudaMallocAsync(&ptr2, size, otherStream);
```

##### 4.3.4.4.4. 禁用 Reuse 策略

尽管这些可控的 reuse 策略可以提升 memory 复用效率，但用户也可能希望将其禁用。例如，启用 opportunistic reuse（如 `cudaMemPoolReuseAllowOpportunistic`）会因为 CPU 与 GPU 执行交错顺序不同，而导致 allocation 模式在不同运行之间产生差异。内部依赖插入（如 `cudaMemPoolReuseAllowInternalDependencies`）在某些情况下可能会以不可预期甚至非确定性的方式串行化 work，而用户可能更希望在 allocation 失败时显式 synchronize 某个 event 或 stream。

#### 4.3.4.5. Synchronization API 行为

由于 allocator 属于 CUDA driver 的一部分，因此可以与 synchronize APIs 集成，这是其中一个优化点。当用户请求 CUDA driver 执行 synchronize 时，driver 会等待所有异步 work 完成。在返回之前，driver 会判断哪些 free 操作在该 synchronize 下已经保证完成。这些 allocation 将被标记为可复用，无论指定的 stream 是什么，或 allocation 策略是否被禁用。driver 还会在此处检查 `cudaMemPoolAttrReleaseThreshold`，并释放任何可以释放的多余 physical memory。
