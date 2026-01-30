# 4.14 Memory Synchronization Domains

### 4.14.1. Memory Fence 干扰

某些 CUDA 应用可能会因为内存 fence/flush 操作等待超出 CUDA 内存一致性模型所需的事务而导致性能下降。

```
__managed__ int x = 0;
__device__ cuda::atomic<int, cuda::thread_scope_device> a(0);
__managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);

线程 1 (SM)
x = 1;
a = 1;

线程 2 (SM)
while (a != 1) ;
assert(x == 1);
b = 1;

线程 3 (CPU)
while (b != 1) ;
assert(x == 1);
```

考虑以上示例。CUDA 内存一致性模型保证断言条件成立，因此 thread 1 对 x 的写入必须在 thread 2 对 b 的写入之前，对 thread 3 可见。

对 a 的 release/acquire 仅提供 device-scope 顺序保证，仅能确保 x 对 thread 2 可见，无法保证对 thread 3 可见。因此，b 的 system-scope release/acquire 不仅需要保证 thread 2 自己的写入对 thread 3 可见，还需要确保所有对 thread 2 可见的其他线程写入也对 thread 3 可见。这一性质称为累积性（cumulativity）。由于 GPU 在执行时无法知道哪些写入是由源代码保证可见的，哪些是偶然可见的，它必须对所有在途内存操作进行广泛的保守处理。

这有时会导致干扰：GPU 可能会等待一些源代码层面上不必等待的内存操作，从而使 fence/flush 操作所需的时间超出实际需要。

需要注意，fence 可以显式出现在代码中的 intrinsic 或 atomic 操作中（如示例所示），也可以隐式出现在任务边界处实现 synchronizes-with 关系。

一个常见的例子是：一个 kernel 在本地 GPU 内存中进行计算，同时另一个并行 kernel（如 NCCL）在进行跨设备通信。完成时，本地 kernel 会隐式刷新其写入，以满足下游任务的同步关系。这可能会不必要地等待来自通信 kernel 的慢速 nvlink 或 PCIe 写操作。





### 4.14.2 Isolating Traffic with Domains

从 compute capability 9.0（Hopper 架构）GPU 和 CUDA 12.0 开始，内存同步 domain 功能提供了减轻此类干扰的方法。通过显式的代码协作，GPU 可以减少 fence 操作的范围。每次 kernel launch 都会分配一个 domain ID。写入和 fence 操作都会被标记上该 ID，只有属于同一 domain 的写入才会被该 fence 顺序控制。在“计算与通信并行”示例中，通信 kernel 可以被放置到不同的 domain 中。

使用 domain 时，必须遵守规则：不同 domain 之间的排序或同步，必须使用 system-scope fence；同一 domain 内则仅需要 device-scope fence。这是为了满足累积性要求，因为一个 kernel 的写入不会被来自另一个 domain 的 fence 包括。本质上，累积性通过确保跨 domain 流量提前被刷新到 system scope 来满足。

需要注意，这改变了 thread_scope_device 的定义。但由于 kernel 默认属于 domain 0（如下所述），因此保持了向后兼容。



### 4.14.3. Using Domains in CUDA

**域** 可以通过新的 launch attributes `cudaLaunchAttributeMemSyncDomain` 和 `cudaLaunchAttributeMemSyncDomainMap` 访问。前者在逻辑域 `cudaLaunchMemSyncDomainDefault` 和 `cudaLaunchMemSyncDomainRemote` 之间选择，后者提供从逻辑域到物理域的映射。remote 域用于执行远程内存访问的 kernel，以隔离其内存流量与本地 kernel 的流量。但选择特定域不会影响 kernel 合法访问的内存。

域的数量可以通过设备属性 `cudaDevAttrMemSyncDomainCount` 查询。compute capability 9.0（Hopper）设备有 4 个域。为保证可移植性，所有设备均可使用该功能，但 CUDA 会在 9.0 之前的设备上报告域数为 1。

逻辑域简化了应用组合。底层组件（如 NCCL）在发起 kernel 时可以选择一个语义上的逻辑域，而无需关注周围的应用架构。上层可以通过映射控制逻辑域。若未设置，默认逻辑域为 default，且默认映射为将 default 域映射到 0，将 remote 域映射到 1（在多域 GPU 上）。特定库可以在 CUDA 12.0 及更高版本中标记 remote 域，例如，NCCL 2.16 会这么做。

这为常见应用提供了无需修改其他组件、框架或应用代码的开箱即用模式。另一种用法，例如在使用 NVSHMEM 或没有明确 kernel 分类的应用中，可以通过划分流来实现：Stream A 可以将逻辑域映射到物理 domain 0，Stream B 映射到 1，以此类推。

```c
// 使用远程逻辑域发起 kernel 的示例
cudaLaunchAttribute domainAttr;
domainAttr.id = cudaLaunchAttrMemSyncDomain;
domainAttr.val = cudaLaunchMemSyncDomainRemote;
cudaLaunchConfig_t config;
// 填充其他配置字段
config.attrs = &domainAttr;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, myKernel, kernelArg1, kernelArg2...);
// 为流设置映射的示例
// （这是 compute capability 9.0（Hopper）及更高版本流的默认设置，若未显式设置，提供此图示）
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(stream, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
// 将不同流映射到不同物理域的示例，忽略逻辑域设置
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 0;
cudaStreamSetAttribute(streamA, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);

mapAttr.memSyncDomainMap.default_ = 1;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(streamB, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
```

与其他 launch attributes 一样，这些属性在 CUDA 流、通过 `cudaLaunchKernelEx` 调用的个别启动以及 CUDA 图中的 kernel 节点中均统一暴露。典型用法是在流层设置映射，在启动层设置逻辑域（或将流使用的一部分括起来），如上所述。

在流捕获过程中，两个属性都会被复制到图节点。图使用节点自身的属性，实际上是间接指定了物理域。设置在流上的 domain 相关属性在图执行时不会使用。
