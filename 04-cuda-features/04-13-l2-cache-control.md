# 4.13 L2 Cache Control

当一个 CUDA kernel 在 global memory 中反复访问某一数据区域时，这类访问可以视为 persisting 访问。相反，如果数据只被访问一次，则可以视为 streaming 访问。

compute capability 8.0 及以上的 device 支持影响数据在 L2 cache 中的持久性，从而可能为 global memory 提供更高带宽和更低延迟的访问。

该功能通过两个主要 API 提供：

▶ CUDA runtime API（自 CUDA 11.0 起）提供对 L2 cache persistence 的程序化控制。
▶ libcu++ 库中的 cuda::annotated_ptr API（自 CUDA 11.5 起）允许在 CUDA kernel 中为指针标注 memory access 属性，以实现类似效果。

以下章节重点介绍 CUDA runtime API。有关 cuda::annotated_ptr 方法的详细信息，请参考 libcu++ 文档。

### 4.13.1. L2 Cache Set-Aside for Persisting Accesses

L2 cache 的一部分可以被预留，用于 persisting 的 global memory 访问。persisting 访问会优先使用该预留的 L2 cache 区域，而 normal 或 streaming 的 global memory 访问仅在该区域未被 persisting 访问使用时才能利用它。

L2 cache 的 set-aside 大小可以在一定范围内调整：

```c
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* 为 persisting 访问预留 L2 cache 的 3/4 或允许的最大值 */
```

当 GPU 配置为 Multi-Instance GPU (MIG) 模式时，L2 cache set-aside 功能将被禁用。

在使用 Multi-Process Service (MPS) 时，无法通过 cudaDeviceSetLimit 修改 L2 cache 的 set-aside 大小。相反，该大小只能在启动 MPS server 时通过环境变量 CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT 指定。

### 4.13.2. L2 Policy for Persisting Accesses

access policy window 用于指定 global memory 中一段连续区域，并为该区域内的访问指定在 L2 cache 中的 persistence 属性。

下面的代码示例演示了如何使用 CUDA Stream 设置一个 L2 persisting access window。

### CUDA Stream 示例

```c
cudaStreamAttrValue stream_attribute;
// Stream 级别属性数据结构

stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr);
// Global Memory 数据指针

stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
// persisting 访问的字节数
// （必须小于 cudaDeviceProp::accessPolicyMaxWindowSize）

stream_attribute.accessPolicyWindow.hitRatio = 0.6;
// cache 命中比例提示

stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
// cache 命中时的访问属性类型

stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
// cache 未命中时的访问属性类型

// 将属性设置到类型为 cudaStream_t 的 CUDA Stream
cudaStreamSetAttribute(stream,
                       cudaStreamAttributeAccessPolicyWindow,
                       &stream_attribute);
```

当 kernel 随后在该 CUDA Stream 中执行时，在 global memory 区间 `[ptr..ptr+num_bytes)` 内的访问比访问其他 global memory 位置更可能在 L2 cache 中保持持久。

L2 persistence 也可以为 CUDA Graph Kernel Node 设置，如下所示：

### CUDA GraphKernelNode 示例

```c
cudaKernelNodeAttrValue node_attribute;
// Kernel 级别属性数据结构

node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr);
// Global Memory 数据指针

node_attribute.accessPolicyWindow.num_bytes = num_bytes;
// persisting 访问的字节数
// （必须小于 cudaDeviceProp::accessPolicyMaxWindowSize）

node_attribute.accessPolicyWindow.hitRatio = 0.6;
// cache 命中比例提示

node_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
// cache 命中时的访问属性类型

node_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
// cache 未命中时的访问属性类型

// 将属性设置到类型为 cudaGraphNode_t 的 CUDA Graph Kernel node
cudaGraphKernelNodeSetAttribute(node,
                                cudaKernelNodeAttributeAccessPolicyWindow,
                                &node_attribute);
```

hitRatio 参数用于指定获得 hitProp 属性的访问比例。在上述两个示例中，global memory 区间 `[ptr..ptr+num_bytes)` 中 60% 的访问具有 persisting 属性，40% 的访问具有 streaming 属性。

哪些具体的 memory 访问被归类为 persisting（即 hitProp）是随机的，其概率约为 hitRatio；具体分布取决于硬件架构和 memory 区域大小。

例如，如果 L2 set-aside cache 大小为 16KB，而 accessPolicyWindow 中的 num_bytes 为 32KB：

▶ 当 hitRatio 为 0.5 时，硬件会随机选择 32KB 窗口中的 16KB 区域，将其标记为 persisting 并缓存到 set-aside L2 cache 区域中。
▶ 当 hitRatio 为 1.0 时，硬件会尝试将整个 32KB 窗口缓存到 set-aside L2 cache 区域中。由于 set-aside 区域小于窗口大小，cache line 会被逐出，以保持最近使用的 16KB 数据留在 set-aside 区域。

因此，hitRatio 可以用于避免 cache line 抖动，并整体减少 L2 cache 中数据的进出量。

当 hitRatio 小于 1.0 时，还可以手动控制来自不同 CUDA Stream 的 accessPolicyWindow 在 L2 中缓存的数据量。例如，若 L2 set-aside cache 大小为 16KB，两个不同 CUDA Stream 中并发执行的 kernel 各自具有 16KB 的 accessPolicyWindow，且 hitRatio 均为 1.0，则它们在竞争共享的 L2 资源时可能会互相逐出对方的 cache line。如果两个 accessPolicyWindow 的 hitRatio 均为 0.5，则它们更不容易逐出自身或彼此的 persisting cache line。

### 4.13.3. L2 Access Properties

针对不同的 global memory 访问，定义了三种 access property 类型：

1. **cudaAccessPropertyStreaming**：具有 streaming 属性的 memory 访问在 L2 cache 中不太可能保持持久，因为这些访问会被优先逐出。
2. **cudaAccessPropertyPersisting**：具有 persisting 属性的 memory 访问更可能在 L2 cache 中保持持久，因为这些访问会优先保留在 L2 cache 的 set-aside 区域中。
3. **cudaAccessPropertyNormal**：该属性会强制将之前应用的 persisting 属性重置为 normal 状态。来自之前 CUDA kernel 的 persisting 访问可能在 L2 cache 中保留很长时间，即使它们已经不再使用。这种“使用后持久”会减少后续未使用 persisting 属性的 kernel 可用的 L2 cache 容量。将 access policy window 设置为 cudaAccessPropertyNormal 会移除之前访问的 persisting（优先保留）状态，就好像之前的访问从未设置过 access property 一样。

### 4.13.4. L2 Persistence 示例

```c
cudaStream_t stream;
cudaStreamCreate(&stream);
// 创建 CUDA Stream

cudaDeviceProp prop;
// CUDA 设备属性变量

cudaGetDeviceProperties(&prop, device_id);
// 查询 GPU 属性

size_t size = min(int(prop.l2CacheSize * 0.75),
                  prop.persistingL2CacheMaxSize);

cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
// 为 persisting 访问预留 L2 cache 的 3/4 或允许的最大值

size_t window_size =
    min(prop.accessPolicyMaxWindowSize, num_bytes);
// 选择用户定义 num_bytes 与最大 window 大小中的较小值

cudaStreamAttrValue stream_attribute;
// Stream 级别属性数据结构

stream_attribute.accessPolicyWindow.base_ptr =
    reinterpret_cast<void*>(data1);
// Global Memory 数据指针

stream_attribute.accessPolicyWindow.num_bytes = window_size;
// persisting 访问的字节数

stream_attribute.accessPolicyWindow.hitRatio = 0.6;
// cache 命中比例提示

stream_attribute.accessPolicyWindow.hitProp =
    cudaAccessPropertyPersisting;
// persistence 属性

stream_attribute.accessPolicyWindow.missProp =
    cudaAccessPropertyStreaming;
// cache 未命中时的访问属性

cudaStreamSetAttribute(stream,
                       cudaStreamAttributeAccessPolicyWindow,
                       &stream_attribute);
// 将属性设置到 CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);
    // data1 会被 kernel 多次使用
}

// [data1 + num_bytes) 区域可从 L2 persistence 中受益

cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);
// 同一 stream 中的不同 kernel 也可受益于 data1 的 persistence

stream_attribute.accessPolicyWindow.num_bytes = 0;
// 将 window 大小设置为 0 以禁用该策略

cudaStreamSetAttribute(stream,
                       cudaStreamAttributeAccessPolicyWindow,
                       &stream_attribute);
// 覆盖 CUDA Stream 上的 access policy 属性

cudaCtxResetPersistingL2Cache();
// 移除 L2 中的所有 persistent cache line

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);
// data2 现在可以在 normal 模式下使用完整的 L2
```

### 4.13.5. Reset L2 Access to Normal

来自之前 CUDA kernel 的 persisting L2 cache line 可能在使用后很长时间仍然保留在 L2 中。因此，为了让 streaming 或 normal 访问能够以正常优先级使用 L2 cache，将其重置为 normal 状态非常重要。

persisting 访问可以通过以下三种方式重置为 normal 状态：

1. 使用 access property cudaAccessPropertyNormal 重置之前的 persisting memory 区域。
2. 调用 cudaCtxResetPersistingL2Cache() 将所有 persisting L2 cache line 重置为 normal。
3. 未被访问的 cache line 最终会自动恢复为 normal。但强烈不建议依赖自动重置，因为其发生时间不可确定。

### 4.13.6. 管理 L2 set-aside cache 的使用

在不同 CUDA Stream 中并发执行的多个 CUDA kernel 可能分别设置了不同的 access policy window。然而，L2 的 set-aside 区域在这些并发 kernel 之间是共享的。因此，该区域的总使用量等于所有并发 kernel 各自使用量之和。

当 persisting 访问总量超过 set-aside L2 cache 容量时，将访问标记为 persisting 所带来的收益会逐渐减弱。

为了有效管理 set-aside L2 cache 的使用，应用程序需要考虑以下因素：

▶ L2 set-aside cache 的大小。
▶ 可能并发执行的 CUDA kernel。
▶ 所有可能并发执行的 CUDA kernel 的 access policy window。
▶ 何时以及如何执行 L2 reset，以便 normal 或 streaming 访问能够以相同优先级使用之前预留的 L2 cache。

### 4.13.7. 查询 L2 cache 属性

与 L2 cache 相关的属性属于 cudaDeviceProp 结构体的一部分，可以通过 CUDA runtime API cudaGetDeviceProperties 查询。

CUDA Device Properties 包括：

▶ l2CacheSize：GPU 上可用的 L2 cache 大小。
▶ persistingL2CacheMaxSize：可为 persisting memory 访问预留的 L2 cache 最大大小。
▶ accessPolicyMaxWindowSize：access policy window 的最大大小。

### 4.13.8. 控制用于 Persisting Memory Access 的 L2 Cache Set-Aside 大小

用于 persisting memory 访问的 L2 set-aside cache 大小可以通过 CUDA runtime API cudaDeviceGetLimit 查询，并通过 CUDA runtime API cudaDeviceSetLimit 以 cudaLimit 形式进行设置。该限制的最大值为 cudaDeviceProp::persistingL2CacheMaxSize。

```c
enum cudaLimit {
/* 其他字段未显示 */
cudaLimitPersistingL2CacheSize
};
```
