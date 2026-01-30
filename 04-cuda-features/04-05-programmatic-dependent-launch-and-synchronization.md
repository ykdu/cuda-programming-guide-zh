# 4.5 Programmatic Dependent Launch and Synchronization

> 原文：CUDA Programming Guide v13.1  
> https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

TODO: 翻译内容

Programmatic Dependent Launch 机制允许一个依赖的次级 kernel，在其所依赖的主 kernel 尚未在同一 CUDA stream 上执行完成之前就被启动。该机制从 compute capability 9.0 的设备开始支持。当次级 kernel 可以完成大量不依赖主 kernel 结果的工作时，这一技术可以带来性能收益。

### 4.5.1. 背景

CUDA 应用通过在 GPU 上启动并执行多个 kernel 来利用 GPU 资源。一个典型的 GPU 活动时间线如图 39 所示。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-activity.png)*Figure 39: GPU activity timeline*

在该示例中，secondary_kernel 在 primary_kernel 执行完成之后才启动。通常需要串行执行，因为 secondary_kernel 依赖 primary_kernel 产生的结果数据。如果 secondary_kernel 不依赖 primary_kernel，那么两者可以通过 CUDA Streams 并发启动。即使 secondary_kernel 依赖 primary_kernel，也仍然存在一定的并发执行潜力。例如，几乎所有 kernel 都包含某种形式的 preamble 阶段，在该阶段会执行诸如清零缓冲区或加载常量值等任务。

![dfsf](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/secondary-kernel-preamble.png)*Figure 40 Preamble section of `secondary_kernel`*

图 40 展示了 secondary_kernel 中可以并发执行、且不会影响应用结果的部分。需要注意的是，并发启动还可以将 secondary_kernel 的启动延迟隐藏在 primary_kernel 的执行期间。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/preamble-overlap.png)*Figure 41 Concurrent execution of `primary_kernel` and `secondary_kernel`*

图 41 所示的 secondary_kernel 的并发启动与执行，可以通过 Programmatic Dependent Launch 实现。

Programmatic Dependent Launch 对 CUDA kernel 启动 API 进行了修改，具体说明见下一节。这些 API 至少需要 compute capability 9.0 才能支持重叠执行。

### 4.5.2. API 描述

在 Programmatic Dependent Launch 中，primary kernel 和 secondary kernel 会在同一个 CUDA stream 中启动。primary kernel 应当在其所有 thread block 准备好允许 secondary kernel 启动时，执行 `cudaTriggerProgrammaticLaunchCompletion`。secondary kernel 必须使用可扩展启动 API（extensible launch API）进行启动，如下所示。

```
__global__ void primary_kernel() {
    // Initial work that should finish before starting secondary kernel

    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
    // Independent work

    // Will block until all primary kernels the secondary kernel is dependent on have completed
    cudaGridDependencySynchronize();

    // Dependent work
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);
```

当 secondary kernel 使用 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性启动时，CUDA driver 可以安全地提前启动 secondary kernel，而不需要等待 primary kernel 完成或其 memory flush 完成之后再启动 secondary kernel。

当 primary kernel 的所有 thread block 都已经启动并执行了 `cudaTriggerProgrammaticLaunchCompletion` 时，CUDA driver 就可以启动 secondary kernel。如果 primary kernel 没有显式执行该 trigger，则该触发会在 primary kernel 的所有 thread block 退出之后隐式发生。

在这两种情况下，secondary kernel 的 thread block 都可能在 primary kernel 写入的数据变为可见之前就已经启动。因此，当 secondary kernel 以 Programmatic Dependent Launch 方式配置时，**必须始终使用** `cudaGridDependencySynchronize` 或其他手段来确认 primary kernel 的结果数据已经可用。

请注意，这些方法为 primary kernel 与 secondary kernel 的并发执行提供了可能性，但这种行为是机会性的，并不保证一定会发生并发 kernel 执行。依赖这种方式实现的并发执行是不安全的，并且可能导致死锁。

### 4.5.3. 在 CUDA Graphs 中的使用

Programmatic Dependent Launch 可以通过 stream capture 或直接通过 edge data 的方式用于 CUDA Graphs。要在 CUDA Graph 中使用 edge data 来编程该特性，需要在连接两个 kernel node 的 edge 上使用 `cudaGraphDependencyTypeProgrammatic` 类型的 `cudaGraphDependencyType` 值。该 edge 类型会使上游 kernel 对下游 kernel 中的 `cudaGridDependencySynchronize()` 可见。

该 edge 类型必须与 `cudaGraphKernelNodePortLaunchCompletion` 或 `cudaGraphKernelNodePortProgrammatic` 中的任意一个输出端口一起使用。

通过 stream capture 生成的等价 graph 形式如下所示：

| Stream code (abbreviated)                                    | Resulting graph edge                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization; attribute.val.programmaticStreamSerializationAllowed = 1; | cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; |
| cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 0; | cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; |
| cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 1; | cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion; |

