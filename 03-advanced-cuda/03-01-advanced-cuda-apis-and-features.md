# 3.1 Advanced CUDA APIs and Features

本节将介绍更高级的 CUDA APIs 和特性。这些主题涵盖的技术或特性通常不需要修改 CUDA kernel，但仍然可以从 host 侧影响应用层行为，包括 GPU work 的执行方式和性能，以及 CPU 侧的性能表现。

### 3.1.1. cudaLaunchKernelEx

在最早版本中引入 triple chevron notation 时，kernel 的 Kernel Configuration 只有四个可编程参数：thread block dimensions、grid dimensions、dynamic shared memory（可选，未指定时为 0）以及 stream（未指定时使用 default stream）。

某些 CUDA 特性可以从 kernel launch 时提供的附加属性和提示中获益。`cudaLaunchKernelEx` 使程序能够通过 `cudaLaunchConfig_t` 结构体来设置上述执行配置参数。此外，`cudaLaunchConfig_t` 结构体还允许程序传入零个或多个 `cudaLaunchAttributes`，以控制或建议 kernel launch 的其他参数。例如，本章后续讨论的 `cudaLaunchAttributePreferredSharedMemoryCarveout`（参见 Configuring L1/Shared Memory Balance）就是通过 `cudaLaunchKernelEx` 来指定的。本章后续讨论的 `cudaLaunchAttributeClusterDimension` 属性，用于为 kernel launch 指定期望的 cluster size。

所有受支持属性及其含义的完整列表，收录在 CUDA Runtime API Reference Documentation 中。

### 3.1.2. Launching Clusters

Thread block clusters 是在前面小节中介绍的一种可选级别的 thread block 组织方式，在 compute capability 9.0 及更高版本中可用，它使得应用可以保证 cluster 中的 thread blocks 同时在同一个 GPC 上执行。这使得比单个 SM 能容纳的更大规模 thread 组可以相互交换数据并相互同步。
 第 2.1.10.1 节展示了如何使用 triple chevron notation 指定并启动一个使用 clusters 的 kernel。在该节中，使用了 `__cluster_dims__` 注解来指定必须用于启动 kernel 的 cluster 尺寸。当使用 triple chevron notation 时，cluster 的大小是隐式决定的。

#### 3.1.2.1. 使用 cudaLaunchKernelEx 启动带 Clusters 的 kernel

与使用 triple chevron notation 启动带 clusters 的 kernels 不同，thread block cluster 的大小可以按每次 launch 配置。下面的代码示例展示了如何使用 `cudaLaunchKernelEx` 启动一个 cluster kernel。

```
    // kernel 定义
    // kernel 上没有编译时附加属性
    __global__ void cluster_kernel(float *input, float* output)
    {

    }
    int main()
    {
        float *input, *output;
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
        // 带运行时 cluster 大小的 kernel 调用
        {
            cudaLaunchConfig_t config = {0};
            // grid 维度不受 cluster launch 的影响，仍然按 block 数量枚举
            // grid 维度应当是 cluster 大小的倍数
            config.gridDim = numBlocks;
            config.blockDim = threadsPerBlock;
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = 2; // X 维度的 cluster 大小
            attribute[0].val.clusterDim.y = 1;
            attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute;
            config.numAttrs = 1;

            cudaLaunchKernelEx(&config, cluster_kernel, input, output);
        }
    }
```

有两个 `cudaLaunchAttribute` 类型与 thread block clusters 相关：`cudaLaunchAttributeClusterDimension` 和 `cudaLaunchAttributePreferredClusterDimension`。
 属性 id `cudaLaunchAttributeClusterDimension` 指定执行 cluster 所需的维度。该属性的值 `clusterDim` 是一个三维值。grid 的对应维度 (x, y, z) 必须能被指定的 cluster dimension 的各自维度整除。
 将其设置类似于在 kernel 定义时使用 `__cluster_dims__` 属性（如 “使用 Triple Chevron Notation 启动带 Clusters” 一节所示），但可以在运行时针对同一个 kernel 的不同 launch 进行更改。
 在 compute capability 10.0 及更高版本的 GPU 上，另一个属性 id `cudaLaunchAttributePreferredClusterDimension` 允许应用额外指定 cluster 的首选维度。首选维度必须是 kernel 上的 `__cluster_dims__` 属性或传递给 `cudaLaunchKernelEx` 的 `cudaLaunchAttributeClusterDimension` 属性所指定的最小 cluster 维度的整数倍。
 也就是说，除了首选 cluster dimension 之外，必须指定一个最小 cluster dimension。grid 的对应维度 (x, y, z) 必须能被指定的首选 cluster dimension 的各自维度整除。
 所有 thread blocks 将至少在最小 cluster dimension 下执行。尽可能地，将使用首选维度的 clusters，但并不是所有 cluster 都能保证以首选维度执行。所有 thread blocks 将在最小或首选 cluster dimension 下执行。使用首选 cluster dimension 的 kernels 必须编写为在最小或首选 cluster dimension 下都能正确运行。

#### 3.1.2.2. Blocks as Clusters

当 kernel 使用 `__cluster_dims__` 注解定义时，grid 中 clusters 的数量是隐式的，并且可以通过将 grid 尺寸除以指定的 cluster 尺寸来计算。

```c
    __cluster_dims__((2, 2, 2)) __global__ void foo();

    // 每个 cluster 有 2x2x2 个 thread blocks 的 8x8x8 clusters。
    foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
```

在上面的例子中，kernel 被作为一个 16x16x16 个 thread block 的 grid 启动，这意味着使用了 8x8x8 个 clusters。

kernel 也可以选择使用 `__block_size__` 注解，该注解在定义 kernel 时指定所需的 block 大小和 cluster 大小。当使用该注解时，triple chevron 启动变为以 cluster 为维度的 grid，而不是 thread blocks，如下面所示。

```
    // 关于每个 block 的 threads 数和每个 cluster 的 blocks 数
    // 如何作为 kernel 的一个属性处理的实现细节。
    __block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

    // 8x8x8 clusters。
    foo<<<dim3(8, 8, 8)>>>();
```

`__block_size__` 需要两个各包含 3 个元素的元组。第一个元组表示 block 维度，第二个元组表示 cluster size。如果不传入第二个元组，则假定其值是 `(1,1,1)`。要指定 stream，必须在 `<<<>>>` 内将第二和第三个参数传为 `1` 和 `0`，最后是 stream。传入其他值将导致未定义行为。

请注意，`__block_size__` 的第二元组和 `__cluster_dims__` 不能同时指定。这也是非法的。同样，不能在没有指定 `__cluster_dims__` 的情况下使用 `__block_size__`。当指定了 `__block_size__` 的第二元组时，这意味着启用了 “Blocks as Clusters”，编译器会将 `<<<>>>` 中的第一个参数识别为 clusters 的数量，而不是 thread blocks。

### 3.1.3 更多关于 Streams 和 Events

**CUDA Streams** 介绍了 CUDA streams 的基础知识。默认情况下，提交到某个 CUDA stream 上的操作是串行的：必须等待前一个操作完成之后才能开始执行。唯一的例外是新引入的 **Programmatic Dependent Launch and Synchronization** 特性。使用多个 CUDA streams 是实现并发执行的一种方式；另一种方式是使用 **CUDA Graphs**。这两种方法也可以结合使用。

在特定情况下，如果没有 event 依赖、没有隐式同步、且有足够资源，那么提交到不同 CUDA streams 上的工作可能会并发执行。

如果在不同 CUDA streams 的独立操作之间插入了对 NULL stream 的 CUDA 操作，那么这些独立操作不能并发运行，除非这些 streams 是非阻塞 CUDA streams。非阻塞 CUDA streams 是通过运行时 API `cudaStreamCreateWithFlags()` 并使用 `cudaStreamNonBlocking` 标志创建的。为了提高潜在的 GPU 并发执行能力，建议用户创建非阻塞 CUDA streams。

还建议用户选择足够满足他们问题需求的最不通用的同步选项。例如，如果需要 CPU 等待（阻塞）某个特定 CUDA stream 上的所有工作完成，那么对该 stream 使用 `cudaStreamSynchronize()` 会比使用 `cudaDeviceSynchronize()` 更好，因为后者会不必要地等待该设备上所有 CUDA streams 的 GPU 工作完成。而如果需要 CPU 等待但不阻塞，那么使用 `cudaStreamQuery()` 并在轮询循环中检查其返回值可能更好。

可以通过 CUDA events 实现类似的同步效果，例如，在某个 stream 上记录一个 event，然后调用 `cudaEventSynchronize()` 以阻塞方式等待该 event 捕获的工作完成。这样做同样比使用 `cudaDeviceSynchronize()` 更好、更有针对性。调用 `cudaEventQuery()` 并检查其返回值，例如在轮询循环中，会是一个非阻塞的替代方案。

如果这类操作发生在应用的关键路径中，那么选择显式同步的方法尤为重要。表 4 提供了 host 上各种同步选项的high-level summary。

表 4 主机显式同步选项摘要

|                              | 等待特定 stream         | 等待特定 event         | 等待设备上的所有工作    |
| ---------------------------- | ----------------------- | ---------------------- | ----------------------- |
| Non-blocking（需要轮询循环） | cudaStreamQuery()       | cudaEventQuery()       | N/A                     |
| blocking                     | cudaStreamSynchronize() | cudaEventSynchronize() | cudaDeviceSynchronize() |

为了实现 streams 之间的同步（即表达依赖关系），建议使用无定时信息的 CUDA events，如 **CUDA Events** 中所述。用户可以调用 `cudaStreamWaitEvent()` 来强制某个 stream 上之后提交的操作等待先前记录的 event 完成（例如在另一个 stream 上）。注意，对于任何等待或查询 event 的 CUDA API，用户有责任确保已调用 `cudaEventRecord` API，否则未记录的 event 将始终返回成功。

默认情况下，CUDA events 带有定时信息，因为它们可以用于 `cudaEventElapsedTime()` API 调用。然而，仅用于表达跨 streams 依赖关系的 CUDA event 并不需要定时信息。在这种情况下，建议通过使用带有 `cudaEventDisableTiming` 标志的 `cudaEventCreateWithFlags()` API 创建禁用定时信息的 events，以提高性能。

#### 3.1.3.1 Stream 优先级

可以在创建时使用 `cudaStreamCreateWithPriority()` 指定 streams 的相对优先级。可允许的优先级范围（从最高优先级到最低优先级）可以通过 `cudaDeviceGetStreamPriorityRange()` 函数获得。在运行时，GPU scheduler 利用 stream 优先级来确定任务执行顺序，但这些优先级只是提示而不是保证。在选择要启动的工作时，处于高优先级 stream 中的待处理任务优先于处于低优先级 stream 的任务。高优先级任务不会抢占已经在运行的低优先级任务。GPU 在任务执行期间不会重新评估工作队列，提高 stream 的优先级不会中断正在进行的工作。Stream 优先级会影响任务执行顺序，但不会强制严格的顺序，因此用户可以利用 stream 优先级来影响任务执行，而不依赖严格的顺序保证。

下面的示例代码获取当前设备可用的优先级范围，并创建两个具有最高和最低可用优先级的非阻塞 CUDA streams。

```c
// 获取该设备的 stream 优先级范围
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// 创建具有最高和最低可用优先级的 streams
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);
```

#### 3.1.3.2 显式同步

如前所述，有多种方式使 streams 与其他 streams 进行同步。以下是在不同粒度级别上常用的方法：

- `cudaDeviceSynchronize()` 等待所有 host 线程的所有 streams 上之前的命令完成。
- `cudaStreamSynchronize()` 以一个 stream 作为参数，等待该 stream 上所有之前的命令完成。它可用于使主机与特定 stream 同步，同时允许其他 streams 在设备上继续执行。
- `cudaStreamWaitEvent()` 以一个 stream 和一个 event 作为参数（参见 **CUDA Events** 了解 events 描述），并使该调用之后添加到指定 stream 的所有命令延迟执行，直到该 event 完成。
- `cudaStreamQuery()` 为应用提供一种方式，以确定 stream 上所有之前的命令是否完成。

#### 3.1.3.3 隐式同步

如果主机线程在两个不同 streams 的命令之间发出以下任何操作，则这两个命令不能并发运行：

- 分配 page-locked host memory
- 分配 device memory
- 执行 device memory set
- 对同一 device memory 的两个地址进行内存拷贝
- 对 NULL stream 的任何 CUDA 命令
- 切换 L1/shared memory 配置

需要进行依赖性检查的操作包括该 stream 上的其他命令以及对该 stream 上 `cudaStreamQuery()` 的任何调用。因此，应用程序应遵循以下指导，以提高并发 kernel 执行的潜力：

- 所有独立的操作应在依赖操作之前发出，
- 所有类型的同步应尽可能推迟。

### 3.1.4 程序化依赖 kernel 启动（Programmatic Dependent Kernel Launch）

如前所述，CUDA Streams 的语义保证了 kernels 会按顺序执行。这意味着如果有两个依次启动的 kernels，且第二个 kernel 依赖于第一个 kernel 的结果，那么程序员可以放心：在第二个 kernel 开始执行时，它所依赖的数据已经可用。

然而，也可能出现这样的情况：第一个 kernel 已经将后续 kernel 依赖的数据写入到 global memory 中，但它自身仍有更多工作要做。同样，依赖的第二个 kernel 可能在需要来自第一个 kernel 的数据之前，有一些独立的工作要执行。在这种情况下，如果硬件资源允许，可以部分重叠两个 kernels 的执行。这样的重叠甚至还可以覆盖第二个 kernel 的启动开销。除了硬件资源可用性之外，可实现的重叠程度还取决于内核自身的具体结构，例如：

- 第一个 kernel 在执行中什么时候完成了第二个 kernel 所依赖的工作？
- 第二个 kernel 在执行中什么时候开始处理来自第一个 kernel 的数据？

由于这些行为高度依赖于各个 kernel 的具体情况，因此难以完全自动化。因此 CUDA 提供了一种机制，使应用**开发者能够指定两个 kernel 之间的同步点**。这种机制称为 Programmatic Dependent Kernel Launch。如下图所示。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/pdl.png)

PDL 有三个主要组成部分：

1. 第一个 kernel（即所谓的 primary kernel）需要调用一个特殊函数，以表明它已经完成了后续依赖 kernels 所需要的数据和状态。这通过调用 `cudaTriggerProgrammaticLaunchCompletion()` 来完成。
2. 相反地，依赖的第二个 kernel 需要表明它已经完成了与 primary kernel 无关的部分，并且现在正在等待 primary kernel 完成它所依赖的工作。这通过调用 `cudaGridDependencySynchronize()` 来完成。
3. 第二个 kernel 需要使用特殊属性 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动，并将其 `programmaticStreamSerializationAllowed` 字段设置为 `1`。

下面的代码片段展示了如何实现这一点：

Listing 3：带有两个kernel的示例

```c
__global__ void primary_kernel() {
    // 初始工作，在启动 secondary kernel 之前应完成

    // 触发 secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // 可以与 secondary kernel 重叠执行的工作
}
__global__ void secondary_kernel()
{
    // 初始化、独立工作等

    // 将阻塞直到所有它依赖的 primary kernels
    // 都完成并将结果刷新到 global memory
    cudaGridDependencySynchronize();

    // 依赖的工作
}
// 使用特殊属性启动 secondary kernel
// 配置属性
cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;

// 在 kernel 启动配置中设置属性
cudaLaunchConfig_t config = {0};

// 基础启动配置
config.gridDim = grid_dim;
config.blockDim = block_dim;
config.dynamicSmemBytes = 0;
config.stream = stream;

// 添加 PDL 特殊属性
config.attrs = attribute;
config.numAttrs = 1;

// 启动 primary kernel
primary_kernel<<<grid_dim, block_dim, 0, stream>>>();

// 使用带属性的配置启动 secondary (dependent) kernel
cudaLaunchKernelEx(&config, secondary_kernel);
```

### 3.1.5. Batched Memory Transfers

在 CUDA 开发中，一个常见模式是使用 batching 技术。所谓 batching，大致指将多个（通常较小的）任务组合成一个单一（通常较大）的操作。batch 的各个组成部分不必完全相同，但通常如此。这个概念的一个示例是 cuBLAS 提供的 batch matrix multiplication 操作。

一般来说，和 CUDA Graph 以及 PDL 一样，进行 batching 的目的，是为了减少将每个批次任务分别 dispatch 所带来的开销。就内存传输而言，发起一次内存拷贝本身就可能产生一定的 CPU 和 driver 开销。此外，当前形式的 `cudaMemcpyAsync()` 并不一定能够向 driver 提供足够的信息，以便其对传输过程进行优化，例如在源地址和目的地址特性方面的提示。在 Tegra 平台上，内存传输可以选择使用 SM 或 Copy Engine（CE）来执行。当前使用哪一种，是由 driver 中的启发式策略来决定的。这一点很重要，因为使用 SM 可能会带来更快的传输速度，但同时会占用一部分可用的计算资源；而使用 CE 可能会使传输速度变慢，但由于 SM 可以被释放出来执行其他工作，反而可能带来更高的整体应用性能。

这些考虑促使设计了 `cudaMemcpyBatchAsync()` 函数（及其对应的 `cudaMemcpyBatch3DAsync()`）。这些函数允许对 batched memory transfers 进行优化。除了源和目标指针列表之外，该 API 使用 memory copy attributes 来指定排序预期，以及有关源和目标位置的提示，还可以提示是否希望将传输与 compute 重叠（这一点当前只在带有 CEs 的 Tegra 平台上支持）。

下面先考虑一个简单的从 pinned host memory 到 pinned device memory 的 batch 数据传输示例：

Listing 4 Example of Homogeneous Batched Memory Transfer from Pinned Host Memory to Pinned Device Memory

```c
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<void *> sizes(batch_size);
// 为源和目标缓冲区分配内存
// 使用 stream 编号初始化
for (size_t i = 0; i < batch_size; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
    cudaMemsetAsync(srcs[i], sizes[i], stream);
}

// 为这一批拷贝设置 attributes
cudaMemcpyAttributes attrs = {};
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
// batch 中所有的拷贝具有相同的 copy attributes
size_t attrsIdxs = 0;  // attributes 的索引

// 发起 batched memory transfer
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);
```

`cudaMemcpyBatchAsync()` 函数的前几个参数看起来非常直观。它们由一组数组组成，这些数组分别包含源指针、目的指针以及传输大小。每个数组都必须包含 `batch_size` 个元素。新增的信息来自于 attributes。该函数需要一个指向 attributes 数组的指针，以及一个与之对应的 attribute 索引数组。原则上，也可以传入一个 `size_t` 数组，用来记录发生失败的传输的索引；不过在这里传入 `nullptr` 是安全的，在这种情况下，失败的索引将不会被记录。

接下来来看 attributes。在这个示例中，所有的传输都是同构的，因此只使用了一个 attribute，并将其应用到所有传输上。这是通过 `attrIndex` 参数来控制的。原则上，`attrIndex` 可以是一个数组，其中第 *i* 个元素包含该 attribute 适用的第一条传输的索引。在本例中，`attrIndex` 被视为一个只包含单个元素的数组，其值为 `0`，这意味着 `attribute[0]` 将适用于索引为 0 及之后的所有传输，也就是说，适用于全部传输。

最后需要注意的是，这里将 `srcAccessOrder` 属性设置为了 `cudaMemcpySrcAccessOrderStream`。这表示源数据将按照常规的 stream 顺序进行访问。换句话说，在涉及这些源指针或目的指针的数据的先前 kernel 执行完成之前，该 memcpy 操作将会被阻塞。

在下一个示例中，我们将考虑一个更复杂的 heterogeneous batch transfer。

Listing 5 Example of Heterogeneous Batched Memory Transfer using some Ephemeral Host Memory to Pinned Device Memory

```c
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<void *> sizes(batch_size);

// 为 src 和 dst 缓冲区分配内存
for (size_t i = 0; i < batch_size - 10; i++) {
    cudaMallocHost(&srcs[i], sizes[i]);
    cudaMalloc(&dsts[i], sizes[i]);
}

int buffer[10];
for (size_t i = batch_size - 10; i < batch_size; i++) {
    srcs[i] = &buffer[10 - (batch_size - i];
    cudaMalloc(&dsts[i], sizes[i]);
}

// 为这一批拷贝设置 attributes
cudaMemcpyAttributes attrs[2] = {};
attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
attrs[1].srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;

size_t attrsIdxs[2];
attrsIdxs[0] = 0;
attrsIdxs[1] = batch_size - 10;
// 发起 batched memory transfer
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 2 /*numAttrs*/, nullptr /*failIdx*/, stream);
```

这里我们有两种传输：`batch_size-10` 次从 pinned host memory 到 pinned device memory 的传输，以及 10 次从一个 host 数组到 pinned device memory 的传输。此外，buffer 数组不仅在 host 上，而且仅在当前作用域中存在 — 它的地址就是所谓的 ephemeral pointer。此指针在 API 调用完成后可能不再有效（因为是 asynchronous）。为了使用这样的 ephemeral pointers 执行拷贝，attribute 中的 srcAccessOrder 必须设置为 `cudaMemcpySrcAccessOrderDuringApiCall`。

现在我们有两个 attributes，第一个适用于所有从索引 0 开始并小于 `batch_size-10` 的传输。第二个适用于所有从 `batch_size-10` 开始并小于 `batch_size` 的传输。

如果不是从栈上分配 buffer 数组，而是使用 malloc 从堆上分配，那么数据将不再是 ephemeral。它将在指针显式释放之前一直有效。在这种情况下，如何阶段性安排拷贝的最佳方案将取决于系统是否具有 hardware managed memory 或通过地址转换实现对 host memory 的 GPU coherent access。如果是这样，那么使用 stream ordering 是最优的；如果不是，则立即执行传输更加合理。在这种情况下，attribute 的 `srcAccessOrder` 应设置为 `cudaMemcpyAccessOrderAny`。

`cudaMemcpyBatchAsync` 函数还允许程序员提供有关源和目标位置的提示。这可以通过设置 `cudaMemcpyAttributes` 结构的 `srcLocation` 和 `dstLocation` 字段来完成。`srcLocation` 和 `dstLocation` 字段的类型都是 `cudaMemLocation`，该结构包含位置的类型以及位置的 ID。这个 `cudaMemLocation` 结构也可以用来在使用 `cudaMemPrefetchAsync()` 时为 runtime 提供预取提示。下面的代码示例说明了如何设置从 device 到 host 的特定 NUMA node 的位置提示：

Listing 6 Example of Setting Source and Destination Location Hints

```c
// 为源和目标缓冲区分配内存
std::vector<void *> srcs(batch_size);
std::vector<void *> dsts(batch_size);
std::vector<void *> sizes(batch_size);

// 我们将用于提供位置提示的 cudaMemLocation 结构
// Device device_id
cudaMemLocation srcLoc = {cudaMemLocationTypeDevice, dev_id};
// Host 带有 numa Node numa_id
cudaMemLocation dstLoc = {cudaMemLocationTypeHostNuma, numa_id};

// 为 src 和 dst 缓冲区分配内存
for (size_t i = 0; i < batch_size; i++) {
    cudaMallocManaged(&srcs[i], sizes[i]);
    cudaMallocManaged(&dsts[i], sizes[i]);

    cudaMemPrefetchAsync(srcs[i], sizes[i], srcLoc, 0, stream);
    cudaMemPrefetchAsync(dsts[i], sizes[i], dstLoc, 0, stream);
    cudaMemsetAsync(srcs[i], sizes[i], stream);
}
// 为这一批拷贝设置 attributes
cudaMemcpyAttributes attrs = {};

// 这些是 managed memory pointers，因此适合使用 Stream Order
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

// 现在我们可以在这里指定位置提示
attrs.srcLocHint = srcLoc;
attrs.dstlocHint = dstLoc;

// batch 中所有的拷贝具有相同的 copy attributes
size_t attrsIdxs = 0;
// 发起 batched memory transfer
cudaMemcpyBatchAsync(&dsts[0], &srcs[0], &sizes[0], batch_size,
    &attrs, &attrsIdxs, 1 /*numAttrs*/, nullptr /*failIdx*/, stream);
```

最后要介绍的一点是用于提示是否希望为传输使用 SMs 或 CEs 的 flags。这个字段是 `cudaMemcpyAttributesflags::flags`，可能的值包括：

  * `cudaMemcpyFlagDefault` – 默认行为
  * `cudaMemcpyFlagPreferOverlapWithCompute` – 这提示系统应优先使用 CEs 来让传输与计算重叠。这个 flag 在非 Tegra 平台上被忽略

总之，“cudaMemcpyBatchAsync” 相关的要点如下：

  * `cudaMemcpyBatchAsync` 函数（及其 3D 变体）允许程序员指定一批 memory transfers，从而摊薄传输设置开销。
  * 除了源和目标指针及传输大小之外，该函数可以接受一个或多个 memory copy attributes，这些属性提供有关正在传输的内存类型及对应源指针的 stream ordering 行为、有关源和目标位置的提示，以及是否希望与 compute 重叠（如果可能）或是否为传输使用 SMs 的提示。
  * 有了以上信息之后，runtime 就可以尽可能地优化传输。

### 3.1.6 环境变量

CUDA 提供了一系列环境变量（见第 5.2 节），它们会影响程序的执行行为和性能。如果用户没有显式设置这些环境变量，CUDA 会使用一组合理的默认值。不过在某些具体场景下，例如调试问题或追求更优性能时，可能需要针对具体情况对它们进行调整。

例如，适当增大 `CUDA_DEVICE_MAX_CONNECTIONS` 这个环境变量的值，可以降低来自不同 CUDA stream 的独立工作因为“假依赖”而被串行化执行的概率。当多个操作使用了相同的底层资源时，就可能引入这类假依赖。一般建议先使用默认值，只有在确实遇到性能问题时再去探索它的影响，比如发现不同 CUDA stream 之间本应并行的工作出现了意料之外的串行化，而且这种现象无法用 SM 资源不足等其他因素来解释。需要注意的是，在使用 MPS 的情况下，这个环境变量的默认值会不同，而且通常更低。

类似地，对于对延迟非常敏感的应用，将 `CUDA_MODULE_LOADING` 这个环境变量设置为 `EAGER` 可能更合适。这样可以把所有与模块加载相关的开销，提前放到应用初始化阶段，从而避免出现在关键执行路径中。当前的默认行为是延迟（lazy）模块加载。在默认模式下，也可以通过在应用初始化阶段对各个 kernel 进行“预热”调用，来达到类似 eager 模块加载的效果，从而更早触发模块加载。

关于 CUDA 各种环境变量的更多细节，请参考 CUDA Environment Variables。建议在启动应用之前就将这些环境变量设置好；如果在应用运行过程中再去设置，可能不会生效。
