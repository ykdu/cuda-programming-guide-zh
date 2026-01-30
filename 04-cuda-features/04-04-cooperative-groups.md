# 4.4 Cooperative Groups

### 4.4.1. 介绍

Cooperative Groups 是对 CUDA programming model 的一种扩展，用于组织协作的 threads 组。Cooperative Groups 允许开发者控制 threads 协作的粒度，从而能够表达更丰富、更高效的并行分解方式。Cooperative Groups 还提供了常见并行原语（例如 scan 和 parallel reduce）的实现。

从历史上看，CUDA programming model 只提供了一种简单的同步机制：在一个 thread block 内对所有 threads 进行 barrier，同步通过 `__syncthreads()` intrinsic function 实现。为了表达更复杂的并行交互模式，许多以性能为导向的程序员不得不自行编写临时性的、且不安全的同步原语，用于在单个 warp 内，或在运行于同一 GPU 上的多个 thread blocks 之间进行同步。尽管这些方式通常能够带来性能提升，但也导致了大量脆弱代码的积累，这些代码编写困难、调优复杂，并且在不同 GPU 代际之间维护成本很高。Cooperative Groups 提供了一种安全且具备前瞻性的机制，用于编写高性能代码。

完整的 Cooperative Groups API 可在 Cooperative Groups API 文档中查看。

### 4.4.2. Cooperative Group Handle 与成员函数

Cooperative Groups 通过 Cooperative Group Handle 进行管理。Cooperative Group handle 允许参与的 threads 获取其在 group 中的位置、group 的大小以及其他 group 信息。部分 group 成员函数如下表所示。

**表 10：部分成员函数**

| 访问函数         | 返回值                                            |
| ---------------- | ------------------------------------------------- |
| `thread_rank()`  | 当前调用 thread 在 group 中的 rank。              |
| `num_threads()`  | group 中的 threads 总数。                         |
| `thread_index()` | 该 thread 在 launch 的 block 中的三维 index。     |
| `dim_threads()`  | launch 的 block 的三维尺寸（以 threads 为单位）。 |

完整的成员函数列表可在 Cooperative Groups API 中查看。

### 4.4.3. 默认行为 / 无显式分组执行

表示 grid 和 thread blocks 的 groups 会根据 kernel 的 launch 配置被隐式创建。这些“隐式”groups 为开发者提供了一个起点，开发者可以在此基础上将其进一步显式拆分为更细粒度的 groups。可以通过以下方式访问这些隐式 groups：

**表 11：由 CUDA Runtime 隐式创建的 Cooperative Groups**

| 访问函数                  | Group 范围                                                   |
| ------------------------- | ------------------------------------------------------------ |
| `this_thread_block()`     | 返回一个包含当前 thread block 中所有 threads 的 group handle。 |
| `this_grid()`             | 返回一个包含整个 grid 中所有 threads 的 group handle。       |
| `coalesced_threads()` [1] | 返回一个包含当前 warp 中处于 active 状态的 threads 的 group handle。 |
| `this_cluster()` [2]      | 返回当前 cluster 中 threads 的 group handle。                |

[1] `coalesced_threads()` 运算符返回当前时间点处于 active 状态的 threads 集合，不保证返回哪些 threads（只要它们是 active 的），也不保证它们在整个执行过程中始终保持 coalesced。

[2] 当 launch 的 grid 不是 cluster grid 时，`this_cluster()` 假定 cluster 为 1x1x1。需要 Compute Capability 9.0 或更高版本。

更多信息可在 Cooperative Groups API 中查看。

#### 4.4.3.1. 尽早创建隐式 Group Handle

为了获得最佳性能，建议在 kernel 中尽早创建隐式 group 的 handle（越早越好，在发生任何分支之前），并在整个 kernel 生命周期中复用该 handle。

#### 4.4.3.2. 仅通过引用传递 Group Handle

当将 group handle 作为参数传递给函数时，建议通过引用方式传递。Group handle 必须在声明时完成初始化，因为不存在默认构造函数。不建议对 group handle 进行拷贝构造。

### 4.4.4. 创建 Cooperative Groups

Groups 通过将一个父 group 划分为若干子 groups 来创建。当一个 group 被划分时，会创建一个 group handle 来管理生成的子 group。开发者可以使用以下划分操作：

**表 12：Cooperative Group 划分操作**

| 划分类型            | 描述                                                         |
| ------------------- | ------------------------------------------------------------ |
| `tiled_partition`   | 将父 group 划分为一系列固定大小的子 groups，按一维、行优先（row-major）方式排列。 |
| `stride_partition`  | 将父 group 划分为大小相等的子 groups，threads 以 round-robin 的方式分配到各个子 groups。 |
| `labeled_partition` | 根据条件标签将父 group 划分为一维子 groups，标签可以是任意整数类型。 |
| `binary_partition`  | `labeled_partition` 的特化形式，标签只能为 “0” 或 “1”。      |

以下示例展示如何创建一个 tiled partition：

```c
namespace cg = cooperative_groups;
// 获取当前 thread 的 cooperative group
cg::thread_block my_group = cg::this_thread_block();

// 将 cooperative group 划分为大小为 8 的 tiles
cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(cta);

// 以 my_subgroup 的身份执行工作
```

最佳的划分策略取决于具体场景。更多信息可参考 Cooperative Groups API。

#### 4.4.4.1. 避免 Group 创建风险

划分 group 是一种 collective 操作，group 中的所有 threads 都必须参与。如果 group 的创建位于某个并非所有 threads 都能到达的条件分支中，可能会导致死锁或数据损坏。

### 4.4.5. 同步

在 Cooperative Groups 引入之前，CUDA programming model 仅允许在 kernel 完成边界进行 thread blocks 之间的同步。Cooperative Groups 允许开发者在不同粒度上对协作 threads 的 groups 进行同步。

#### 4.4.5.1. Sync

可以通过调用 collective 函数 `sync()` 来同步一个 group。与 `__syncthreads()` 类似，`sync()` 提供以下保证：

- 在同步点之前，group 中 threads 执行的所有 memory 访问（例如 reads 和 writes），在同步点之后对 group 中所有 threads 可见。
- 在任意 thread 继续执行之前，group 中所有 threads 都必须到达该同步点。

以下示例展示了一个等价于 `__syncthreads()` 的 `cooperative_groups::sync()`：

```c
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

// 同步 block 中的 threads
cg::sync(my_group);
```

Cooperative Groups 也可用于同步整个 grid。从 CUDA 13 起，Cooperative Groups 不再支持 multi-device 同步。详细信息请参阅 Large Scale Groups 章节。

更多关于同步的信息可参考 Cooperative Groups API。

#### 4.4.5.2. Barriers

Cooperative Groups 提供了一个类似于 `cuda::barrier` 的 barrier API，可用于更高级的同步场景。Cooperative Groups 的 barrier API 与 `cuda::barrier` 存在以下几个关键区别：

- Cooperative Groups 的 barriers 会自动初始化
- group 中所有 threads 在每个 phase 中都必须调用一次 barrier 并等待
- `barrier_arrive` 返回一个 `arrival_token` 对象，该对象必须传递给对应的 `barrier_wait`，并在其中被消费，之后不能再次使用

在使用 Cooperative Groups barriers 时，程序员必须注意避免以下风险：

- 在调用 `barrier_arrive` 之后、调用 `barrier_wait` 之前，group 不能执行任何 collective 操作
- `barrier_wait` 只保证 group 中所有 threads 都调用了 `barrier_arrive`
- `barrier_wait` 不保证所有 threads 都调用了 `barrier_wait`

```c
namespace cg = cooperative_groups;

cg::thread_block my_group = this_block();

auto token = cluster.barrier_arrive();

// 可选：执行一些本地处理以隐藏同步延迟
local_processing(block);

// 确保 cluster 中其他 blocks 已经运行并初始化共享数据
cluster.barrier_wait(std::move(token));
```

### 4.4.6. Collective 操作

Cooperative Groups 提供了一组 collective 操作，可由一个 group 的 threads 共同执行。这些操作要求指定 group 中的所有 threads 参与，才能完成该操作。

除非 Cooperative Groups API 明确允许使用不同的值，否则 group 中所有 threads 在每次 collective 调用时，必须为对应参数传入相同的值。否则该调用的行为是未定义的。

#### 4.4.6.1. Reduce

`reduce` 函数用于对指定 group 中每个 thread 提供的数据执行并行 reduction。reduction 的类型必须通过提供下表中的某个 operator 来指定。

**表 13：Cooperative Groups Reduction Operators**

| Operator | 返回值             |
| -------- | ------------------ |
| plus     | group 中所有值的和 |
| less     | 最小值             |
| greater  | 最大值             |
| bit_and  | 按位 AND reduction |
| bit_or   | 按位 OR reduction  |
| bit_xor  | 按位 XOR reduction |

当硬件支持时，reduction 会使用硬件加速（需要 Compute Capability 8.0 或更高）。对于不支持硬件加速的旧硬件，提供 software fallback。仅 4B 类型会获得硬件加速。

关于 reduction 的更多信息可参考 Cooperative Groups API。

以下示例展示如何使用 `cooperative_groups::reduce()` 执行 block 范围内的求和 reduction。

```c
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

int val = data[threadIdx.x];

int sum = cg::reduce(cta, val, cg::plus<int>());

// 存储 reduction 结果
if (my_group.thread_rank() == 0) {
    result[blockIdx.x] = sum;
}
```

#### 4.4.6.2. Scans

Cooperative Groups 提供了 `inclusive_scan` 和 `exclusive_scan` 的实现，可用于任意 group 大小。这些函数会对指定 group 中每个 thread 提供的数据执行 scan 操作。

程序员可以选择性地指定一个 reduction operator，如上方 Reduction Operators 表所列。

```c
namespace cg = cooperative_groups;

cg::thread_block my_group = cg::this_thread_block();

int val = data[my_group.thread_rank()];

int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());

result[my_group.thread_rank()] = exclusive_sum;
```

关于 scan 的更多信息可参考 Cooperative Groups Scan API。

#### 4.4.6.3. Invoke One

Cooperative Groups 提供了 `invoke_one` 函数，用于当某个 group 需要由单个 thread 代表其执行一段串行代码时使用。`invoke_one` 会从调用 group 中选择一个任意的 thread，并由该 thread 使用提供的参数调用指定的可调用函数。`invoke_one_broadcast` 与 `invoke_one` 相同，但调用结果会被广播到 group 中的所有 threads。

thread 的选择机制不保证是确定性的。

以下示例展示了 `invoke_one` 的基本用法。

```c
namespace cg = cooperative_groups;
cg::thread_block my_group = cg::this_thread_block();

// 确保只有一个 thread 在该 thread block 中打印消息
cg::invoke_one(my_group, []() {
    printf("Hello from one thread in the block!");
});

// 同步，确保所有 threads 等待消息打印完成
cg::sync(my_group);
```

在 invocable function 内部，不允许在调用 group 内进行通信或同步。允许与调用 group 之外的 threads 进行通信。

### 4.4.7. 异步数据移动

Cooperative Groups 在 CUDA 中提供的 `memcpy_async` 功能，提供了一种在 global memory 与 shared memory 之间执行异步 memory copy 的方式。`memcpy_async` 对于优化 memory transfer、以及通过将 computation 与 data transfer 重叠来提升性能尤为有用。

`memcpy_async` 函数用于启动一次从 global memory 到 shared memory 的异步加载。`memcpy_async` 的设计用途类似于一种“prefetch”，即在数据真正被使用之前先将其加载进来。

`wait` 函数会强制 group 中的所有 threads 等待，直到异步 memory transfer 完成。在访问 shared memory 中的数据之前，group 中的所有 threads 都必须调用 `wait`。

下面的示例展示了如何使用 `memcpy_async` 和 `wait` 来进行数据预取。

```c
namespace cg = cooperative_groups;

cg::thread_group my_group = cg::this_thread_block();

__shared__ int shared_data[];

// 从 global memory 到 shared memory 执行异步拷贝
cg::memcpy_async(my_group,
                 shared_data + my_group.rank(),
                 input + my_group.rank(),
                 sizeof(int));

// 在这里执行其他 work 以隐藏延迟，此时不能使用 shared_data

// 等待异步拷贝完成
cg::wait(my_group);

// 预取的数据现在可以使用
```

更多信息可参考 Cooperative Groups API。

#### 4.4.7.1. Memcpy Async 对齐要求

只有当 source 是 global memory、destination 是 shared memory，且两者都至少满足 4-byte 对齐时，`memcpy_async` 才是异步的。为了获得最佳性能，建议 shared memory 和 global memory 均采用 16-byte 对齐。

### 4.4.8. 大规模 Groups

Cooperative Groups 支持跨越整个 grid 的大规模 groups。前文介绍的所有 Cooperative Groups 功能都适用于这些大规模 groups，但有一个显著的例外：对整个 grid 进行同步需要使用 `cudaLaunchCooperativeKernel` runtime launch API。

从 CUDA 13 开始，multi-device launch APIs 以及与 Cooperative Groups 相关的引用已被移除。

#### 4.4.8.1. 何时使用 `cudaLaunchCooperativeKernel`

`cudaLaunchCooperativeKernel` 是一个 CUDA runtime API，用于 launch 使用 cooperative groups 的单 device kernel，专门用于需要跨 thread blocks 同步的 kernel。该函数确保 kernel 中的所有 threads 都可以在整个 grid 范围内进行同步与协作，而这在传统 CUDA kernels 中是无法实现的，因为传统 kernels 只允许在单个 thread block 内进行同步。`cudaLaunchCooperativeKernel` 还确保 kernel 的 launch 是原子的，也就是说，只要该 API 调用成功，所提供数量的 thread blocks 就一定会在指定 device 上被 launch。

一个良好的实践是在使用前，先通过查询 device 属性 `cudaDevAttrCooperativeLaunch` 来确认 device 是否支持 cooperative launch：

```
int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch,
                       cudaDevAttrCooperativeLaunch,
                       dev);
```

如果 device 0 支持该属性，`supportsCoopLaunch` 将被设置为 1。仅支持 compute capability 为 6.0 及以上的 devices。此外，还需要运行在以下环境之一：

- 未启用 MPS 的 Linux 平台
- 启用了 MPS 且 device 的 compute capability 为 7.0 或更高的 Linux 平台
- 最新版本的 Windows 平台
