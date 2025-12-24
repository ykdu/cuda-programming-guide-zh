# 4.6 Green Contexts

green context（GC）是一种轻量级的 context，从它被创建时起就与一组特定的 GPU 资源相关联。用户可以在创建 green context 时对 GPU 资源（当前是 streaming multiprocessors（SMs）和 work queues（WQs））进行划分，使得针对 green context 的 GPU 工作只能使用其已分配的 SMs 和 work queues。这样做有助于减少或更好地控制由于使用公共资源而引起的干扰。一个应用可以拥有多个 green context。

使用 green contexts 不需要对任何 GPU 代码（kernel）进行更改，只需在 host 端做少量改动（例如，创建 green context 以及为该 green context 创建 stream）。green context 功能在各种场景中可能是有用的。例如，它可以帮助确保某些 SM 始终可用于启动延迟敏感型 kernel 的执行（假设没有其他约束），或者提供一种无需修改任何 kernel 即可快速测试使用更少 SM 的影响的方法。

green context 支持最早通过 CUDA Driver API 提供。从 CUDA 13.1 开始，contexts 通过 execution context（EC）抽象在 CUDA runtime 中暴露出来。目前，一个 execution context 可以对应 primary context（即 runtime API 用户一直隐式交互的 context）或 green context。本节在提及 green context 时将 execution context 和 green context 这两个术语交替使用。

随着 green contexts 在 runtime 中的暴露，强烈建议直接使用 CUDA runtime API。本节也将仅使用 CUDA runtime API。

本节其余内容组织如下：Section 4.6.1 提供一个动机示例，Section 4.6.2 强调易用性，Section 4.6.3 介绍 device resource 和 resource descriptor 结构体。Section 4.6.4 解释如何创建 green context，Section 4.6.5 解释如何启动针对它的工作，Section 4.6.6 强调一些额外的 green context API。最后，Section 4.6.7 以一个示例结束。



### 4.6.1. 动机 / 使用时机

在启动一个 CUDA kernel 时，用户无法直接控制该 kernel 将使用多少个 SMs 执行。用户只能通过改变 kernel 的 launch geometry，或者通过任何会影响 kernel 在每个 SM 上可同时驻留的最大 active thread blocks 数量的因素，来间接影响这一行为。此外，当多个 kernel 在 GPU 上并行执行时（例如在不同的 CUDA streams 上运行，或作为 CUDA graph 的一部分运行），它们也可能争用相同的 SM 资源。

然而，在某些使用场景中，用户需要确保始终有 GPU 资源可用于延迟敏感型工作及时启动，并因此尽可能快地完成。green contexts 通过对 SM 资源进行划分提供了一种实现方式，使得某个给定的 green context 只能使用特定的 SMs（即在其创建时已配置的那些 SMs）。

Figure 42 展示了这样一个示例。假设某个应用中有两个相互独立的 kernels A 和 B，分别运行在两个不同的 non-blocking CUDA streams 上。Kernel A 先被启动，并开始执行，占用了所有可用的 SM 资源。随后在某一时刻，当延迟敏感的 kernel B 被启动时，已经没有任何 SM 资源可用。结果是，kernel B 只能在 kernel A 开始回落之后才能启动执行，即只有当 kernel A 的 thread blocks 执行完成后才能开始执行。第一幅图展示了这种关键工作 B 被延迟的场景。纵轴表示被占用的 SM 百分比，横轴表示时间。

![Figure 42: Motivation — GCs’ static resource partitioning enables latency-sensitive work B to start and complete sooner](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/green_contexts_motivation.png)

*Figure 42（动机）*  
*GCs 的静态资源划分使延迟敏感型工作 B 能够更早开始执行并更早完成。*

使用 green contexts，可以对 GPU 的 SMs 进行划分，使得由 kernel A 目标指向的 green context A 可以访问 GPU 的一部分 SMs，而由 kernel B 目标指向的 green context B 可以访问剩余的 SMs。在这种设置下，无论 kernel A 的 launch configuration 如何，它都只能使用分配给 green context A 的那些 SMs。这样一来，当关键的 kernel B 被启动时，可以保证其立即有可用的 SMs 开始执行，除非受到其他资源约束的影响。如 Figure 42 中第二幅图所示，尽管 kernel A 的执行时长可能会增加，但延迟敏感型工作 B 将不再因为 SM 不可用而被延迟。图中为了说明起见，green context A 被配置为拥有相当于 GPU 总 SM 数量 80% 的 SMs。

这种行为可以在不对 kernels A 和 B 做任何代码修改的情况下实现。用户只需要确保它们被启动在属于相应 green contexts 的 CUDA streams 上即可。每个 green context 可以访问的 SM 数量应由用户在创建 green context 时根据具体情况进行决定。

**Work Queues**

Streaming multiprocessors 是一种可以为 green context 配置的资源类型。另一种资源类型是 work queues。可以将 workqueue 理解为一种黑盒的资源抽象，它与其他因素一起，也会影响 GPU 工作执行的并发性。如果相互独立的 GPU 工作任务（例如提交在不同 CUDA streams 上的 kernels）映射到了同一个 workqueue，就可能在这些任务之间引入一种虚假的依赖关系，从而导致它们被串行执行。用户可以通过 CUDA_DEVICE_MAX_CONNECTIONS 环境变量来影响 GPU 上 work queues 的上限数量（参见 Section 5.2 和 Section 3.1）。

在前述示例的基础上，假设工作 B 映射到了与工作 A 相同的 workqueue。在这种情况下，即使 SM 资源是可用的（即使用了 green contexts），工作 B 仍然可能需要等待工作 A 完整执行结束。与 SMs 类似，用户也无法直接控制底层实际使用的具体 work queues。但 green contexts 允许用户以期望的并发 stream-ordered 工作负载数量的形式，表达其期望的最大并发度。driver 随后可以将该数值作为一种提示，尽量避免来自不同 execution contexts 的工作使用相同的 workqueue(s)，从而防止 execution contexts 之间出现不必要的干扰。

> [!注意事项]
> 即使为不同的 green context 分别配置了不同的 SM 资源和 work queues，独立 GPU 工作的并发执行也并不能得到保证。最好将 Green Contexts 章节中描述的所有技术理解为：用于移除可能阻止并发执行的因素（即减少潜在的干扰），而不是保证并发执行本身。

**Green Contexts versus MIG or MPS**

为完整起见，本节将 green contexts 与另外两种资源划分机制进行简要对比：MIG（Multi-Instance GPU）和 MPS（Multi-Process Service）。

MIG 会将支持 MIG 的 GPU 静态划分为多个 MIG instances（“更小的 GPUs”）。这种划分必须在应用启动之前完成，不同的应用可以使用不同的 MIG instances。对于那些长期无法充分利用 GPU 可用资源的应用而言，使用 MIG 是有益的，这种问题在 GPU 规模不断增大的情况下尤为明显。通过 MIG，用户可以将不同的应用运行在不同的 MIG instances 上，从而提升 GPU 利用率。对于云服务提供商（CSPs）来说，MIG 不仅因能够提高此类应用的 GPU 利用率而具有吸引力，还因为它可以在运行于不同 MIG instances 上的客户之间提供质量保障（QoS）和隔离性。更多细节请参阅上文链接的 MIG 文档。

然而，使用 MIG 无法解决前面描述的那个问题场景，即关键工作 B 因为同一应用中的其他 GPU 工作占用了所有 SM 资源而被延迟。即使应用运行在单个 MIG instance 上，这个问题仍然可能存在。为了解决这一问题，可以将 green contexts 与 MIG 结合使用。在这种情况下，可用于划分的 SM 资源将是该 MIG instance 所拥有的资源。

MPS 主要面向不同的 processes（例如 MPI 程序），允许它们在不进行 time-slicing 的情况下同时运行在 GPU 上。它要求在应用启动之前先运行一个 MPS daemon。默认情况下，MPS clients 会争用其运行所在 GPU 或 MIG instance 的所有可用 SM 资源。在这种多 client processes 的设置下，MPS 可以通过 active thread percentage 选项支持对 SM 资源进行动态划分，该选项为某个 MPS client process 可以使用的 SM 百分比设置了一个上限。与 green contexts 不同，MPS 中的 active thread percentage 划分发生在 process 层级，并且该百分比通常在应用启动之前通过环境变量进行指定。MPS 的 active thread percentage 表示某个 client application 最多只能使用 GPU 的 x% 的 SMs，即 N 个 SMs。然而，这些 SMs 可以是 GPU 上的任意 N 个 SMs，并且随着时间可能发生变化。相对而言，在创建时被配置为拥有 N 个 SMs 的 green context，只能使用这特定的 N 个 SMs。

从 CUDA 13.1 开始，如果在启动 MPS control daemon 时显式启用，MPS 也支持 static partitioning。在 static partitioning 模式下，用户需要在应用启动时指定某个 MPS client process 可以使用的静态分区，此时将不再适用基于 active thread percentage 的动态共享。MPS 的 static partitioning 模式与 green contexts 的一个关键区别在于，MPS 面向的是不同的 processes，而 green contexts 也可以应用在单个 process 内。此外，与 green contexts 不同，采用 static partitioning 的 MPS 不允许对 SM 资源进行 oversubscription。

在 MPS 中，对于通过 cuCtxCreate driver API 创建并设置了 execution affinity 的 CUDA context，也可以进行程序化的 SM 资源划分。这种程序化划分允许来自一个或多个 processes 的不同 client CUDA contexts 各自使用不超过指定数量的 SMs。与 active thread percentage 划分类似，这些 SMs 可以是 GPU 上的任意 SMs，并且会随时间变化，这一点不同于 green contexts 的情况。即使在启用了 static MPS partitioning 的情况下，该选项仍然是可用的。需要注意的是，与 MPS context 相比，创建 green context 要轻量得多，因为许多底层结构由 primary context 所拥有并因此被共享。

### 4.6.2. Green Contexts: Ease of use

为了强调使用 green contexts 是多么简单，假设你有如下代码片段，该片段创建两个 CUDA streams，然后调用一个在这些 CUDA streams 上通过 `<<<>>>` 启动 kernels 的函数。如前所述，除了改变 kernels 的 launch geometry 之外，人们无法影响这些 kernels 可以使用多少个 SMs。

``` c++
int gpu_device_index = 0; // GPU 序号
CUDA_CHECK(cudaSetDevice(gpu_device_index));

cudaStream_t strm1, strm2;
CUDA_CHECK(cudaStreamCreateWithFlags(&strm1, cudaStreamNonBlocking));
CUDA_CHECK(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

// 无法控制在每个 stream 上运行的 kernel(s) 可以使用多少个 SMs
code_that_launches_kernels_on_streams(strm1, strm2); // 此函数及 kernels 内部被抽象掉的部分是你的大部分代码

// 清理代码未显示
```

从 CUDA 13.1 开始，可以使用 green contexts 控制某个 kernel 可以访问的 SM 数量。下面的代码片段展示了这是多么简单。通过几行额外的代码且无需修改任何 kernel，你就可以控制在这些不同 streams 上启动的 kernel(s) 可以使用的 SMs 资源。

``` c++
int gpu_device_index = 0; // GPU 序号
CUDA_CHECK(cudaSetDevice(gpu_device_index));

/* ------------------ 创建 green contexts 所需的代码 --------------------------- */

// 获取所有可用的 GPU SM 资源
cudaDevResource initial_GPU_SM_resources {};
CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));

// 拆分 SM 资源。本例创建一个拥有 16 个 SM 的组和一个拥有 8 个 SM 的组。假设你的 GPU 有 >= 24 个 SM
cudaDevSmResource result[2] {{}, {}};
cudaDevSmResourceGroupParams group_params[2] =  {
        {.smCount=16, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0},
        {.smCount=8,  .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0}};
CUDA_CHECK(cudaDevSmResourceSplit(&result[0], 2, &initial_GPU_SM_resources, nullptr, 0, &group_params[0]));

// 为每个资源生成 resource descriptors
cudaDevResourceDesc_t resource_desc1 {};
cudaDevResourceDesc_t resource_desc2 {};
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc1, &result[0], 1));
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc2, &result[1], 1));

// 创建 green contexts
cudaExecutionContext_t my_green_ctx1 {};
cudaExecutionContext_t my_green_ctx2 {};
CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx1, resource_desc1, gpu_device_index, 0));
CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx2, resource_desc2, gpu_device_index, 0));

/* ------------------ 修改后的代码 --------------------------- */

// 你只需要用不同的 CUDA API 来创建 streams
cudaStream_t strm1, strm2;
CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm1, my_green_ctx1, cudaStreamDefault, 0));
CUDA_CHECK(cudaExecutionCtxStreamCreate(&strm2, my_green_ctx2, cudaStreamDefault, 0));

/* ------------------ 未改变的代码 --------------------------- */

// 不需要修改此函数中的任何代码或你 的 kernel(s)。
// 提醒：此函数 + kernels 内被抽象掉的部分是你大部分的代码
// 现在在 stream strm1 上运行的 kernel(s) 最多将使用 16 个 SMs，在 strm2 上的最多使用 8 个 SMs。
code_that_launches_kernels_on_streams(strm1, strm2);

// 清理代码未显示

```

各种 execution context APIs（其中一些在前面的示例中已经展示）都接受一个显式的 `cudaExecutionContext_t` handle，因此会忽略当前线程所处的 context。到目前为止，不使用 driver API 的 CUDA runtime 用户默认只能与通过 `cudaSetDevice()` 隐式设置为线程当前的 primary context 交互。这种向显式基于 context 编程的转变提供了更易于理解的语义，并且与之前依赖 thread-local state（TLS）的隐式基于 context 编程相比，还可以带来额外的好处。

下面几节将详细解释前面代码片段中显示的所有步骤。

### 4.6.3. Green Contexts: Device Resource and Resource Descriptor

green context 的核心是与特定 GPU 设备关联的 device resource（`cudaDevResource`）。资源可以组合并封装到一个描述符中（`cudaDevResourceDesc_t`）。green context 只能访问创建它时所用描述符中封装的那些资源。

```c++
struct {
     enum cudaDevResourceType type; 
     union {
         struct cudaDevSmResource sm; 
         struct cudaDevWorkqueueConfigResource wqConfig; 
         struct cudaDevWorkqueueResource wq; 
     };
 };
```

上述是当前 cudaDevResource 数据结构的定义。

支持的有效资源类型包括 `cudaDevResourceTypeSm`、`cudaDevResourceTypeWorkqueueConfig` 和 `cudaDevResourceTypeWorkqueue`，而 `cudaDevResourceTypeInvalid` 表示一种无效的资源类型。

一个有效的 device resource 可以与以下项目关联：

- 一组特定的 streaming multiprocessors（SMs）（资源类型为 `cudaDevResourceTypeSm`）
- 一种特定的 workqueue 配置（资源类型为 `cudaDevResourceTypeWorkqueueConfig`）
- 一个已存在的 workqueue 资源（资源类型为 `cudaDevResourceTypeWorkqueue`）

可以使用 `cudaExecutionCtxGetDevResource` 和 `cudaStreamGetDevResource` API 分别检查某个 execution context 或 CUDA stream 是否关联了某种类型的 `cudaDevResource` 资源。execution context 有可能同时关联不同类型的 device resources（例如 SMs 和 work queues），而 stream 只能关联 SM 类型的资源。

默认情况下，某个 GPU 设备拥有三种 device resource 类型：一种包含 GPU 全部 SMs 的 SM 类型资源、一种包含所有可用 work queues 的 workqueue 配置资源以及与之对应的 workqueue 资源。这些资源可以通过 `cudaDeviceGetDevResource` API 获取。

**相关 device resource 结构体的概览**

不同类型的 resource 结构体都包含一些字段，这些字段要么由用户显式设置，要么由相关的 CUDA API 调用来填充。建议对所有 device resource 结构体进行 zero-init 初始化。

- SM 类型的 device resource（`cudaDevSmResource`）包含以下几个相关字段：
  - `unsigned int smCount`：该资源中可用的 SM 数量
  - `unsigned int minSmPartitionSize`：对该资源进行划分所需的最小 SM 数量
  - `unsigned int smCoscheduledAlignment`：保证会被 co-schedule 到同一个 GPU processing cluster 上的 SM 数量，该字段与 thread block clusters 相关。当 `flags` 为 0 时，`smCount` 必须是该值的整数倍。
  - `unsigned int flags`：支持的 flags 包括 0（默认值）以及 `cudaDevSmResourceGroupBackfill`（参见 `cudaDevSmResourceGroup` 的 flags 定义）。

上述字段要么由用于创建该 SM 类型资源的相应 split API（`cudaDevSmResourceSplitByCount` 或 `cudaDevSmResourceSplit`）进行设置，要么由 `cudaDeviceGetDevResource` API 填充，该 API 用于获取某个 GPU 设备的 SM 资源。这些字段不应由用户直接设置。更多细节请参见下一节。

- workqueue configuration 类型的 device resource（`cudaDevWorkqueueConfigResource`）包含以下几个相关字段：
  - `int device`：workqueue 资源所在的 device
  - `unsigned int wqConcurrencyLimit`：期望的 stream-ordered 工作负载数量，用于避免出现虚假的依赖关系
  - `enum cudaDevWorkqueueConfigScope sharingScope`：workqueue 资源的共享范围。支持的取值包括 `cudaDevWorkqueueConfigScopeDeviceCtx`（默认值）和 `cudaDevWorkqueueConfigScopeGreenCtxBalanced`。在默认模式下，所有 workqueue 资源会在所有 contexts 之间共享；而在 balanced 模式下，driver 会尽量在不同 green contexts 之间使用不重叠的 workqueue 资源，并将用户指定的 `wqConcurrencyLimit` 作为参考提示。

这些字段需要由用户进行设置。目前不存在类似 split APIs 那样、用于生成 workqueue configuration resource 的 CUDA API，唯一的例外是由 `cudaDeviceGetDevResource` API 填充的 workqueue configuration resource。该 API 可以获取某个 GPU 设备的 workqueue configuration resources。

- 最后，预先存在的 workqueue resource（`cudaDevResourceTypeWorkqueue`）没有任何可以由用户设置的字段。与其他资源类型一样，可以通过 `cudaDevGetDevResource` 获取某个 GPU 设备上已有的 workqueue resource。

### 4.6.4 Green Context 创建示例

创建 green context 主要涉及四个步骤：

- 步骤 1：从初始资源开始，例如通过获取 GPU 上可用的资源
- 步骤 2：将 SM 资源划分为一个或多个分区（使用可用的分割 API 之一）
- 步骤 3：创建一个资源描述符，按需组合不同的资源
- 步骤 4：从该描述符创建 green context，并为其配置资源
   在 green context 创建完成之后，你可以创建属于该 green context 的 CUDA streams。随后在这样的 stream 上启动的 GPU 工作（例如通过 `<<< >>>` 启动的 kernel）将只能访问该 green context 所配置的资源。只要用户将属于 green context 的 stream 传递给库，库也可以很容易地利用 green context。有关详细信息，请参见 Green Contexts - Launching work。

#### 4.6.4.1 步骤 1：获取可用的 GPU 资源

创建 green context 的第一步是获取可用的 device 资源，并填充 `cudaDevResource` 结构体。目前有三种可能的起点：一个 device、一个 execution context 或一个 CUDA stream。
 相关的 CUDA runtime API 函数签名如下：

- 对于 device：`cudaError_t cudaDeviceGetDevResource(int device, cudaDevResource* resource, cudaDevResourceType type)`
- 对于 execution context：`cudaError_t cudaExecutionCtxGetDevResource(cudaExecutionContext_t ctx, cudaDevResource* resource, cudaDevResourceType type)`
- 对于 stream：`cudaError_t cudaStreamGetDevResource(cudaStream_t hStream, cudaDevResource* resource, cudaDevResourceType type)`

对于这些 API，所有有效的 `cudaDevResourceType` 类型都是允许的，除了 `cudaStreamGetDevResource` 只能支持 SM-类型资源。

通常，起点是 GPU device。下面的代码片段展示了如何获取给定 GPU device 的可用 SM 资源。在 `cudaDeviceGetDevResource` 调用成功之后，用户可以检查此资源中可用的 SM 数量。

```c++
    int current_device = 0; // 假定 device ordinal 为 0
    CUDA_CHECK(cudaSetDevice(current_device));
    cudaDevResource initial_SM_resources = {};
    CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU 设备 */,
                                       &initial_SM_resources /* 填充的 device 资源 */,
                                       cudaDevResourceTypeSm /* 资源类型 */));

    std::cout << "Initial SM resources: " << initial_SM_resources.sm.smCount << " SMs" << std::endl; // 可用 SM 数量
    // 与划分相关的特殊字段（参见下面的步骤 3）
    std::cout << "Min. SM partition size: " <<  initial_SM_resources.sm.minSmPartitionSize << " SMs" << std::endl;
    std::cout << "SM co-scheduled alignment: " <<  initial_SM_resources.sm.smCoscheduledAlignment << " SMs" << std::endl;
```

也可以获取可用的 workqueue config. 资源，如下代码所示：

```c++
    int current_device = 0; // 假定 device ordinal 为 0
    CUDA_CHECK(cudaSetDevice(current_device));
    cudaDevResource initial_WQ_config_resources = {};
    CUDA_CHECK(cudaDeviceGetDevResource(current_device /* GPU 设备 */,
                                       &initial_WQ_config_resources /* 填充的 device 资源 */,
                                       cudaDevResourceTypeWorkqueueConfig /* 资源类型 */));
    std::cout << "Initial WQ config. resources: " << std::endl;
    std::cout << "  - WQ 并发限制: " << initial_WQ_config_resources.wqConfig.wqConcurrencyLimit << std::endl;
    std::cout << "  - WQ 共享范围: " << initial_WQ_config_resources.wqConfig.sharingScope << std::endl;
```

在 `cudaDeviceGetDevResource` 调用成功之后，用户可以检查此资源的 `wqConcurrencyLimit`。当起点是 GPU device 时，`wqConcurrencyLimit` 将与 `CUDA_DEVICE_MAX_CONNECTIONS` 环境变量的值或其默认值一致。

#### 4.6.4.2 步骤 2：划分 SM 资源

创建绿色上下文的第二个步骤是**将可用的 `cudaDevResource` SM 资源静态划分**为一个或多个分区，并且可能会有一些 SM 留在剩余分区中。可以使用 `cudaDevSmResourceSplitByCount()` 或 `cudaDevSmResourceSplit()` API 来完成这种划分。其中，`cudaDevSmResourceSplitByCount()` API 只能创建一个或多个相同规格的分区，以及一个可能存在的剩余分区；而 `cudaDevSmResourceSplit()` API 则还可以创建不同规格的分区，以及一个可能的剩余分区。接下来的部分将详细说明这两个 API 的功能。这两种 API 都仅适用于 **SM 类型的 device resource**。 

**cudaDevSmResourceSplitByCount API**

`cudaDevSmResourceSplitByCount` 运行时 API 的函数原型是：

```c++
cudaError_t cudaDevSmResourceSplitByCount(cudaDevResource* result,
                                         unsigned int* nbGroups,
                                         const cudaDevResource* input,
                                         cudaDevResource* remaining,
                                         unsigned int useFlags,
                                         unsigned int minCount)
```

如图 43 所示，用户请求将 `input` 这一 SM 类型的 device resource 拆分成 `*nbGroups` 个每组拥有 `minCount` SM 的同类分区。
 不过最终结果中，**每组实际得到的 SM 数量 `N` 会大于或等于 `minCount`**，而 `*nbGroups` 的实际数量会**小于或等于用户最初请求的组数**。这些调整是由于某些粒度或对齐要求导致的，这些要求与具体 GPU 架构有关。 

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/green_contexts_resource_split_by_count.png)

*Figure 43*  
*使用cudaDevSmResourceSplitByCount API切分资源*

表 30 列出了当前所有受支持的 compute capability 在默认 `useFlags=0` 情况下的最小 SM partition size 和 SM co-scheduled alignment。你也可以通过 `cudaDevSmResource` 的 `minSmPartitionSize` 和 `smCoscheduledAlignment` 字段来获取这些值，如“Step 1: Get available GPU resources”部分所示。某些要求可以通过设置不同的 `useFlags` 值来降低。表 14 给出了一些相关示例，突出显示了请求值与最终结果之间的差异并附带解释。对于 compute capability 9.0，如果 `useFlags` 为零，则每个 partition 的最小 SM 数为 8，且 SM 数必须是 8 的倍数。

**表 14：Split functionality**

| Requested   |          |                                                   | Actual（for GH200 with 132 SMs） |               |                              |
| ----------- | -------: | ------------------------------------------------- | -------------------------------- | ------------: | ---------------------------- |
| `*nbGroups` | minCount | useFlags                                          | `*nbGroups with N SMs`           | Remaining SMs | Reason                       |
| 2           |       72 | 0                                                 | 1 组，72 个 SM                   |            60 | 不能超过 132 个 SM           |
| 6           |       11 | 0                                                 | 6 组，每组 16 个 SM              |            36 | 必须满足 “8 的倍数” 这个要求 |
| 6           |       11 | `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING` | 6 组，每组 12 个 SM              |            60 | 要求被降低为 “2 的倍数”      |
| 2           |        1 | 0                                                 | 2 组，每组 8 个 SM               |           116 | 最小 8 个 SM 的要求          |

注意：

1. `cudaDevSmResourceSplit` API 的返回值取决于 `result`：
    • 当 `result != nullptr` 时，只有在拆分成功并且创建了 `nbGroups` 个满足指定要求的有效 `cudaDevResource` group 时，API 才会返回 `cudaSuccess`；否则会返回错误。由于不同类型的错误可能返回相同的错误码（例如 `CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`），因此在开发过程中，建议使用 `CUDA_LOG_FILE` 环境变量来获取更详细的错误描述。
    • 当 `result == nullptr` 时，即使某个 group 得到的 `smCount` 为 0，API 也可能返回 `cudaSuccess`；而在 `result` 非 nullptr 的情况下，这种情况会返回错误。可以把这种模式看作一次 dry-run 测试，用来探索当前支持哪些配置，尤其适合在 discovery mode 下使用。

2. 当一次调用成功且 `result != nullptr` 时，索引 `i` 位于 `[0, nbGroups)` 范围内的 `result[i]` device resource 类型为 `cudaDevResourceTypeSm`，并且其 `result[i].sm.smCount` 要么是用户指定且非 0 的 `groupParams[i].smCount`，要么是通过 discovery 得到的值。无论哪种情况，`result[i].sm.smCount` 都会满足以下所有约束：

    • 是 2 的倍数；
    • 位于 `[2, input.sm.smCount]` 范围内；

    • 当 `flags == 0` 时，是实际 `group_params[i].coscheduledSmCount` 的倍数；否则，值大于或等于 `groups_params[i].coscheduledSmCount`。

3. 将 `coscheduledSmCount` 或 `preferredCoscheduledSmCount` 字段设为 0，表示应使用这些字段的默认值；这些默认值可能因 GPU 不同而有所差异。这两个默认值都等于通过 `cudaDeviceGetDevResource` API 为指定 device 获取到的 SM resource 的 `smCoscheduledAlignment`（而不是任意 SM resource 的）。如果需要查看这些默认值，可以在一次成功的 `cudaDevSmResourceSplit` 调用之后，检查对应 `groupParams` 条目中更新后的值，前提是最初将它们设为 0；详见下文。

```c++
int gpu_device_index = 0;
cudaDevResource initial_GPU_SM_resources {};
CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));
std::cout << "Default value will be equal to " << initial_GPU_SM_resources.sm.smCoscheduledAlignment << std::endl;

int default_split_flags = 0;
cudaDevSmResourceGroupParams group_params_tmp = {.smCount=0, .coscheduledSmCount=0, .preferredCoscheduledSmCount=0, .flags=0};
CUDA_CHECK(cudaDevSmResourceSplit(nullptr, 1, &initial_GPU_SM_resources, nullptr /*remainder*/, default_split_flags, &group_params_tmp));
std::cout << "coscheduledSmcount default value: " << group_params.coscheduledSmCount << std::endl;
std::cout << "preferredCoscheduledSmcount default value: " << group_params.preferredCoscheduledSmCount << std::endl;
```

4. 如果存在 remainder group，它在 SM 数量或 co-scheduling 要求上将不受任何限制。这部分由用户自行探索和决定。

在进一步介绍各个 `cudaDevSmResourceGroupParams` 字段的详细信息之前，表 16 展示了一些示例用例中这些值可能的取法。假设已经像前一个代码片段那样，初始化并填充了一个 `initial_GPU_SM_resources` device resource，并且它就是即将被拆分的资源。表中的每一行都基于同一个起始资源。为简化说明，表中只展示每个用例对应的 `nbGroups` 值以及可用于下方类似代码片段中的 `groupParams` 字段。

```c++
int nbGroups = 2; // update as needed
unsigned int default_split_flags = 0;
cudaDevResource remainder {}; // update as needed
cudaDevResource result_use_case[2] = {{}, {}}; // Update depending on number of groups planned. Increase size if you plan to also use a workqueue resource
cudaDevSmResourceGroupParams group_params_use_case[2] = {{.smCount = X, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0},
                                                         {.smCount = Y, .coscheduledSmCount=0, .preferredCoscheduledSmCount = 0, .flags = 0}}
CUDA_CHECK(cudaDevSmResourceSplit(&result_use_case[0], nbGroups, &initial_GPU_SM_resources, remainder, default_split_flags, &group_params_use_case[0]));
```

表 16：split API 使用场景

| #    | 目标 / 使用场景                                              | nbGroups | remainder   | smCount | coscheduledSmCount | preferredCoscheduledSmCount | flags                          | i    |
| ---- | ------------------------------------------------------------ | -------- | ----------- | ------- | ------------------ | --------------------------- | ------------------------------ | ---- |
| 1    | 一个使用 16 个 SM 的 resource。<br>不关心剩余 SM。<br>可以使用 cluster。 | 1        | nullptr     | 16      | 0                  | 0                           | 0                              | 0    |
| 2a   | 一个 resource 使用 16 个 SM，<br>另一个使用其余全部资源。<br>不使用 cluster。<br><br>说明：<br>在 (2a) 中，第 2 个 resource 是 remainder；<br>在 (2b) 中，第 2 个 resource 是 result_use_case[1]。 | 1（2a）  | not nullptr | 16      | 2                  | 2                           | 0                              | 0    |
| 2b   | 同上（第二种方式）                                           | 2（2b）  | nullptr     | 16      | 2                  | 2                           | 0                              | 0    |
|      |                                                              |          |             | 0       | 2                  | 2                           | cudaDevSmResourceGroupBackfill | 1    |
| 3    | 两个 resource，<br>分别使用 28 和 32 个 SM。<br>使用 size 为 4 的 cluster。 | 2        | nullptr     | 28      | 4                  | 4                           | 0                              | 0    |
|      |                                                              |          |             | 32      | 4                  | 4                           | 0                              | 1    |
| 4    | 一个 resource 使用尽可能多的 SM，<br>可运行 size 为 8 的 cluster，<br>并保留一个 remainder。 | 1        | not nullptr | 0       | 8                  | 8                           | 0                              | 0    |
| 5    | 一个 resource 使用尽可能多的 SM，<br>可运行 size 为 4 的 cluster，<br>另一个使用 8 个 SM。<br><br>注意：顺序很重要！<br>改变 groupParams 数组中条目的顺序，<br>可能导致 8-SM group 分配不到任何 SM。 | 2        | nullptr     | 8       | 2                  | 2                           | 0                              | 0    |
|      |                                                              |          |             | 0       | 4                  | 4                           | 0                              | 1    |

**关于各个 cudaDevSmResourceGroupParams struct 字段的详细说明**

`smCount`:

- 控制 `result` 中对应 group 所分配的 SM 数量。
- **取值：** 0（discovery mode），或者有效的非零值（non-discovery mode）。
  - 有效的非零 `smCount` 需要满足以下条件：必须是 2 的倍数，且位于 `[2, input->sm.smCount]` 范围内，并且当 `flags == 0` 时，必须是实际 `coscheduledSmCount` 的倍数；否则，必须大于或等于 `coscheduledSmCount`。
- **使用场景：** 当 SM 数量未知或尚未固定时，可以使用 discovery mode 来探索可行的配置；当需要明确请求特定数量的 SM 时，则使用 non-discovery mode。
- **注意：** 在 discovery mode 下，如果使用非 nullptr 的 `result` 并且 split 调用成功，最终得到的实际 SM 数量也会满足有效非零取值的所有要求。

`coscheduledSmCount`:

- 控制被组合在一起进行“co-scheduled”的 SM 数量，用于在 compute capability 9.0 及以上架构上支持不同 cluster 的启动。因此，它会影响最终 group 中的 SM 数量，以及这些 SM 能支持的 cluster size。
- **取值：** 0（当前架构的默认值），或者有效的非零值。
  - 有效的非零取值要求：必须是 2 的倍数，并且不超过最大限制。
- **使用场景：** 对于 cluster，可以使用默认值，或者手动选择一个值，但需要考虑当前架构所支持的最大可移植 cluster size。如果代码中不使用 cluster，可以使用最小支持值 2，或者直接使用默认值。
- **注意：** 当使用默认值时，在一次成功的 split 调用之后，实际得到的 `coscheduledSmCount` 同样会满足有效非零取值的要求。如果 `flags` 不为 0，最终得到的 `smCount` 将大于或等于 `coscheduledSmCount`。可以把 `coscheduledSmCount` 理解为为有效的结果 group 提供了一种有保障的底层“结构”，也就是说，在最坏情况下，该 group 至少能够运行一个大小为 `coscheduledSmCount` 的 cluster。这种结构性的保证并不适用于 remainder group；在 remainder group 中，具体能够启动哪些 cluster size，需要由用户自行探索。

`preferredCoscheduledSmCount`:

- 作为一个提示，告知 driver 在条件允许的情况下，尝试将实际大小为 `coscheduledSmCount` 的 SM group 合并成更大的、大小为 `preferredCoscheduledSmCount` 的 group。这样可以让代码使用 compute capability（CC）10.0 及以上设备所支持的 preferred cluster dimensions 功能。参见 `cudaLaunchAttributeValue::preferredClusterDim`。
- **取值**：0（当前架构的默认值），或者有效的非零值。
  - 有效的非零取值要求：必须是实际 `coscheduledSmCount` 的倍数。
- **使用场景**：如果你使用 preferred cluster，并且运行在 compute capability 10.0（Blackwell）或更高版本的设备上，可以手动选择一个大于 2 的值。如果不使用 cluster，则应选择与 `coscheduledSmCount` 相同的值：要么使用最小支持值 2，要么两个字段都设为 0。
- 注意：当使用默认值时，在一次成功的 split 调用之后，实际得到的 `preferredCoscheduledSmCount` 也会满足有效非零取值的要求。

`flags`:

- 用于控制最终 group 的 SM 数量是否必须是实际 `coscheduledSmCount` 的倍数（默认行为），还是允许将额外的 SM backfill 到该 group 中。在 backfill 的情况下，最终的 SM 数量（`result[i].sm.smCount`）将大于或等于指定的 `groupParams[i].smCount`。
- **取值**：0（默认值），或者 `cudaDevSmResourceGroupBackfill`。
- **使用场景**：使用 0（默认值）时，可以保证结果 group 具备支持多个 `coscheduledSmCount` size 的 cluster 的灵活性。如果希望在 group 中获得尽可能多的 SM，可以使用 backfill 选项，但其中一部分 SM（即 backfilled 的 SM）将不提供任何 coscheduling 保证。
- 注意：使用 backfill flag 创建的 group 仍然可以支持 cluster，例如可以保证至少支持一个大小为 `coscheduledSmCount` 的 cluster。

#### 4.6.4.3. Step 2 (continued): Add workqueue resources

如果你还想指定一个 workqueue 资源，那么需要显式完成这一步。下面的示例展示了如何为特定 device 创建一个 workqueue configuration 资源，该资源具有平衡的 sharing scope 并且并发限制为四。

```c++
cudaDevResource split_result[2] = {{}, {}};
// 未展示填充 split_result[0] 的代码；假设已调用 split API 并设置 nbGroups=1

// 最后一个资源将是 workqueue 资源。
split_result[1].type = cudaDevResourceTypeWorkqueueConfig;
split_result[1].wqConfig.device = 0; // 假设 device ordinal 是 0
split_result[1].wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
split_result[1].wqConfig.wqConcurrencyLimit = 4;

```

设置 workqueue concurrency limit 为四是向 driver 提示用户希望最多有四个并发的 stream-ordered 工作。driver 会在可能的情况下尝试按这个提示分配 work queues。

#### 4.6.4.4. Step 3: Create a Resource Descriptor

下一步，在资源被分割之后，是使用 `cudaDevResourceGenerateDesc` API 为预计会分配给 green context 的所有资源生成一个 resource descriptor。

相关的 CUDA runtime API 函数签名是：

```c++
cudaError_t cudaDevResourceGenerateDesc(cudaDevResourceDesc_t *phDesc, cudaDevResource *resources, unsigned int nbResources)
```

可以组合多个 `cudaDevResource` 资源。下面的代码示例展示了如何生成一个封装三组资源的 resource descriptor。你只需要确保这些资源在 `resources` 数组中是连续分配的。

```c++
cudaDevResource actual_split_result[5] = {};
// 未展示填充 actual_split_result 的代码

// 生成一个 resource desc，封装 3 个资源：actual_split_result[2] 到 [4]
cudaDevResourceDesc_t resource_desc;
CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc, &actual_split_result[2], 3));
```

也支持组合不同类型的资源。例如，你可以生成一个既包含 SM 又包含 workqueue 资源的 descriptor。

要让 `cudaDevResourceGenerateDesc` 调用成功：

- 所有 `nbResources` 资源都必须属于同一个 GPU device。
- 如果组合了多个 SM 型资源，它们应该来自同一次 split API 调用，并具有相同的 `coscheduledSmCount` 值（如果不属于 remainder 组）。
- 只能存在一个 workqueue config 或 workqueue 类型的资源。

#### 4.6.4.5. Step 4: Create a Green Context

最后一步是使用 `cudaGreenCtxCreate` API 从 resource descriptor 创建一个 green context。这个 green context 只能访问在创建时指定的 resource descriptor 中封装的资源（例如 SMs、work queues）。这些资源会在本步骤中被真正分配。

相关的 CUDA runtime API 函数签名是：

```c++
cudaError_t cudaGreenCtxCreate(cudaExecutionContext_t *phCtx, cudaDevResourceDesc_t desc, int device, unsigned int flags)
```

`flags` 参数应设置为 0。还建议在创建 green context 之前，通过 `cudaInitDevice` 或 `cudaSetDevice` API 明确初始化 device 的 primary context，这也会将 primary context 设置为当前线程的当前上下文。这样做可以确保在创建 green context 时不会有额外的 primary context 初始化开销。

看下面的代码示例：

```c++
int current_device = 0; // 假设只有一个 GPU
CUDA_CHECK(cudaSetDevice(current_device)); // 或者 cudaInitDevice

cudaDevResourceDesc_t resource_desc {};
// 未展示生成 resource_desc 的代码

// 在 current_device GPU 上创建一个 green_ctx，并让它访问 resource_desc 中的资源
cudaExecutionContext_t green_ctx {};
CUDA_CHECK(cudaGreenCtxCreate(&green_ctx, resource_desc, current_device, 0));
```

成功创建 green context 后，用户可以对该 execution context 的每种资源类型调用 `cudaExecutionCtxGetDevResource` 来验证其资源。

**创建多个 green contexts**

一个应用可以有多个 green context，这种情况下上面的一些步骤需要重复。对于大多数用例，这些 green contexts 会各自拥有一组不重叠的已分配 SMs。例如，对于五个同构 `cudaDevResource` 组（`actual_split_result` 数组），一个 green context 的 descriptor 可能封装 `actual_split_result[2]` 到 `[4]` 的资源，而另一个 green context 的 descriptor 可能封装 `actual_split_result[0]` 到 `[1]`。在这种情况下，某个特定的 SM 只会被分配给这两个 green context 中的一个。

但也可以使用 SM 过度订阅（oversubscription），并在某些情况下使用。例如，可以让第二个 green context 的 descriptor 封装 `actual_split_result[0]` 到 `[2]`。在此情况下，`actual_split_resource[2]` 的所有 SMs 会被过度订阅，也就是说这些 SMs 会被分配给两个 green contexts，而资源 `actual_split_resource[0]` 到 `[1]` 和 `actual_split_resource[3]` 到 `[4]` 可能只被其中一个 green context 使用。SM oversubscription 应根据具体情况谨慎使用。

### 4.6.5 Green Contexts — Launching work

要启动一个针对先前创建好的 green context 的 kernel，首先需要用 `cudaExecutionCtxStreamCreate` API 为这个 green context 创建一个 stream。然后，通过在那个 stream 上使用 `<<< >>>` 或 `cudaLaunchKernel` API 启动 kernel，就能确保该 kernel 只能使用这个 stream 所属 execution context 提供的资源（比如 SMs、work queues）。

```c++
// 为之前创建的 green_ctx green context 创建一个 green_ctx_stream CUDA stream
cudaStream_t green_ctx_stream;
int priority = 0;
CUDA_CHECK(cudaExecutionCtxStreamCreate(&green_ctx_stream,
                                        green_ctx,
                                        cudaStreamDefault,
                                        priority));

// Kernel my_kernel 将只使用 green_ctx_stream 的 execution context 所能访问到的资源（如 SMs、work queues）
my_kernel<<<grid_dim, block_dim, 0, green_ctx_stream>>>();
CUDA_CHECK(cudaGetLastError());
```

鉴于 `green_ctx` 是一个 green context，上面传给 stream 创建 API 的默认 flag 等价于 `cudaStreamNonBlocking`。

**CUDA graphs**

 对于作为 CUDA graph 一部分启动的 kernels（见 CUDA Graphs），还有一些细节需要注意。与 kernels 不同，CUDA graph 所在的 stream 并不决定所使用的 SM 资源，因为这个 stream 只是用于依赖关系追踪。

一个 kernel node（以及其他适用的 node 类型）将在哪个 execution context 上执行，是在 node 创建时设置的。如果 CUDA graph 是通过 stream capture 创建的，那么参与 capture 的 stream 的 execution context 会决定相关 graph nodes 的 execution context。如果 graph 是通过 graph API 创建的，那么用户应该对每个相关 node 显式设置 execution context。例如，添加一个 kernel node 时，用户应该使用多态的 `cudaGraphAddNode` API，指定 `cudaGraphNodeTypeKernel` 类型，并在 `.kernel` 字段下的 `cudaKernelNodeParamsV2` 结构体中显式指定 `.ctx` 字段。`cudaGraphAddKernelNode` 不允许用户指定 execution context，因此应避免使用它。请注意，一个 graph 中的不同 graph nodes 可能属于不同的 execution context。

出于验证目的，可以使用 Nsight Systems 的 node tracing 模式（`--cuda-graph-trace node`）来观察特定 graph nodes 将在哪些 green context 上执行。请注意，在默认的 graph tracing 模式下，整个 graph 会显示在其启动 stream 的 green context 下，但正如前面所解释的，这并不能提供各个 graph nodes 的 execution context 信息。

要在程序上验证，可以使用 CUDA driver API `cuGraphKernelNodeGetParams(graph_node, &node_params)`，并将 `node_params.ctx` 上下文句柄字段与预期的 graph node 上下文句柄进行比较。由于 `CUgraphNode` 和 `cudaGraphNode_t` 可以互换使用，所以使用 driver API 是可行的，但用户需要包含相应的 `cuda.h` 头文件并直接与 driver 链接（`-lcuda`）。

**Thread Block Clusters**

 具有 thread block clusters 的 kernels（见 Section 1.2.2.1.1）也可以像其他 kernel 一样在 green context stream 上启动，因此可以使用该 green context 提供的资源。Section 4.6.4.2 展示了如何在分割 device resource 时指定需要 coscheduled 的 SM 数量，以支持 clusters。但像使用 clusters 的任何 kernel 一样，用户应该使用相关的 occupancy APIs 来确定 kernel 的最大潜在 cluster 大小（`cudaOccupancyMaxPotentialClusterSize`），如有需要再确定最大活跃 cluster 数量（`cudaOccupancyMaxActiveClusters`）。如果用户在相关的 `cudaLaunchConfig` 的 `stream` 字段中指定了一个 green context stream，那么这些 occupancy APIs 会考虑为该 green context 提供的 SM 资源。这种用例在以下情况尤其相关：比如库可能由用户传入一个 green context CUDA stream，或者 green context 是从剩余的 device resource 创建的。

下面的代码片段展示了这些 API 如何使用。

```c++
    // 假设 cudaStream_t gc_stream 已经被创建，并且存在一个 __global__ void cluster_kernel。

    // 如果可能，取消注释以支持非可移植的 cluster 大小
    // CUDA_CHECK(cudaFuncSetAttribute(cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1))
    cudaLaunchConfig_t config = {0};
    config.gridDim          = grid_dim; // 必须是 cluster 维度的倍数。
    config.blockDim         = block_dim;
    config.dynamicSmemBytes = expected_dynamic_shared_mem;
    
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 1;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.stream=gc_stream; // 需要传入将用于该 kernel 的 CUDA stream
    
    int max_potential_cluster_size = 0;
    // 下一次调用会忽略 launch config 中的 cluster 维度
    CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_potential_cluster_size, cluster_kernel, &config));
    std::cout << "max potential cluster size is " << max_potential_cluster_size << " for CUDA stream gc_stream" << std::endl;
    // 可以选择用 max_potential_cluster_size 更新 launch config 的 clusterDim。
    // 这样做对于相同的 kernel 和 launch config 会使得 cudaLaunchKernelEx 调用成功。
    
    int num_clusters= 0;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&num_clusters, cluster_kernel, &config));
    std::cout << "Potential max. active clusters count is " << num_clusters << std::endl;
```

**Verify Green Contexts Use** 

除了通过观察 kernel 执行时间是否因为 green context 分配而发生变化之外，用户还可以利用开发者工具 Nsight Systems 或 Nsight Compute，在一定程度上验证 green contexts 是否被正确使用。例如，在 Nsight Systems 报告的 CUDA 硬件时间线部分，属于不同 green contexts 的 CUDA streams 上启动的 kernels 会显示在不同的 *Green Context* 行中。在 Nsight Compute 的会话（Session）页面中，会有一个 *Green Context Resources* 总览，此外在详情部分的 *Launch Statistics* 下，还会显示更新后的 SM 数量。前者会以可视化的位掩码形式展示已分配的资源。当应用使用多个 green contexts 时，这一点尤其有用，因为用户可以确认 green contexts 之间是否存在预期的资源重叠（例如没有重叠或根据 SM oversubscription 的期望出现非零重叠）。

图 45 展示了一个示例中两个 green contexts 所分配的资源：分别拥有 112 个和 16 个 SMs，并且它们之间没有 SM 重叠。这个视图可以帮助用户验证每个 green context 被分配的 SM 资源数量是否符合预期。它还能确认没有发生 SM oversubscription，因为在两个 green contexts 中没有任何一个资源格子同时被标记为绿色（表示被分配给该 GC）。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/green_contexts_ncu_mask.png)

*Figure 45*  
*Green contexts resources section from Nsight Compute*

在 *Launch Statistics* 部分还会明确列出分配给某个 green context 的 SM 数，这些 SM 可被这个 kernel 使用。请注意，这里显示的只是该 kernel 在执行期间可以访问的 SM 数量，而不是 kernel 实际运行时使用了多少个 SM。这同样适用于前面显示的资源总览视图。kernel 实际使用的 SM 数量可能受到多种因素影响，包括 kernel 本身的 launch geometry、GPU 上同时运行的其他工作等。

### 4.6.6. Execution Context 的其他辅助 API

这一小节简要介绍了一些额外的 green context 相关 API。完整的列表请参考对应的 CUDA 运行时 API 文档。

对于使用 CUDA 事件（event）进行同步，可以使用 `cudaError_t cudaExecutionCtxRecordEvent(cudaExecutionContext_t ctx, cudaEvent_t event)` 和 `cudaError_t cudaExecutionCtxWaitEvent(cudaExecutionContext_t ctx, cudaEvent_t event)` 这两个 API。`cudaExecutionCtxRecordEvent` 会记录一个 CUDA 事件，并捕获在调用时指定 execution context 内所有正在进行的工作/活动；而 `cudaExecutionCtxWaitEvent` 会让之后提交给该 execution context 的所有工作等待这个事件所捕获的工作完成。

如果一个 execution context 包含多个 CUDA stream，使用 `cudaExecutionCtxRecordEvent` 会比逐个对每个 stream 调用 `cudaEventRecord` 更方便。要在不使用 execution context API 的情况下实现等效行为，需要对 execution context 内的每个 stream 分别记录一个事件，然后分别让依赖的工作等待这些事件。同样地，如果希望 execution context 下的所有 stream 等待某个事件完成，使用 `cudaExecutionCtxWaitEvent` 比对每个 stream 调用 `cudaStreamWaitEvent` 更方便；不使用 execution context API 的替代方案是对每个 stream 分别调用 `cudaStreamWaitEvent`。

在 CPU 端做阻塞式同步，可以使用 `cudaError_t cudaExecutionCtxSynchronize(cudaExecutionContext_t ctx)`。调用它会阻塞当前线程，直到指定的 execution context 完成其中所有工作。如果指定的 execution context 不是通过 `cudaGreenCtxCreate` 创建的，而是通过 `cudaDeviceGetExecutionCtx` 获得的（因此是 GPU 设备的 primary context），那么调用该函数还会同步同一设备上所有已创建的 green context。

要查询某个 execution context 所关联的 device，可以使用 `cudaExecutionCtxGetDevice`；要获取某个 execution context 的唯一标识符，则可以使用 `cudaExecutionCtxGetId`。

最后，通过 `cudaError_t cudaExecutionCtxDestroy(cudaExecutionContext_t ctx)` 这个 API，可以销毁一个明确创建的 execution context。

### 4.6.7. Green Contexts Example

本节展示了 green context 如何让关键任务更早开始并更早完成。和第 4.6.1 节中的场景类似，这个应用包含两个 kernel，分别运行在两个不同的非阻塞 CUDA stream 上。从 CPU 角度来看，时间线如下：首先在 CUDA stream strm1 上启动一个运行时间很长的 kernel（delay_kernel_us），它会在整个 GPU 上经历多轮执行。随后在短暂等待之后（小于该 kernel 的运行时间），在 stream strm2 上启动一个较短但很关键的 kernel（critical_kernel）。文中会测量这两个 kernel 的 GPU 执行时长，以及从 CPU 启动到执行完成的时间。

为了模拟一个长时间运行的 kernel，这里使用了一个 delay kernel，其中每个 thread block 都会运行固定数量的微秒，并且 thread block 的数量超过了 GPU 可用的 SM 数量。

最开始并没有使用 green context，而是将 critical kernel 启动在一个比长时间运行 kernel 优先级更高的 CUDA stream 上。由于 stream 优先级较高，只要长时间运行 kernel 的部分 thread block 执行完成，critical kernel 就可以开始执行。不过，它仍然需要等待一些可能运行时间较长的 thread block 结束，这会推迟它真正开始执行的时间。

图 46 在 Nsight Systems 报告中展示了这一场景。长时间运行的 kernel 启动在 stream 13 上，而较短但关键的 kernel 启动在 stream 14 上，该 stream 具有更高的优先级。正如图中标出的那样，critical kernel 在这个例子中需要等待 0.9ms 才能开始执行。如果这两个 stream 的优先级相同，critical kernel 的执行时间会被推迟得更晚。



![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/green_contexts_nsys_example_no_GCs_with_prio.png)

*Figure 46  
*Nsight Systems timeline without green contexts*

为了利用 green context 功能，这里创建了两个 green context，并为它们分别分配了互不重叠的 SM 集合。在这个示例中，以一块拥有 132 个 SM 的 H100 为例，出于演示目的，给 critical kernel（Green Context 3）分配了 16 个 SM，给长时间运行的 kernel（Green Context 2）分配了 112 个 SM。如图 47 所示，由于存在只有 Green Context 3 才能使用的 SM，critical kernel 现在几乎可以立刻开始执行。

与单独运行时相比，短 kernel 的执行时间可能会变长，因为它现在能够使用的 SM 数量受到了限制。长时间运行的 kernel 也是如此，它不再能够使用 GPU 上的全部 SM，而是受到其 green context 所分配资源的约束。不过，最关键的结果在于，critical kernel 的工作现在可以比之前明显更早开始并完成。当然，这里不考虑其他限制因素，因为正如前面提到的那样，并不能保证一定能够并行执行。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/green_contexts_nsys_example_w_GCs.png)

*Figure 47  
*Nsight Systems timeline with green contexts*

在所有情况下，具体的 SM 划分方式都应该根据实际场景，通过实验来决定。
