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
