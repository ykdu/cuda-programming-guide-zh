# 4.2 CUDA Graphs

CUDA Graphs 提供了 CUDA 中另一种提交工作的模型。一个 graph 是一系列操作（例如 kernel 启动、数据拷贝等），这些操作通过依赖关系连接在一起，并且 graph 的定义与它的执行是分开的。这使得 graph 可以只定义一次，然后被反复启动。将 graph 的定义与执行分离，可以带来多种优化：首先，相比 stream，CPU 的启动开销会更低，因为大量准备工作已经提前完成；其次，把整个工作流程一次性呈现给 CUDA，使其能够进行一些在 stream 这种分段式提交工作机制下无法实现的优化。

要理解 graph 能带来的优化，可以先看看在 stream 中会发生什么：当你把一个 kernel 放入 stream 时，host driver 会执行一系列操作，为该 kernel 在 GPU 上的执行做准备。这些用于设置和启动 kernel 的操作本身就是开销，而且每提交一个 kernel 都要付出一次。对于执行时间很短的 GPU kernel 来说，这些开销可能会占据端到端总执行时间中相当大的一部分。通过创建一个 CUDA graph，把一个会被多次执行的完整工作流程封装进去，这些开销可以在 graph 实例化时为整个 graph 只付出一次，之后 graph 本身就可以以极低的开销被反复启动。

### 4.2.1 Graph结构

一个操作形成图中的一个节点。操作之间的依赖关系就是边。依赖关系约束了操作的执行顺序。

一旦它所依赖的节点完成，该操作可以在任何时间被调度。调度由 CUDA 系统负责。

#### 4.2.1.1 Node类型

图节点可以是以下之一：

- kernel
- CPU 函数调用
- memcpy
- memset
- 空节点
- 等待 CUDA 事件
- 记录 CUDA 事件
- 发信号给外部信号量
- 等待外部信号量
- conditional node
- memory node
- 子图：用于执行一个单独的嵌套图（如下面的图所示）。

粒度或对齐要求导致的，这些要求与具体 GPU 架构有关。 

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/child-graph.png)

*Figure 21*  
*子图举例*

#### 4.2.1.2 Edge数据

CUDA 12.3 在 CUDA Graphs 中引入了 edge data（边数据）。目前，非默认的 edge data 唯一的用途是启用 Programmatic Dependent Launch。

一般来说，edge data 用来修改由一条边所表示的依赖关系，它由三部分组成：outgoing port（输出端口）、incoming port（输入端口）以及 type（类型）。outgoing port 用来指定相关的边在什么时候被触发。incoming port 用来指定节点中哪一部分依赖于这条边。type 则用来修改边两端之间的关系。

端口的取值与节点类型和方向有关，而 edge type 可能只允许用于特定的节点类型。在所有情况下，零初始化的 edge data 都表示默认行为。outgoing port 为 0 表示等待整个任务完成，incoming port 为 0 表示阻塞整个任务，而 edge type 为 0 表示一个带有内存同步语义的完整依赖关系。

在各种 graph API 中，edge data 可以通过一个与相关节点并行的数组来选择性地指定。如果作为输入参数时省略，则会使用零初始化的数据。如果作为输出（查询）参数时省略，只要被忽略的 edge data 全部是零初始化的，API 就会接受；如果这次调用会丢失信息，则会返回 `cudaErrorLossyQuery`。

在一些 stream capture API 中也可以使用 edge data，包括 `cudaStreamBeginCaptureToGraph()`、`cudaStreamGetCaptureInfo()` 和 `cudaStreamUpdateCaptureDependencies()`。在这些情况下，下游节点尚不存在，这些数据会关联到一条悬空的边（dangling edge，也叫 half edge），它要么在之后连接到未来捕获的节点，要么在 stream capture 结束时被丢弃。需要注意的是，有些 edge type 并不会等待上游节点完全执行结束。在判断一次 stream capture 是否已经完全重新并入原始 stream 时，这类边会被忽略，而且它们在 capture 结束时也不能被丢弃。参见 Stream Capture。

没有任何节点类型定义了额外的 incoming port，只有 kernel 节点定义了额外的 outgoing port。存在一种非默认的依赖类型 `cudaGraphDependencyTypeProgrammatic`，用于在两个 kernel 节点之间启用 Programmatic Dependent Launch。

### 4.2.2 构建与运行 Graph

使用 graphs 提交工作被分成三个不同的阶段：定义、实例化 和 执行。

- 在定义或创建阶段，程序构建 graph 的操作描述以及这些操作之间的依赖关系。
- 实例化 会对 graph 模板做快照、验证它，并执行大量设置与初始化工作，从而尽可能减少 launch 时所需的开销。实例化后得到的对象称为 可执行 graph（executable graph）。
- 可执行的 graph 可以像其他 CUDA 工作一样在 stream 中 launch。它可以在不重复实例化的情况下被 launch 任意多次。

#### 4.2.2.1 Graph创建

Graphs 可以通过两种机制创建：显式的 Graph API 和 stream capture。

##### 4.2.2.1.1 Graph APIs

下面的示例（省略了声明和其他样板代码）展示了如何创建下图的 graph。注意使用 `cudaGraphCreate()` 创建 graph，并用 `cudaGraphAddNode()` 添加 kernel 节点及它们之间的依赖关系。所有用于添加节点与依赖的函数都在 **CUDA Runtime API 文档** 中列出。



![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/create-a-graph.png)

*Figure 22*  
*使用Graph APIs创建一张Graph*

```c++
// 创建 graph — 初始为空
cudaGraphCreate(&graph, 0);

// 创建节点及它们的依赖
cudaGraphNode_t nodes[4];
cudaGraphNodeParams kParams = { cudaGraphNodeTypeKernel };
kParams.kernel.func         = (void *)kernelName;
kParams.kernel.gridDim.x    = kParams.kernel.gridDim.y  = kParams.kernel.gridDim.z  = 1;
kParams.kernel.blockDim.x   = kParams.kernel.blockDim.y = kParams.kernel.blockDim.z = 1;
cudaGraphAddNode(&nodes[0], graph, NULL, NULL, 0, &kParams);
cudaGraphAddNode(&nodes[1], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[2], graph, &nodes[0], NULL, 1, &kParams);
cudaGraphAddNode(&nodes[3], graph, &nodes[1], NULL, 2, &kParams);
```

上面的示例展示了四个 kernel 节点及它们之间的依赖，用于说明如何创建一个非常简单的 graph。在典型的用户应用中，还需要添加内存操作等节点，例如 `cudaGraphAddMemcpyNode()` 等。完整的用于添加节点的 Graph API 函数请参见 **CUDA Runtime API 文档**。

##### 4.2.2.1.2 Stream Capture

Stream capture 提供了一种从现有基于 stream 的 API 构建 graph 的机制。可以将一段向 stream 提交工作的代码放在 `cudaStreamBeginCapture()` 和 `cudaStreamEndCapture()` 的调用之间，如下所示：

```
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);
cudaStreamEndCapture(stream, &graph);
```

调用 `cudaStreamBeginCapture()` 会让一个 stream 进入 capture 模式。当一个 stream 正在 capture 时，提交到这个 stream 的工作不会被排入执行队列；相反，这些工作会追加到内部不断构建的 graph 中。随后调用 `cudaStreamEndCapture()` 会返回该 graph，并结束这个 stream 的 capture 模式。这样通过 stream capture 正在构建的 graph 称为 capture graph。

除了 `cudaStreamLegacy`（也就是 “NULL stream”）之外，任何 CUDA stream 都可以进行 stream capture。注意，它也可以用于 `cudaStreamPerThread`。如果程序正在使用 legacy stream，有时可以通过将 stream 0 重新定义为 per-thread stream 来实现等价功能。参见 *Blocking and non-blocking streams and the default stream*。[NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html?utm_source=chatgpt.com)

可以使用 `cudaStreamIsCapturing()` 来查询一个 stream 是否正在 capture。

可以用 `cudaStreamBeginCaptureToGraph()` 将工作捕获到一个已有的 graph 中。这样不是捕获到内部 graph，而是捕获到用户提供的 graph。[NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html?utm_source=chatgpt.com)

###### 4.2.2.1.2.1 Cross-stream Dependencies and Events

Stream capture 可以处理通过 `cudaEventRecord()` 和 `cudaStreamWaitEvent()` 表示的跨流依赖，**前提是** 所等待的 event 是记录到同一个 capture graph 中的。

当一个 event 在处于 capture 模式的 stream 中被记录时，它会产生一个 captured event。这个 captured event 代表了 capture graph 中的一组节点。

当一个 stream 对 captured event 执行 wait 操作时，如果该 stream 尚未处于 capture 模式，它会被置入 capture 模式；接下来该 stream 中的下一个操作会对 captured event 所代表的那些节点产生附加依赖。此时，这两个 stream 都会被 capture 到同一个 capture graph 中。

即使存在跨流依赖，在 stream capture 情况下仍然必须在与 `cudaStreamBeginCapture()` 相同的 stream 上调用 `cudaStreamEndCapture()`；这个 stream 称为 origin stream。由于基于 event 的依赖而被 capture 到同一个 capture graph 的其他所有 streams，也**必须**重新 join 到 origin stream。如下所示。所有被 capture 到同一个 capture graph 的 streams 在执行 `cudaStreamEndCapture()` 时都会退出 capture 模式。如果未能重新 join 回 origin stream，将会导致整个 capture 操作失败。

```C++
// stream1 是 origin stream
cudaStreamBeginCapture(stream1);

kernel_A<<< ..., stream1 >>>(...);

// 分叉到 stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);

// 将 stream2 重新 join 到 origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_D<<< ..., stream1 >>>(...);
// 在 origin stream 结束 capture
cudaStreamEndCapture(stream1, &graph);

// stream1 和 stream2 都退出 capture 模式
```

上述代码返回的 graph 如图 22 所示。

> 注意
>
> 当一个 stream 退出 capture 模式时，该 stream 中下一个非 capture 的操作（如果存在）仍然会依赖于最近一次之前的非 capture 操作，即使中间的操作已经被移除。

###### 4.2.2.1.2.2. 禁止和不支持的操作

试图对正在 capture 的 stream 或被 capture 的 event 进行同步或查询其执行状态是无效的，因为这些对象并不会实际排入执行队列。类似地，当有 stream 正在 capture 时，去查询或同步一个更大的句柄（例如 device 或 context）也是无效的，因为这些句柄包含了正在 capture 的 stream。

当同一 context 中的任意一个 stream 正在 capture，并且该 stream 不是用 `cudaStreamNonBlocking` 创建的，任何尝试使用 legacy stream 的行为都是无效的。这是因为 legacy stream 始终“包含”其它 stream；向 legacy stream 排队会引入对正在 capture 的 stream 的依赖，而查询或同步 legacy stream 就等价于查询或同步那些正在 capture 的 stream。

因此，在这种情况下调用同步 API 也是无效的。一个典型的同步 API 是 `cudaMemcpy()`——它会向 legacy stream 排队工作并在返回前同步该 stream。

> 注意
>
> 一般规则是，当某个依赖关系会将一个已 capture 的项与一个未被 capture 而直接排入执行队列的项连接起来时，CUDA **倾向于返回错误**，而不是忽略这种依赖关系。唯一的例外是在将 stream 置入或退出 capture 模式时；在这种转换过程中，capture mode 前后添加到 stream 的项之间的依赖会被切断。

试图通过等待一个来自当前正在 capture 且属于不同 capture graph 的 stream 的 captured event 来合并两个独立的 capture graph 是无效的。同样，在没有指定 `cudaEventWaitExternal` 标志的情况下等待来自一个正在 capture 的 stream 的非 capture event 也是无效的。

还有一小部分 API 会将异步操作排入 stream，但这些 API **当前在 graphs 中不受支持**，如果在正在 capture 的 stream 上调用它们会返回错误，例如 `cudaStreamAttachMemAsync()`。

###### 4.2.2.1.2.3. 失效

在 stream capture 过程中一旦尝试执行无效操作，所有相关的 capture graph 都会被置为失效状态。一旦 capture graph 失效，任何正在 capture 的 stream，或与该 graph 关联的 captured event，都不允许继续使用，相关操作都会返回错误，直到通过 `cudaStreamEndCapture()` 结束 stream capture 为止。该调用会使相关 stream 退出 capture mode，但同时也会返回一个错误值以及一个 `NULL` graph。

###### 4.2.2.1.2.4. 捕获状态检查

可以使用 `cudaStreamGetCaptureInfo()` 来检查当前正在进行的 stream capture 操作。通过该接口，用户可以获取 capture 的状态、一个在进程内唯一的 capture ID、底层的 graph 对象，以及 stream 中下一个即将被 capture 的 node 所对应的依赖关系和 edge 数据。这些依赖信息可用于获取 stream 中最近一次被 capture 的 node 的句柄。

4.2.2.1.3. Putting It All Together

图 22 中的示例只是一个概念性的小图示例。对于真实应用中使用 CUDA graph 的情况，无论是使用 graph API 还是通过 stream capture，其用法都会更复杂。下面的代码片段展示了使用 Graph API 和 Stream Capture 并排创建一个 CUDA graph 的对比，用于执行一个简单的两阶段 reduction 算法。

图 23 展示了该 CUDA graph 的图形表示，这个图是对下述代码调用 `cudaGraphDebugDotPrint` 得到的结果，并做了适当调整以便阅读，然后用 **Graphviz** 渲染出来的。



![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/cuda_graph_reduction.png)

*Figure 23*  
*两阶段reduction kernel例子*

```c++
// Graph API
void cudaGraphsManual(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
   cudaStream_t                 streamForGraph;
   cudaGraph_t                  graph;
   std::vector<cudaGraphNode_t> nodeDependencies;
   cudaGraphNode_t              memcpyNode, kernelNode, memsetNode;
   double                       result_h = 0.0;

   cudaStreamCreate(&streamForGraph));

   cudaKernelNodeParams kernelNodeParams = {0};
   cudaMemcpy3DParms    memcpyParams     = {0};
   cudaMemsetParams     memsetParams     = {0};

   memcpyParams.srcArray = NULL;
   memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
   memcpyParams.srcPtr   = make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
   memcpyParams.dstArray = NULL;
   memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
   memcpyParams.dstPtr   = make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
   memcpyParams.extent   = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
   memcpyParams.kind     = cudaMemcpyHostToDevice;

   memsetParams.dst         = (void *)outputVec_d;
   memsetParams.value       = 0;
   memsetParams.pitch       = 0;
   memsetParams.elementSize = sizeof(float); // elementSize can be max 4 bytes
   memsetParams.width       = numOfBlocks * 2;
   memsetParams.height      = 1;

   cudaGraphCreate(&graph, 0);
   cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams);
   cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

   nodeDependencies.push_back(memsetNode);
   nodeDependencies.push_back(memcpyNode);

   void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize, &numOfBlocks};

   kernelNodeParams.func           = (void *)reduce;
   kernelNodeParams.gridDim        = dim3(numOfBlocks, 1, 1);
   kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
   kernelNodeParams.sharedMemBytes = 0;
   kernelNodeParams.kernelParams   = (void **)kernelArgs;
   kernelNodeParams.extra          = NULL;

   cudaGraphAddKernelNode(
      &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

   nodeDependencies.clear();
   nodeDependencies.push_back(kernelNode);

   memset(&memsetParams, 0, sizeof(memsetParams));
   memsetParams.dst         = result_d;
   memsetParams.value       = 0;
   memsetParams.elementSize = sizeof(float);
   memsetParams.width       = 2;
   memsetParams.height      = 1;
   cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams);

   nodeDependencies.push_back(memsetNode);

   memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
   kernelNodeParams.func           = (void *)reduceFinal;
   kernelNodeParams.gridDim        = dim3(1, 1, 1);
   kernelNodeParams.blockDim       = dim3(THREADS_PER_BLOCK, 1, 1);
   kernelNodeParams.sharedMemBytes = 0;
   void *kernelArgs2[3]            = {(void *)&outputVec_d, (void *)&result_d, &numOfBlocks};
   kernelNodeParams.kernelParams   = kernelArgs2;
   kernelNodeParams.extra          = NULL;

   cudaGraphAddKernelNode(
      &kernelNode, graph, nodeDependencies.data(), nodeDependencies.size(), &kernelNodeParams);

   nodeDependencies.clear();
   nodeDependencies.push_back(kernelNode);

   memset(&memcpyParams, 0, sizeof(memcpyParams));

   memcpyParams.srcArray = NULL;
   memcpyParams.srcPos   = make_cudaPos(0, 0, 0);
   memcpyParams.srcPtr   = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
   memcpyParams.dstArray = NULL;
   memcpyParams.dstPos   = make_cudaPos(0, 0, 0);
   memcpyParams.dstPtr   = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
   memcpyParams.extent   = make_cudaExtent(sizeof(double), 1, 1);
   memcpyParams.kind     = cudaMemcpyDeviceToHost;

   cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(), nodeDependencies.size(), &memcpyParams);
   nodeDependencies.clear();
   nodeDependencies.push_back(memcpyNode);

   cudaGraphNode_t    hostNode;
   cudaHostNodeParams hostParams = {0};
   hostParams.fn                 = myHostNodeCallback;
   callBackData_t hostFnData;
   hostFnData.data     = &result_h;
   hostFnData.fn_name  = "cudaGraphsManual";
   hostParams.userData = &hostFnData;

   cudaGraphAddHostNode(&hostNode, graph, nodeDependencies.data(), nodeDependencies.size(), &hostParams);
}
```

```c++
// Stream Capture
void cudaGraphsUsingStreamCapture(float  *inputVec_h,
                      float  *inputVec_d,
                      double *outputVec_d,
                      double *result_d,
                      size_t  inputSize,
                      size_t  numOfBlocks)
{
   cudaStream_t stream1, stream2, stream3, streamForGraph;
   cudaEvent_t  forkStreamEvent, memsetEvent1, memsetEvent2;
   cudaGraph_t  graph;
   double       result_h = 0.0;

   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   cudaStreamCreate(&stream3);
   cudaStreamCreate(&streamForGraph);

   cudaEventCreate(&forkStreamEvent);
   cudaEventCreate(&memsetEvent1);
   cudaEventCreate(&memsetEvent2);

   cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

   cudaEventRecord(forkStreamEvent, stream1);
   cudaStreamWaitEvent(stream2, forkStreamEvent, 0);
   cudaStreamWaitEvent(stream3, forkStreamEvent, 0);

   cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * inputSize, cudaMemcpyDefault, stream1);

   cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2);

   cudaEventRecord(memsetEvent1, stream2);

   cudaMemsetAsync(result_d, 0, sizeof(double), stream3);
   cudaEventRecord(memsetEvent2, stream3);

   cudaStreamWaitEvent(stream1, memsetEvent1, 0);

   reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);

   cudaStreamWaitEvent(stream1, memsetEvent2, 0);

   reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d, numOfBlocks);
   cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream1);

   callBackData_t hostFnData = {0};
   hostFnData.data           = &result_h;
   hostFnData.fn_name        = "cudaGraphsUsingStreamCapture";
   cudaHostFn_t fn           = myHostNodeCallback;
   cudaLaunchHostFunc(stream1, fn, &hostFnData);
   cudaStreamEndCapture(stream1, &graph);
}
```

#### 4.2.2.2. Graph Instantiation

一旦 graph 已经被创建（无论是通过 Graph API 还是通过 stream capture），就需要对该 graph 进行 instantiate，以便创建一个 executable graph，之后才能 launch。假设 `cudaGraph_t graph` 已成功创建，下面的代码将对该 graph 进行 instantiation，并创建 executable graph `cudaGraphExec_t graphExec`：

```
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);  // 实例化 graph
```

当 graph 已创建并且成功实例化成 executable graph 之后，就可以 launch 了。假设 `cudaGraphExec_t graphExec` 已成功创建，下面的代码片段会将该 graph launch 到指定的 stream 中：

```
cudaGraphLaunch(graphExec, stream);  // 将 executable graph launch 到 stream
```

下面是将上述流程综合起来的示例，基于第 4.2.2.1.2 节中的 stream capture 例子，该代码将创建 graph、实例化它，并 launch：

```
cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

### 4.2.3 更新Instantiated Graph

当工作流程发生变化时，graph 就会失效并必须修改。对 graph 结构的重大更改（例如拓扑结构或节点类型）需要重新实例化，因为与拓扑相关的优化必须重新应用。然而，更常见的情况是只有节点的参数（比如 kernel 参数或内存地址）发生了变化，而 graph 的拓扑保持不变。针对这种情况，CUDA 提供了一种轻量的 “Graph Update” 机制，允许就地修改某些节点参数，而无需重建整个 graph，这比重新实例化要高效得多。

这些更新将在下一次 graph launch 时生效，因此它们不会影响之前已 launch 的 graph，即使这些 launch 正在运行。一个 graph 可以重复更新并重新 launch，因此可以在同一个 stream 上排队多个更新/launch。

CUDA 提供了两种机制来更新已实例化的 graph 参数：整体 graph 更新（whole graph update）和单独节点更新（individual node update）。整体 graph 更新允许用户提供一个拓扑完全相同的 `cudaGraph_t` 对象，该对象的节点包含已更新的参数。单独节点更新则允许用户显式地更新具体节点的参数。当需要更新的节点很多、或者调用方不知道 graph 拓扑（比如 graph 是通过 stream capture 从库调用生成的）时，使用更新后的 `cudaGraph_t` 更方便；当需要更新的节点很少且用户持有这些节点的句柄时，使用单独节点更新更为理想，因为它跳过了对未变更节点的拓扑检查和比较，从而在许多情况下更高效。

CUDA 还提供了启用/禁用单个节点的机制而不影响它当前的参数。

下面的各小节将更详细地解释每种方法。

#### **4.2.3.1. Whole Graph Update**

 `cudaGraphExecUpdate()` 允许用一个拓扑结构完全相同的图（称为“更新图”）中的参数去更新一个已经实例化过的图（称为“原始图”）。用于实例化 `cudaGraphExec_t` 的原始图与更新图在拓扑结构上必须 **完全一致**，而且它们依赖关系的指定顺序也必须匹配。最后，CUDA 还需要对 **sink nodes**（即在最终图中**没有依赖其它节点 / 没有 outgoing edge 的节点**）保持一致的顺序。CUDA 依赖特定 API 调用的顺序来实现一致的 sink node 排序。

更明确地说，遵循以下规则会使 `cudaGraphExecUpdate()` **确定性地** 将原始图和更新图的节点进行配对：

1. 对于任何被 capture 的 stream，在该 stream 上执行的 API 调用必须以**相同的顺序** 进行，包括那些不直接创建节点但会等待 event 或其他行为的 API 调用。
2. 所有直接操作给定图节点的 incoming edge（包括被 capture 的 stream API、节点添加 API 和边添加/删除 API）的调用必须以**相同的顺序** 进行。此外，当这些依赖关系通过数组传递给这些 API 时，数组内部依赖项的顺序也必须匹配。
3. sink nodes 必须保持一致的顺序。在调用 `cudaGraphExecUpdate()` 时，sink nodes 是那些在最终图中**没有 dependent node / 没有 outgoing edge 的节点**。以下这些操作（如果存在）会影响 sink node 的顺序，并且作为一个整体 **必须以相同顺序执行**：
   - 通过节点添加 API 添加出的 sink node。
   - 通过删除边使某个节点成为 sink node。
   - 当 `cudaStreamUpdateCaptureDependencies()` 从 capture stream 的依赖集中移除一个 sink node 时。
   - `cudaStreamEndCapture()`。

下面的示例展示了如何使用该 API 去更新一个已实例化的 graph：

```c++
cudaGraphExec_t graphExec = NULL;

for (int i = 0; i < 10; i++) {
    cudaGraph_t graph;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;
    // 在这个例子中我们使用 stream capture 来创建 graph。
    // 你也可以使用 Graph API 去生成 graph。
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // 调用一个用户定义的基于 stream 的工作负载，例如
    do_cuda_work(stream);

    cudaStreamEndCapture(stream, &graph);
    // 如果我们已经实例化过 graph，尝试直接更新它
    // 以避免再次实例化的开销
    if (graphExec != NULL) {
        // 如果图更新失败，errorNode 会设置成导致失败的节点，
        // 而 updateResult 会设置成失败原因码。
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    }
    // 在第一次迭代或者任何更新失败的时候进行实例化
    if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {
        // 如果之前的更新失败，在重新实例化前销毁 cudaGraphExec_t
        if (graphExec != NULL) {
            cudaGraphExecDestroy(graphExec);
        }
        // 使用 graph 实例化 graphExec。
        // 这里不需要使用 error node 和失败信息参数。
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphDestroy(graph);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}
```

一个典型的工作流程是，先通过 stream capture 或 graph API 创建初始的 `cudaGraph_t`，然后像往常一样对该 `cudaGraph_t` 进行实例化并启动。在首次 launch 之后，会使用与初始 graph 相同的方法创建一个新的 `cudaGraph_t`，并调用 `cudaGraphExecUpdate()`。如果图更新成功（如上例中通过 `updateResult` 参数所示），则启动更新后的 `cudaGraphExec_t`。如果由于任何原因更新失败，则需要调用 `cudaGraphExecDestroy()` 和 `cudaGraphInstantiate()`，销毁原有的 `cudaGraphExec_t` 并重新实例化一个新的。

也可以直接更新 `cudaGraph_t` 中的节点（例如使用 `cudaGraphKernelNodeSetParams()`），然后再更新对应的 `cudaGraphExec_t`；不过，相比之下，使用下一节中介绍的**显式节点更新 API** 会更加高效。

conditional handle 的标志位和默认值也会作为 graph 更新过程的一部分被一并更新。

关于具体用法以及当前的限制，请参考 Graph API。

#### 4.2.3.2. 单个节点更新

已经实例化的 graph 节点参数可以直接更新，这样可以避免重新实例化的开销，也避免了创建新的 `cudaGraph_t` 的开销。如果需要更新的节点数量相对于 graph 中的节点总数较少，那么逐个更新节点会是更好的选择。下面列出了可用于更新 `cudaGraphExec_t` 节点的方法：

​                                                           *表 8 逐节点更新（Individual Node Update）API*

| API                                                    | 节点类型               |
| ------------------------------------------------------ | ---------------------- |
| `cudaGraphExecKernelNodeSetParams()`                   | Kernel 节点            |
| `cudaGraphExecMemcpyNodeSetParams()`                   | 内存复制（memcpy）节点 |
| `cudaGraphExecMemsetNodeSetParams()`                   | 内存填充（memset）节点 |
| `cudaGraphExecHostNodeSetParams()`                     | Host 节点              |
| `cudaGraphExecChildGraphNodeSetParams()`               | 子 graph 节点          |
| `cudaGraphExecEventRecordNodeSetEvent()`               | 事件记录节点           |
| `cudaGraphExecEventWaitNodeSetEvent()`                 | 事件等待节点           |
| `cudaGraphExecExternalSemaphoresSignalNodeSetParams()` | 外部信号量 signal 节点 |
| `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`   | 外部信号量 wait 节点   |

有关这些 API 的详细使用方法和当前限制，请参阅 Graph API 文档。

#### 4.2.3.3. 单个节点Enable

在已实例化的 graph 中，kernel 节点、memset 节点和 memcpy 节点能够通过 `cudaGraphNodeSetEnabled()` API 被启用或禁用。这样可以创建一个包含所有可能功能的 graph，然后针对每次 launch 自定义要启用的部分。节点的启用状态可以通过 `cudaGraphNodeGetEnabled()` API 查询。

被禁用的节点在功能上等同于 empty node（空节点），直到它再次被启用。节点的参数不会因为启用/禁用状态而发生改变。启用状态不会受到单独节点更新（individual node update）或通过 `cudaGraphExecUpdate()` 进行的整体 graph 更新影响。如果在节点禁用期间更新了参数，这些更新会在节点重新启用后生效。

更多关于使用方法和局限性的信息，请参考Graph API文档

#### 4.2.3.4 图更新限制

Kernel 节点：

- 函数所属的 context 不能改变。
- 原来不使用 CUDA dynamic parallelism 的节点，不可更新为使用 CUDA dynamic parallelism 的函数。

`cudaMemset` 和 `cudaMemcpy` 节点：

- 操作数分配/映射到的 CUDA 设备 不能改变。
- 源/目标内存必须从与原始源/目标内存 同一个 context 分配。
- 只能更新一维（1D）的 `cudaMemset`/`cudaMemcpy` 节点。

额外的 memcpy 节点限制：

- 不支持更改源或目标的内存类型（例如 `cudaPitchedPtr`、`cudaArray_t` 等）或传输类型（如 `cudaMemcpyKind`）。

外部信号量等待（external semaphore wait）节点和记录（record）节点：

- 不支持更改信号量数量。

条件（Conditional）节点：

- handle 的创建和赋值顺序在两个图之间 必须匹配。
- 不支持更改节点参数（例如条件中包含的图数量、节点 context 等）。
- 条件主体图（body graph）内部节点的参数更改仍需遵循上述规则。

内存节点（Memory nodes）：

- 如果一个 `cudaGraph_t` 当前已被实例化为另一个 `cudaGraphExec_t`，则 不能用这个 `cudaGraph_t` 来更新另一个 `cudaGraphExec_t`。

没有限制的情况：

- 对 host 节点、事件记录（event record）节点、或事件等待（event wait）节点的更新 没有限制。

### 4.2.4. 条件图节点（Conditional Graph Nodes）

条件节点允许对包含在该节点内的 graph 进行条件执行和循环执行。这样，就可以在一个 graph 内完整地表示动态或迭代的工作流，并释放主机 CPU 去并行执行其它工作。

当条件节点的所有依赖满足时，会在 device 上评估条件的值。条件节点可以是以下几种类型：

* **Conditional IF** 节点在节点执行时若条件值非零，则执行其 body graph 一次。可选的第二个 body graph 会在条件值为零时执行一次。
* **Conditional WHILE** 节点在节点执行时若条件值非零，则执行其 body graph，且在条件值不为零时会重复执行该 body graph，直到条件值变为零。
* **Conditional SWITCH** 节点会在条件值等于 n 时执行编号为 n（从零开始）的 body graph 一次；若条件值不对应任何 body graph，则不会 launch 任何 body graph。

条件值由一个 conditional handle 访问，该 handle 必须在节点之前创建。可以通过 device 代码调用 `cudaGraphSetConditional()` 来设置条件值。创建 handle 时也可以指定一个默认值，在每次 graph launch 时应用该默认值。

当创建条件节点时，会同步创建一个空的 graph，并将 handle 返回给用户以便填充该 graph。这个 conditional body graph 可以通过 graph APIs 或 `cudaStreamBeginCaptureToGraph()` 来填充。

条件节点可以嵌套使用。

#### 4.2.4.1. 条件句柄（Conditional Handles）

条件值由 `cudaGraphConditionalHandle` 表示，通过调用 `cudaGraphConditionalHandleCreate()` 创建该 handle。

一个 handle 必须关联到单个条件节点。handles 不可销毁，因此不需要显式去管理它们。

如果在创建 handle 时指定了 `cudaGraphCondAssignDefault`，则每次 graph 执行开始时条件值会被初始化为指定的默认值；如果未指定此标志，则每次 graph 执行开始时条件值未定义，代码不应假定条件值会跨执行保持不变。

在进行 whole graph update 时，handle 的默认值和相关 flags 也会被更新。

#### 4.2.4.2. 条件节点的 body graph 要求（Conditional Node Body Graph Requirements）

一般要求：

* body graph 中的所有 nodes 必须驻留在同一个 device 上。
* graph 中只能包含 kernel nodes、empty nodes、memcpy nodes、memset nodes、child graph nodes 和 conditional nodes。

对于 kernel nodes：

* graph 中的 kernel 不允许使用 CUDA Dynamic Parallelism 或 Device Graph Launch。
* 若没有启用 MPS，则允许使用 cooperative launches。

对于 memcpy/memset nodes：

* 只允许涉及 device memory 和/或 pinned device-mapped host memory 的复制/填充。
* 不允许涉及 CUDA arrays 的复制/填充。
* 在 instantiation 时，两端的内存必须可由当前 device 访问。注意，即使目标内存在其他设备上，复制操作也会从 graph 所在的 device 执行。

#### 4.2.4.3. 条件IF节点（Conditional IF Nodes）

如果 IF 节点被执行时其条件值非零，那么该 IF 节点包含的 body graph 会被执行一次。下面的示意图展示了一个由 3 个节点组成的图，其中中间的节点 B 是一个 conditional 节点。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/conditional-if-node.png)

*Figure 24*  
*条件IF节点*

以下代码展示了如何创建一个含有 IF conditional 节点的 graph。条件的默认值是通过前面的 kernel 在图中设置的，IF 节点体（body graph）则使用 graph API 进行填充。

```c++
__global__ void setHandle(cudaGraphConditionalHandle handle, int value)
{
    ...
    // 将条件值设置为传入 kernel 的 value
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // 创建 graph
    cudaGraphCreate(&graph, 0);

    // 创建 conditional handle；由于未提供默认值，因此每次 graph 执行开始时条件值未定义
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // 使用前面的 kernel 为 handle 设置条件值
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // 创建并添加 conditional 节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 1; // 只有一个 IF body graph
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // 获取 conditional 节点的 body graph
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // 填充 IF conditional 节点的 body graph
    ...
    cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);

    // 实例化并 launch graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // 清理
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

IF 节点还可以带一个可选的第二个 body graph，当该节点被执行且条件值为零时，这个 else body graph 会被执行一次。

```c++
void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // 创建 graph
    cudaGraphCreate(&graph, 0);

    // 创建 conditional handle；由于未提供默认值，因此每次 graph 执行开始时条件值未定义
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // 使用 upstream kernel 设置 handle 的条件值
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // 创建并添加 IF conditional 节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 2; // 有 if 和 else 两个 body graph
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // 获取 IF conditional 节点的 body graphs
    cudaGraph_t ifBodyGraph = cParams.conditional.phGraph_out[0];
    cudaGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1];

    // 填充 IF 和 ELSE 的 body graph
    ...
    cudaGraphAddNode(&node, ifBodyGraph, NULL, 0, &params);
    ...
    cudaGraphAddNode(&node, elseBodyGraph, NULL, 0, &params);

    // 实例化并 launch graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // 清理
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

#### 4.2.4.4. 条件WHILE节点（Conditional WHILE Nodes）

当 WHILE 节点的条件值非零时，该 WHILE 节点的 body graph 将会被执行。条件值会在这个节点被执行时以及每次 body graph 执行完成之后进行评估。下图展示了一个包含 3 个节点的 graph，其中中间的节点 B 是一个 conditional 节点：

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/conditional-while-node.png)

*Figure 25  
*条件WHILE节点*

下面的代码示例说明了如何构建一个包含 WHILE conditional node 的 graph。这里使用 `cudaGraphCondAssignDefault` 来创建 handle，因此无需在它之前放置一个 kernel 来设置默认条件值。conditional node 的 body graph 是通过 graph API 填充的：

```c++
__global__ void loopKernel(cudaGraphConditionalHandle handle, char *dPtr)
{
   // 将 dPtr 指向的值递减，当值变为 0 时将条件句柄设置为 0
   if (--(*dPtr) == 0) {
      cudaGraphSetConditional(handle, 0);
   }
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    // 分配 1 字节 device 内存作为输入
    char *dPtr;
    cudaMalloc((void **)&dPtr, 1);

    // 创建 graph
    cudaGraphCreate(&graph, 0);

    // 使用默认值 1 创建条件句柄
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);
    // 创建并添加 WHILE 条件节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;
    cudaGraphAddNode(&node, graph, NULL, 0, &cParams);

    // 获取条件节点的主体 graph
    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];
    // 填充条件节点的主体 graph
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)loopKernel;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &dPtr;
    cudaGraphAddNode(&node, bodyGraph, NULL, 0, &params);
    // 初始化 device 内存、实例化并 launch graph
    cudaMemset(dPtr, 10, 1); // 将 dPtr 值设为 10；循环直到 dPtr 为 0
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // 清理
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(dPtr);
}
```

#### 4.2.4.5. 条件WHILE节点（Conditional SWITCH Nodes）

对于一个 SWITCH 节点，如果节点执行时条件值等于 n，那么此 SWITCH 节点的第 n（从零开始计数）个 body 图将被执行一次。下图展示了一个包含 3 个节点的图，其中中间的节点 B 是一个条件节点：

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/conditional-switch-node.png)

*Figure 26  
*条件SWITCH节点*

下面的代码示例说明了如何创建一个包含 SWITCH 条件节点的 graph。条件的值由前面的 kernel 设置。各个条件 body 使用 graph API 进行填充。

```c++
__global__ void setHandle(cudaGraphConditionalHandle handle, int value)
{
    ...
    // 将条件值设置为传入 kernel 的 value
    cudaGraphSetConditional(handle, value);
    ...
}
void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[2];
    int value = 1;

    // 创建 graph
    cudaGraphCreate(&graph, 0);

    // 创建 conditional handle；
    // 因为未提供默认值，所以每次 graph 执行开始时条件值未定义
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // 通过 conditional 前面的 kernel 设置 handle 值
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    kernelArgs[1] = &value;
    cudaGraphAddNode(&node, graph, NULL, 0, &params);

    // 创建并添加 conditional SWITCH 节点
    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeSwitch;
    cParams.conditional.size   = 5;
    cudaGraphAddNode(&node, graph, &node, 1, &cParams);

    // 获取这个 conditional 节点的 body 图数组
    cudaGraph_t *bodyGraphs = cParams.conditional.phGraph_out;

    // 使用 graph API 填充各个 body 图
    ...
    cudaGraphAddNode(&node, bodyGraphs[0], NULL, 0, &params);
    ...
    cudaGraphAddNode(&node, bodyGraphs[4], NULL, 0, &params);

    // 实例化并 launch graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    // 清理
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}
```

### 4.2.5. Graph Memory Nodes

#### 4.2.5.1. Introduction

Graph memory nodes 允许 graph 创建和拥有内存分配。Graph memory nodes 具有 **GPU ordered lifetime semantics（GPU 有序生命周期语义）**，它决定了何时可以在 device 上访问内存。这种 GPU 有序生命周期语义使得驱动可管理内存重用，并且与 stream 有序分配 API（比如 `cudaMallocAsync` 和 `cudaFreeAsync`）的行为一致，这些 API 可以在创建 graph 时被 capture。

Graph 分配（graph allocations）在整个 graph 生命周期（包括重复的 instantiation 和 launch）中拥有固定地址。这使得在 graph 内的其他操作可以直接引用这些内存，而无需更新 graph，即便 CUDA 更改了实际的 backing physical memory。

在同一个 graph 内，如果两个分配的 graph ordered lifetimes 不重叠，它们可以使用相同的底层物理内存。根据 GPU ordered lifetime semantics，CUDA 甚至可以在多个不同的 graphs 之间重用同一块物理内存，通过虚拟地址 aliasing 实现。例如，当不同的 graphs 被 launch 到同一个 stream 时，CUDA 可能会将同一块物理内存虚拟 alias 给具有单个 graph 生命周期的分配。

#### 4.2.5.2. API Fundamentals

图内存节点（graph memory nodes）是表示 **内存分配** 或 **内存释放** 操作的 graph 节点。简而言之，进行内存分配的节点称为 **allocation node（分配节点）**，进行内存释放的节点称为 **free node（释放节点）**。通过 allocation node 创建的分配称为 **graph allocation（图内分配）**。CUDA 在创建节点时为图内分配分配虚拟地址。虽然这些虚拟地址在该分配节点生命周期内保持固定，但分配内容在释放操作之后不会持久保存，并可能被指向其他分配的访问覆盖。

图内分配在每次 graph 运行时被视为重新创建。一个 graph allocation 的生命周期（不同于节点本身的生命周期）从 GPU 执行到达对应的 **内存分配 graph 节点时开始**，并在以下任一事件发生时结束：

- GPU 执行到达对应的释放 graph 节点
- GPU 执行到达用 `cudaFreeAsync()` 释放的 stream 调用
- 立即在 `cudaFree()` 释放调用时结束

> 注意
>
> 图对象被销毁并不会自动释放任何仍然存活的图内分配内存，即便这结束了对应节点的生命周期。必须通过另一个 graph 或使用 `cudaFreeAsync()` / `cudaFree()` 显式释放该分配。

与其他 graph 结构一样，图内存节点在 graph 内按依赖边进行排序。程序必须保证 **访问图内存的操作：**

- 排序在分配节点之后
- 排序在释放节点之前

图内分配的生命周期通常根据 **GPU 执行顺序（GPU ordered）** 开始和结束，而不是 API 调用顺序。GPU ordered 指的是工作在 GPU 上实际执行的顺序，而不是描述或入队顺序。因此，这些分配被视为 **“GPU ordered（按 GPU 顺序）”**。

##### 4.2.5.2.1. Graph Node APIs

图内存节点（Graph memory nodes）可以通过节点创建 API `cudaGraphAddNode` 显式创建。添加一个类型为 `cudaGraphNodeTypeMemAlloc` 的节点时分配的地址，会通过传入的 `cudaGraphNodeParams` 结构体中的 `alloc::dptr` 字段返回给用户。在 allocating graph 内部，所有使用该图分配内存的操作都必须在 allocation 节点之后按依赖关系排序执行。同样，任何 free（释放）节点必须在图中所有使用该分配的操作之后按依赖关系排序执行。Free 节点（释放节点）也是使用 `cudaGraphAddNode` 创建的，节点类型为 `cudaGraphNodeTypeMemFree`。

下图展示了一个包含 allocation 节点和 free 节点的示例图。Kernel 节点 a、b 和 c 都在 allocation 节点之后、free 节点之前按依赖关系排序，因此这些 kernel 可以访问该分配的内存。Kernel 节点 e 没有在 allocation 节点之后排序，因此不能安全访问这块内存。而 kernel 节点 d 没有在 free 节点之前排序，因此也不能安全访问这块内存。



![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/kernel-nodes.png)

*Figure 27  *Kernel节点*

下面的代码片段构造了图中所示的结构：

```c++
// 创建图 —— 初始为空
cudaGraphCreate(&graph, 0);

// 基本内存分配的参数
cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
// 指定 device 0 作为内存所在设备
params.alloc.poolProps.location.id = 0;
params.alloc.bytesize = size;

cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &params);

// 将 kernel 节点加入图
cudaGraphAddNode(&a, graph, &allocNode, 1, NULL, &nodeParams);
cudaGraphAddNode(&b, graph, &a, 1, NULL, &nodeParams);
cudaGraphAddNode(&c, graph, &a, 1, NULL, &nodeParams);
cudaGraphNode_t dependencies[2];
// kernel 节点 b 和 c 使用了图分配，因此释放节点必须依赖它们。
// 由于 b 对 a 的依赖已建立了间接关系，free 节点无需显式依赖 a。
dependencies[0] = b;
dependencies[1] = c;
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = params.alloc.dptr;
cudaGraphAddNode(&freeNode, graph, dependencies, NULL, 2, freeNodeParams);
// free 节点不依赖 kernel 节点 d，因此 d 不能访问已释放的图内存。
cudaGraphAddNode(&d, graph, &c, NULL, 1, &nodeParams);

// 节点 e 不依赖分配节点，因此不能访问该内存。
// 即便 freeNode 依赖 e，这个结论仍然成立。
cudaGraphAddNode(&e, graph, NULL, NULL, 0, &nodeParams);
```

##### 4.2.5.2.2. Stream Capture

图内存节点可以通过捕获对应的按 stream 顺序的分配和释放调用 `cudaMallocAsync` 和 `cudaFreeAsync` 来创建。在这种情况下，被捕获的分配 API 返回的虚拟地址可以被图内的其他操作使用。由于按 stream 顺序的依赖项会被捕获到图中，按 stream 顺序的分配 API 的顺序要求就能保证图内存节点相对于被捕获的 stream 操作有正确的顺序（对于编写正确的 stream 代码而言）。

为清晰起见，下面的代码片段展示了如何使用 stream capture 从前面的示例图创建图，忽略掉d和e kernel结点：

```c++
    cudaMallocAsync(&dptr, size, stream1);
    kernel_A<<< ..., stream1 >>>(dptr, ...);

    // 分支到 stream2
    cudaEventRecord(event1, stream1);
    cudaStreamWaitEvent(stream2, event1);
    kernel_B<<< ..., stream1 >>>(dptr, ...);
    // 事件依赖会转化为图依赖，因此通过捕获 kernel C 生成的 kernel 节点会依赖于捕获 cudaMallocAsync 调用创建的分配节点。

    kernel_C<<< ..., stream2 >>>(dptr, ...);

    // 将 stream2 重新合并回起始 stream（stream1）
    cudaEventRecord(event2, stream2);
    cudaStreamWaitEvent(stream1, event2);

    // 释放依赖于所有访问该内存的工作。
    cudaFreeAsync(dptr, stream1);
    // 在起始 stream 中结束捕获
    cudaStreamEndCapture(stream1, &graph);
```

##### 4.2.5.2.3. 在分配图之外访问和释放图内存（Accessing and Freeing Graph Memory Outside of the Allocating Graph）

图分配（graph allocation）**不必须由创建它的图来释放**。当某个图没有释放一个 allocation 时，该 allocation 会在图执行结束之后继续存在，并且可以被之后的 CUDA 操作访问。只要后续访问操作 **通过 CUDA 事件（events）或其他 stream 顺序机制** 保证在 allocation 之后，则这些 allocation 可以在另一个图中访问，也可以直接通过 stream 操作访问。随后，这个 allocation 可以通过常规的 `cudaFree`、`cudaFreeAsync` 调用释放，或者由另一个带有相应 free 节点的图的 launch 来释放，或者由原始创建该 allocation 的图的后续 launch 释放（如果这个图在实例化时使用了 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志）。**在内存被释放之后访问该内存是非法的** —— 释放操作必须在使用图依赖（graph dependencies）、CUDA 事件和其他 stream 顺序机制确保所有访问都完成之后才发生。

> 注意
>
> 由于 graph 分配可能会共享底层物理内存，因此 free 操作必须在所有设备操作完成之后才有序执行。在内存写操作和 free 操作之间，带外同步（例如 compute kernel 内部基于内存的同步）**不足以**保证顺序性。有关一致性和连贯性的更多信息，请参见虚拟别名支持（Virtual Aliasing Support）规则。

下面的三段代码示例展示了在分配图之外访问 graph allocations 时如何通过如下方式建立正确的顺序：使用单个 stream、在多个 stream 之间使用事件、以及在分配与释放 graph 中嵌入事件。

首先，通过使用单个 stream 建立顺序：

```c++
// 分配 graph 的内容
void *dptr;
cudaGraphNodeParams params = { cudaGraphNodeTypeMemAlloc };
params.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
params.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
params.alloc.bytesize = size;
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &params);
dptr = params.alloc.dptr;

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, stream);
kernel<<< ..., stream >>>(dptr, ...);
cudaFreeAsync(dptr, stream);
```

其次，通过记录和等待 CUDA events 来建立顺序：

```c++
// 分配 graph 的内容
void *dptr;

// 分配 graph 的内容
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
dptr = allocNodeParams.alloc.dptr;

// 使用/释放 graph 的内容
kernelNodeParams.kernel.kernelParams[0] = allocNodeParams.alloc.dptr;
cudaGraphAddNode(&freeNode, freeGraph, NULL, NULL, 1, dptr);

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);
cudaGraphInstantiate(&freeGraphExec, freeGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, allocStream);

// 建立 stream2 对分配节点的依赖关系
// 注意：也可以通过 stream synchronize 操作来建立该依赖
cudaEventRecord(allocEvent, allocStream);
cudaStreamWaitEvent(stream2, allocEvent);

kernel<<< ..., stream2 >>> (dptr, ...);

// 建立 stream3 与分配使用之间的依赖关系
cudaStreamRecordEvent(streamUseDoneEvent, stream2);
cudaStreamWaitEvent(stream3, streamUseDoneEvent);

// 此时可以安全地启动释放 graph，该 graph 也可能会访问该内存
cudaGraphLaunch(freeGraphExec, stream3);
```

第三，通过使用 graph 的外部 event 节点来建立顺序：

```c++
// 分配 graph 的内容
void *dptr;
cudaEvent_t allocEvent; // 表示分配何时可以被使用的 event
cudaEvent_t streamUseDoneEvent; // 表示 stream 对该分配的使用已完成的 event

// 带有 event record 节点的分配 graph 内容
cudaGraphAddNode(&allocNode, allocGraph, NULL, NULL, 0, &allocNodeParams);
dptr = allocNodeParams.alloc.dptr;
// 注意：该 event record 节点依赖于 alloc 节点

cudaGraphNodeParams allocEventNodeParams = { cudaGraphNodeTypeEventRecord };
allocEventParams.eventRecord.event = allocEvent;
cudaGraphAddNode(&recordNode, allocGraph, &allocNode, NULL, 1, allocEventNodeParams);
cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

// 带有 event wait 节点的使用/释放 graph 内容
cudaGraphNodeParams streamWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
streamWaitEventNodeParams.eventWait.event = streamUseDoneEvent;
cudaGraphAddNode(&streamUseDoneEventNode, waitAndFreeGraph, NULL, NULL, 0, streamWaitEventNodeParams);

cudaGraphNodeParams allocWaitEventNodeParams = { cudaGraphNodeTypeEventWait };
allocWaitEventNodeParams.eventWait.event = allocEvent;
cudaGraphAddNode(&allocReadyEventNode, waitAndFreeGraph, NULL, NULL, 0, allocWaitEventNodeParams);

kernelNodeParams->kernelParams[0] = allocNodeParams.alloc.dptr;

// allocReadyEventNode 为消费 graph 中的使用操作提供与 alloc 节点之间的顺序保证
cudaGraphAddNode(&kernelNode, waitAndFreeGraph, &allocReadyEventNode, NULL, 1, &kernelNodeParams);

// free 节点必须排在所有外部和内部使用之后
// 因此该节点必须同时依赖 kernelNode 和 streamUseDoneEventNode
dependencies[0] = kernelNode;
dependencies[1] = streamUseDoneEventNode;

cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = dptr;
cudaGraphAddNode(&freeNode, waitAndFreeGraph, &dependencies, NULL, 2, freeNodeParams);
cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, allocStream);

// 通过让 stream2 等待 event 节点来建立依赖关系，从而满足顺序要求
cudaStreamWaitEvent(stream2, allocEvent);
kernel<<< ..., stream2 >>> (dptr, ...);
cudaStreamRecordEvent(streamUseDoneEvent, stream2);

// waitAndFreeGraphExec 中的 event wait 节点建立了对 “readyForFreeEvent” 的依赖
// 该依赖用于防止 stream2 中运行的 kernel 在执行顺序上位于 free 节点之后仍然访问该分配
cudaGraphLaunch(waitAndFreeGraphExec, stream3);
```

##### 4.2.5.2.4. cudaGraphInstantiateFlagAutoFreeOnLaunch

在正常情况下，如果一个 graph 中存在尚未释放的内存分配，CUDA 会阻止该 graph 被重新 launch，因为在相同地址上存在多个未释放的分配会导致内存泄漏。使用 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志来 instantiate 一个 graph，可以在该 graph 仍然存在未释放的内存分配时允许它被重新 launch。在这种情况下，该 launch 会自动插入对这些未释放分配的异步 free 操作。

Auto free on launch 对于 **single-producer multiple-consumer** 算法非常有用。在每次迭代中，producer graph 会创建多个分配，并且根据运行时条件，不同的 consumer 会访问这些分配。这种可变的执行序列意味着 consumers 无法进行 free 操作，因为后续的 consumer 可能仍然需要访问这些内存。Auto free on launch 的方式意味着 launch 循环不需要跟踪 producer 的所有分配 —— 这些信息仅限于 producer 的创建和销毁逻辑内部处理。总体而言，auto free on launch 简化了原本必须在每次重新 launch 之前释放 graph 所有分配的算法。

> 注意
>  `cudaGraphInstantiateFlagAutoFreeOnLaunch` 标志并不会改变 graph 销毁时的行为。为了避免内存泄漏，即便 graph 是用该标志 instantiate 的，应用程序仍然需要显式释放未释放的内存。下面代码示例展示了如何使用 `cudaGraphInstantiateFlagAutoFreeOnLaunch` 来简化一个 single-producer / multiple-consumer 算法：

```c++
// 创建一个 producer graph，在其中分配内存并填充数据
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
cudaMallocAsync(&data1, blocks * threads, cudaStreamPerThread);
cudaMallocAsync(&data2, blocks * threads, cudaStreamPerThread);
produce<<<blocks, threads, 0, cudaStreamPerThread>>>(data1, data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&producer,
                              graph,
                              cudaGraphInstantiateFlagAutoFreeOnLaunch);
cudaGraphDestroy(graph);

// 通过捕获一个异步库调用来创建第一个 consumer graph
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consumerFromLibrary(data1, cudaStreamPerThread);
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer1, graph, 0); // 普通 instantiate
cudaGraphDestroy(graph);

// 创建第二个 consumer graph
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consume2<<<blocks, threads, 0, cudaStreamPerThread>>>(data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer2, graph, 0);
cudaGraphDestroy(graph);

// 在循环中 launch
bool launchConsumer2 = false;
do {
    cudaGraphLaunch(producer, myStream);
    cudaGraphLaunch(consumer1, myStream);
    if (launchConsumer2) {
        cudaGraphLaunch(consumer2, myStream);
    }
} while (determineAction(&launchConsumer2));

cudaFreeAsync(data1, myStream);
cudaFreeAsync(data2, myStream);

cudaGraphExecDestroy(producer);
cudaGraphExecDestroy(consumer1);
cudaGraphExecDestroy(consumer2);
```

##### 4.2.5.2.5. Memory Nodes in Child Graphs

CUDA 12.9 引入了将子图所有权转移到父图的能力。被移动到父图的子图可以包含内存分配和释放节点。这样就允许含有分配或释放节点的子图在添加到父图之前独立构建。

以下限制适用于已被移动的子图：

- 无法单独进行实例化或销毁。
- 不能作为另一个父图的子图添加。
- 不能用作 `cuGraphExecUpdate` 的参数。
- 不能再添加额外的内存分配或释放节点。

```c++
// 创建子图
cudaGraphCreate(&child, 0);

// 基本分配的参数
cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
// 指定设备 0 作为驻留设备
allocNodeParams.alloc.poolProps.location.id = 0;
allocNodeParams.alloc.bytesize = size;

cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);
// 这里可以添加使用该分配的其他节点
cudaGraphNodeParams freeNodeParams = { cudaGraphNodeTypeMemFree };
freeNodeParams.free.dptr = allocNodeParams.alloc.dptr;
cudaGraphAddNode(&freeNode, graph, &allocNode, NULL, 1, freeNodeParams);

// 创建父图
cudaGraphCreate(&parent, 0);

// 将子图移动到父图
cudaGraphNodeParams childNodeParams = { cudaGraphNodeTypeGraph };
childNodeParams.graph.graph = child;
childNodeParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
cudaGraphAddNode(&parentNode, parent, NULL, NULL, 0, &childNodeParams);
```

#### 4.2.5.3. Optimized Memory Reuse

CUDA 通过两种方式来复用内存：

* 在 graph 内部，virtual memory 和 physical memory 的复用基于 virtual address 的分配方式，类似于 stream ordered allocator 的机制。
* 在 graph 之间，physical memory 的复用通过 virtual aliasing 实现：不同的 graph 可以将同一块 physical memory 映射到各自独立的 virtual address。

##### 4.2.5.3.1. Address Reuse within a Graph

CUDA 可以在 graph 内通过将相同的 virtual address 范围分配给生命周期不重叠的不同 allocation 来复用内存。由于 virtual address 可能会被复用，指向生命周期互不重叠的不同 allocation 的指针并不保证是唯一的。

下图展示了添加一个新的 allocation 节点（2），该节点可以复用由其依赖的节点（1）释放的地址。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/new-alloc-node.png)

*Figure 28  *添加新的 Alloc 节点 2*

下图展示了添加一个新的 alloc 节点（4）。该新的 alloc 节点不依赖于 free 节点（2），因此无法复用与 alloc 节点（2）关联的地址。如果 alloc 节点（2）使用了由 free 节点（1）释放的地址，那么新的 alloc 节点（3）就必须使用一个新的地址。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/adding-new-alloc-nodes.png)

*Figure 29  Kernel节点*

##### 4.2.5.3.2. Physical Memory Management and Sharing

CUDA 负责在 GPU 执行顺序中、到达 allocation node 之前，将 physical memory 映射到 virtual address。为了优化 memory footprint 和映射开销，如果多个 graph 不会同时运行，它们的不同 allocation 可以使用同一块 physical memory；但是，如果 physical page 同时绑定到多个正在执行的 graph，或者绑定到尚未释放的 graph allocation，则不能复用这些 physical page。

在 graph instantiation、launch 或 execution 期间的任何时刻，CUDA 都可能更新 physical memory 的映射。为了防止仍然存活的 graph allocation 引用同一块 physical memory，CUDA 也可能在后续的 graph launch 之间引入同步。和任何 allocate-free-allocate 模式一样，如果程序在 allocation 生命周期之外访问某个指针，这种错误访问可能会悄无声息地读取或写入属于其他 allocation 的活动数据（即使该 allocation 的 virtual address 是唯一的）。可以使用 compute sanitizer 工具来捕获这种错误。

下图展示了在同一 stream 中顺序 launch 的多个 graph。在这个例子中，每个 graph 都会释放它所分配的全部内存。由于同一 stream 中的 graph 从不并发执行，CUDA 可以、也应该使用同一块 physical memory 来满足所有 allocation 的需求。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/sequentially-launched-graphs.png)

*Figure 30  顺序 Launch 的 Graph*

#### **4.2.5.4. Performance Considerations**

当多个 graph 被 launch 到同一个 stream 中时，由于这些 graph 的执行不会重叠，CUDA 会尝试为它们分配相同的 physical memory。作为一种优化，graph 的 physical mapping 会在多次 launch 之间被保留，以避免 remapping 的开销。如果在之后的某个时刻，其中一个 graph 的 launch 方式使得它的执行可能与其他 graph 重叠（例如被 launch 到不同的 stream 中），那么 CUDA 就必须执行一定的 remapping，因为并发执行的 graph 需要使用不同的内存以避免数据损坏。

通常情况下，CUDA 中 graph memory 的 remapping 往往由以下操作触发：

* 更改 graph 被 launch 到的 stream
* 对 graph memory pool 执行 trim 操作，该操作会显式释放未使用的内存（在 graph-memory-nodes-physical-memory-footprint 中讨论）
* 在另一个 graph 的 allocation 尚未释放、且映射到同一块内存的情况下重新 launch 一个 graph，会在重新 launch 之前触发 memory 的 remap

Remapping 必须按照 execution 顺序进行，并且只能在该 graph 之前的所有执行完成之后进行（否则仍在使用中的内存可能会被解除映射）。由于这种顺序依赖关系，再加上映射操作本身是 OS 调用，mapping 操作的开销可能相对较高。应用可以通过始终将包含 allocation memory node 的 graph launch 到同一个 stream 中来避免这类开销。

##### 4.2.5.4.1. First Launch / cudaGraphUpload

由于 graph 将在哪个 stream 中执行在 instantiation 阶段是未知的，因此在 graph instantiation 期间无法分配或映射 physical memory。映射操作会在 graph launch 时完成。调用 `cudaGraphUpload` 可以通过立即为该 graph 执行所有 mapping 并将 graph 与 upload stream 关联，从而将 allocation 的开销从 launch 中分离出来。如果随后将该 graph launch 到同一个 stream 中，它就可以在不进行任何额外 remapping 的情况下启动。

如果 graph upload 和 graph launch 使用不同的 stream，其行为类似于切换 stream，很可能会导致 remap 操作。此外，不相关的 memory pool 管理操作也可能从空闲的 stream 中回收内存，从而抵消 upload 带来的效果。

##### 4.2.5.5. 物理内存占用（Physical Memory Footprint）

由于异步分配的内存池管理机制，即使一个 graph 中包含了 memory nodes（即便这些分配已被释放），销毁该 graph 并不会立即把物理内存归还给操作系统供其他进程使用。要显式地将内存归还给操作系统，应用程序应调用 `cudaDeviceGraphMemTrim` API。

`cudaDeviceGraphMemTrim` 会取消映射并释放那些由 graph memory nodes 保留但当前 **未被使用** 的物理内存。尚未 freed 的分配以及已调度或正在运行的 graphs 被视为仍在使用物理内存，因此不会受到影响。使用 trim API 会使这些物理内存对其他 allocation APIs 以及其他应用/进程可用，但这也会导致 CUDA 在下次 launch 被 trim 的 graph 时重新分配和重新映射这些内存。需要注意的是，`cudaDeviceGraphMemTrim` 操作的内存池与 `cudaMemPoolTrimTo()` 不同。graph memory pool 并未暴露给 stream-ordered memory allocator。CUDA 允许应用通过 `cudaDeviceGetGraphMemAttribute` API 查询 graph 的内存占用情况。查询属性 `cudaGraphMemAttrReservedMemCurrent` 会返回当前进程中 driver 为 graph allocations 保留的物理内存量；查询 `cudaGraphMemAttrUsedMemCurrent` 会返回至少被一个 graph 映射的物理内存量。这两个属性都可以用来追踪 CUDA 为满足某个 allocating graph 所获取的新物理内存。它们对于检查共享机制节省了多少内存也很有用。

#### 4.2.5.6. 对等访问（Peer Access）

图（graph）分配可以配置为允许多个 GPU 访问，在这种情况下 CUDA 会根据需要将这些分配映射到各个对等 GPU 上。CUDA 允许需要不同映射的图分配重用相同的虚拟地址。当这种情况发生时，该地址范围会映射到所有由不同分配所要求的 GPU 上。这意味着一个分配有时可能允许比创建时请求的更多对等访问；然而，依赖这些额外的映射仍然是错误的做法。

##### 4.2.5.6.1. Peer Access with Graph Node APIs

`cudaGraphAddNode` API 在分配节点参数结构体的 `accessDescs` 数组字段中接受映射请求。参数结构体中嵌套的 `poolProps.location` 结构指定了分配的常驻设备。默认假定分配 GPU 本身需要访问，因此应用无需在 `accessDescs` 数组中为常驻设备指定条目。

```c++
cudaGraphNodeParams allocNodeParams = { cudaGraphNodeTypeMemAlloc };
allocNodeParams.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
allocNodeParams.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
// 指定设备 1 作为常驻设备
allocNodeParams.alloc.poolProps.location.id = 1;
allocNodeParams.alloc.bytesize = size;
// 在设备 1 上分配一块内存，并假定设备 1 本身可访问
cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);
accessDescs[2];
// accessDescs 的设置模板（add node API 仅支持 ReadWrite 和 Device 访问）
accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[0].location.type = cudaMemLocationTypeDevice;
accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[1].location.type = cudaMemLocationTypeDevice;
// 请求设备 0 和 2 的访问权限。设备 1 的访问要求由常驻设备隐含
accessDescs[0].location.id = 0;
accessDescs[1].location.id = 2;

// 访问请求数组包含 2 个条目
allocNodeParams.accessDescCount = 2;
allocNodeParams.accessDescs = accessDescs;
// 分配一块常驻在设备 1 且可被设备 0、1、2 访问的内存（0 和 2 来自 accessDescs，1 由常驻设备隐含）
cudaGraphAddNode(&allocNode, graph, NULL, NULL, 0, &allocNodeParams);
```

##### 4.2.5.6.2. Peer Access with Stream Capture

对于流捕获（stream capture），分配节点会记录捕获时分配内存池的对等可访问性。在捕获了某次 `cudaMallocFromPoolAsync` 调用之后，若随后更改了该内存池的对等访问设置，这不会影响图在分配节点上所做的映射。

```c++
// accessDesc 的模板设置（add node API 仅支持 ReadWrite 和 Device 访问）
accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
accessDesc.location.type = cudaMemLocationTypeDevice;
accessDesc.location.id = 1;

// 假设 memPool 初始常驻并且在设备 0 上可访问

cudaStreamBeginCapture(stream);
cudaMallocAsync(&dptr1, size, memPool, stream);
cudaStreamEndCapture(stream, &graph1);

cudaMemPoolSetAccess(memPool, &accessDesc, 1);
cudaStreamBeginCapture(stream);
cudaMallocAsync(&dptr2, size, memPool, stream);
cudaStreamEndCapture(stream, &graph2);

// 即使 memPool 现在也为设备 1 提供访问权限，为 dptr1 分配的图节点仍然只有设备 0 的访问权限。
// 而为 dptr2 分配的图节点将具有设备 0 和设备 1 的访问权限，因为这是 cudaMallocAsync 调用时池的访问性。
```

### 4.2.6. Device Graph Launch

在许多workflows中，需要在运行时根据数据作出决策，并根据这些决策执行不同的操作。与其将这种决策过程卸载到 host（这可能需要一次 device 到 host 的往返），用户可能更希望在 device 上直接完成。为此，CUDA 提供了一种从 device 启动 graph 的机制。

Device graph launch 为在 device 上实现动态控制流提供了一种便捷方式，无论是像循环这样简单的控制结构，还是像 device 端工作调度器这样复杂的逻辑。

能够从 device 启动的 graph 统称为 device graph，而不能从 device 启动的 graph 则称为 host graph。

Device graph 可以从 host 和 device 两端启动，而 host graph 只能从 host 启动。与 host 启动不同的是，如果在一次 device graph 启动仍在运行时，又尝试从 device 再次启动该 graph，将会产生错误，并返回 `cudaErrorInvalidValue`；因此，同一个 device graph 不能同时在 device 上启动两次。同时从 host 和 device 启动同一个 device graph 会导致未定义行为。

#### 4.2.6.1. Device Graph Creation

要使一个 graph 能够从 device 启动，必须在实例化（instantiation）时显式地为 device launch 进行配置。这通过在调用 `cudaGraphInstantiate()` 时传入 `cudaGraphInstantiateFlagDeviceLaunch` 标志来实现。与 host graph 一样，device graph 的结构在实例化时即被固定，之后无法修改，除非重新实例化；并且实例化操作只能在 host 上进行。为了能够被实例化为 device launch，graph 必须满足一系列要求。

##### 4.2.6.1.1. Device Graph Requirements

通用要求：

* graph 中的所有节点必须全部位于同一个 device 上。
* graph 只能包含 kernel 节点、memcpy 节点、memset 节点以及 child graph 节点。

Kernel 节点相关要求：

* graph 中的 kernel 不允许使用 CUDA Dynamic Parallelism。
* 只要未启用 MPS，就允许使用 cooperative launch。

Memcpy 节点相关要求：

* 只允许涉及 device memory 和/或 pinned 的 device-mapped host memory 的拷贝操作。
* 不允许涉及 CUDA array 的拷贝。
* 在实例化时，拷贝操作的两个操作数都必须能从当前 device 访问。需要注意的是，拷贝操作始终由 graph 所在的 device 执行，即使目标内存位于其他 device 上。

##### 4.2.6.1.2. Device Graph Upload

为了能够在设备端启动 graph，首先必须将它上传到设备上，以便为 graph 分配和初始化所需的设备端资源。这可以通过两种方式完成。

第一种是**显式上传**：可以调用 `cudaGraphUpload()` 来进行上传，或者在实例化 graph 时通过 `cudaGraphInstantiateWithParams()` 请求同时进行上传。

第二种方式是**隐式上传**：先从主机端启动这个 graph，graph 的上传步骤将在 launch 调用过程中自动完成。

下面的代码示例展示了这三种方法的用法：

```c++
// 在实例化之后显式上传
cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec1, stream);

// 在实例化参数中指定上传标志，作为实例化的一部分显式上传
cudaGraphInstantiateParams instantiateParams = {0};
instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
instantiateParams.uploadStream = stream;
cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

// 通过主机 launch 隐式上传
cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphLaunch(deviceGraphExec3, stream);
```

##### 4.2.6.1.3 Device Graph Update

Device graph 只能由主机端进行更新，并且在对可执行 graph 做更新之后，必须将更新后的 graph 重新上传到设备，才能使这些更改生效。重新上传可采用前面 **Device Graph Upload** 小节中介绍的相同方式。与 host graph 不同，如果在更新正在应用期间从设备端启动这个 device graph，将导致未定义行为。

#### 4.2.6.2. 设备端启动（Device Launch）

Device graph 可以通过 `cudaGraphLaunch()` 从 host 端和 device 端启动，并且该 API 在 device 端与在 host 端具有相同的函数签名。无论是在 host 端还是在 device 端，device graph 都是通过同一个 handle 来启动的。当从 device 端启动时，device graph 必须由另一个 graph 触发启动。

device 端的 graph 启动是按线程进行的，不同线程可以在同一时间发起多个启动，因此用户需要明确选择一个线程来负责启动某个特定的 graph。

与 host 端启动不同，device graph 不能被启动到普通的 CUDA stream 中，而只能启动到特定命名的 stream 中，这些 stream 分别表示不同的启动模式。下表列出了可用的启动模式。

表 9：仅用于 device 的 Graph 启动 Stream

| Stream                                  | 启动模式             |
| --------------------------------------- | -------------------- |
| `cudaStreamGraphFireAndForget`          | Fire and forget 启动 |
| `cudaStreamGraphTailLaunch`             | Tail 启动            |
| `cudaStreamGraphFireAndForgetAsSibling` | Sibling 启动         |

##### 4.2.6.2.1. Fire and Forget Launch（即发即忘启动）

顾名思义，fire and forget 启动会被立即提交到 GPU 上执行，并且其运行过程不依赖于发起启动的 graph。在 fire-and-forget 场景中，发起启动的 graph 是父 graph，而被启动的 graph 是子 graph。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/fire-and-forget-simple.png)

*Figure 31  Fire and forget 启动*

上图可以通过下面的示例代码生成：

```c++
__global__ void launchFireAndForgetGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // 创建、实例化并上传 device graph
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // 创建并实例化用于启动的 graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchFireAndForgetGraph<<<1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // 启动 host graph，它会进一步启动 device graph
    cudaGraphLaunch(gExec1, stream);
}
```

在一次执行过程中，一个 graph 最多可以启动 120 个 fire-and-forget graph。这个计数会在同一个父 graph 的不同启动之间重置。

###### 4.2.6.2.1.1. Graph Execution Environments（graph 执行环境）

为了全面理解 device 端的同步模型，首先需要理解 execution environment 这一概念。

当一个 graph 从 device 端启动时，它会被启动到一个独立的 execution environment 中。某个 graph 的 execution environment 会封装该 graph 内的所有工作，以及它所生成的所有 fire and forget 工作。只有当该 graph 本身执行完成，并且所有由它生成的子工作也都完成时，才能认为这个 graph 已经完成。

下图展示了上一节 fire-and-forget 示例代码所生成的 execution environment 封装关系。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/fire-and-forget-environments.png)

*Figure 32  带有 execution environment 的 fire and forget 启动*

这些 environment 也是分层(hierarchical)的，因此一个 graph 的 environment 可以包含由 fire and forget 启动产生的多层子 environment。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/fire-and-forget-nested-environments.png)

*Figure 33  嵌套的 fire and forget environment*

当一个 graph 从 host 端启动时，会存在一个 stream environment，作为被启动 graph 的 execution environment 的父环境。stream environment 会封装整个启动过程中所生成的所有工作。当整个 stream environment 被标记为完成时，stream 启动才算完成（也就是说，下游依赖的工作此时才可以开始执行）。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/device-graph-stream-environment.png)

*Figure 34  stream environment 的可视化示意*

##### 4.2.6.2.2. Tail Launch（尾随启动）

与在 host 上不同，在 GPU 端**不能用传统方法**（比如 `cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()`）来与 device graph 进行同步。为了在 device 上实现**串行工作依赖关系**，CUDA 提供了一种不同的 launch 模式 —— **tail launch**，可以实现类似的同步效果。

**Tail launch** 会在**图的执行环境被认为完成时**执行 —— 换句话说，只有当该图及其所有派生的子图全部完成之后才会执行。当一个图完成后，tail launch 列表中的下一个图的执行环境就会取代已完成图的执行环境，成为其父执行环境的子环境。和 fire-and-forget launch 一样，一个图可以排队多个图来进行 tail launch。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tail-launch-simple.png)

*Figure 35  一个简单的 tail launch*

上图可以通过下面的示例代码生成：

```c++
__global__ void launchTailGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // 创建、实例化并上传 device graph。
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // 创建并实例化用于 launch 的 graph。
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchTailGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // 发起 host graph 的 launch，这将继而在 device 端 launch 图。
    cudaGraphLaunch(gExec1, stream);
}
```

对于某个图排队的 tail launch，它们会**按 enqueue（入队）顺序**一个接一个执行。因此，第一个排队的图会先运行，然后才是第二个，依此类推。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tail-launch-ordering-simple.png)

*Figure 36   Tail Launch 的执行顺序*

如果一个 tail graph 本身也排队了 tail launch，那么这些来自该 tail graph 的 launch 会**优先于**tail launch 列表中之前其它图排队的 launch 执行。这些新的 tail launch 同样是按它们入队的顺序执行。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/tail-launch-ordering-complex.png)

*Figure 37  来自多个 graph 的 Tail Launch 入队时的执行顺序*

一个图最多 **可以有 255 个待执行的 tail launch** 排队。

###### 4.2.6.2.2.1. Tail Self-launch

device graph 可以将**自身**入队为一个 tail launch，不过对于同一个 graph，在任意时刻**只能有一个 self-launch** 被入队。为了能够查询当前正在运行的 device graph 并对其进行重新 launch，CUDA 新增了一个 device 端函数：

```c++
cudaGraphExec_t cudaGetCurrentGraphExec();
```

如果当前正在运行的是一个 device graph，该函数会返回当前 graph 的 handle。如果当前执行的 kernel 并不是某个 device graph 中的 node，那么该函数会返回 NULL。

下面的示例代码展示了如何使用该函数来实现一个 relaunch 循环：

```c++
__device__ int relaunchCount = 0;

__global__ void relaunchSelf() {
    int relaunchMax = 100;

    if (threadIdx.x == 0) {
        if (relaunchCount < relaunchMax) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        }

        relaunchCount++;
    }
}
```

##### 4.2.6.2.3. Sibling Launch

Sibling launch 是 fire-and-forget launch 的一种变体。在这种模式下，被 launch 的 graph **不是**作为发起 launch 的 graph 的 execution environment 的子环境运行，而是作为该 graph 的**父 execution environment** 的子环境运行。换句话说，Sibling launch 等价于**从发起 launch 的 graph 的父 execution environment 发起一次 fire-and-forget launch**。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/sibling-launch-simple.png)

*Figure 38  一个简单的 sibling launch 示例*

上图可以通过下面的示例代码生成：

```c++
__global__ void launchSiblingGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForgetAsSibling);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // 创建、实例化并上传 device graph。
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // 创建并实例化用于 launch 的 graph。
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchSiblingGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // 发起 host graph 的 launch，这将继而 launch device graph。
    cudaGraphLaunch(gExec1, stream);
}
```

由于 sibling launch **并不是在发起 launch 的 graph 的 execution environment 中执行**，因此它们**不会阻塞**该 graph 入队的 tail launch。

### 4.2.7. 使用 Graph APIs

`cudaGraph_t` 对象**不是线程安全的**。用户需要自行确保不会有多个线程同时访问同一个 `cudaGraph_t`。

同一个 `cudaGraphExec_t` **不能与自身并发执行**。对某个 `cudaGraphExec_t` 的 launch 会被顺序排在该 executable graph 之前所有 launch 之后执行。

Graph 的执行是在 stream 中完成的，用于与其他异步工作进行顺序控制。不过，stream **仅用于顺序约束**；它既不会限制 graph 内部的并行度，也不会影响 graph node 在哪里执行。

参见 Graph API。

### 4.2.8. CUDA User Objects

CUDA User Objects 可用于帮助管理 CUDA 中异步工作所使用资源的生命周期。该特性在 cuda-graphs 和 stream capture 场景下尤其有用。

多种资源管理方案与 CUDA graphs **并不兼容**。例如，基于 event 的资源池，或者同步创建、异步销毁的管理方式。

```c++
// 使用资源池分配的库 API
void libraryWork(cudaStream_t stream) {
    auto &resource = pool.claimTemporaryResource();
    resource.waitOnReadyEventInStream(stream);
    launchWork(stream, resource);
    resource.recordReadyEvent(stream);
}
```

```c++
// 使用异步资源销毁的库 API
void libraryWork(cudaStream_t stream) {
    Resource *resource = new Resource(...);
    launchWork(stream, resource);
    cudaLaunchHostFunc(
        stream,
        [](void *resource) {
            delete static_cast<Resource *>(resource);
        },
        resource,
        0);
    // 此处未展示错误处理相关逻辑
}
```

这些方案在 CUDA graphs 中实现起来非常困难，原因在于资源的指针或 handle **并不是固定的**，这就需要额外的间接访问或 graph update，而且每次提交工作时都需要同步的 CPU 代码。如果这些细节对库的调用者是隐藏的，或者在 capture 过程中使用了不允许的 API，这些方案同样无法与 stream capture 一起工作。已有的一些解决方案包括将资源暴露给调用者，而 CUDA user objects 提供了另一种思路。

CUDA user object 将**用户指定的析构回调**与一个内部的 refcount 关联起来，其行为类似于 C++ 的 `shared_ptr`。引用既可以由 CPU 端的用户代码持有，也可以由 CUDA graphs 持有。需要注意的是，对于用户持有的引用，不像 C++ 智能指针那样有一个对象来表示引用本身，用户需要手动跟踪这些引用。一个典型的使用方式是在创建 user object 后，立即将唯一的用户侧引用转移给某个 CUDA graph。

当某个引用与 CUDA graph 关联之后，CUDA 会自动管理相关的 graph 操作。对一个 `cudaGraph_t` 进行 clone 时，新 graph 会保留源 `cudaGraph_t` 所拥有的每一个引用，并且引用的数量保持一致。实例化得到的 `cudaGraphExec_t` 会保留源 `cudaGraph_t` 中的所有引用。如果在未同步的情况下销毁 `cudaGraphExec_t`，这些引用会一直保留，直到执行完成为止。

下面是一个使用示例。

```c++
cudaGraph_t graph;  // 已存在的 graph

Object *object = new Object;  // C++ 对象，可能具有非平凡的析构函数
cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object,  // 这里使用了 CUDA 提供的模板封装，
             // 该封装会提供一个用于 delete C++ 对象指针的回调
    1,  // 初始 refcount
    cudaUserObjectNoDestructorSync  // 表明该回调无法通过 CUDA 等待
);
cudaGraphRetainUserObject(
    graph,
    cuObject,
    1,  // 引用数量
    cudaGraphUserObjectMove  // 转移调用方持有的一个引用
                             //（不修改总引用计数）
);
// 当前线程不再持有任何引用，无需调用 release API
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // 会保留一个
                                                               // 新的引用
cudaGraphDestroy(graph);  // graphExec 仍然持有一个引用
cudaGraphLaunch(graphExec, 0);  // 异步 launch 可访问 user object
cudaGraphExecDestroy(graphExec);  // launch 未同步，如有需要，释放会被延迟
cudaStreamSynchronize(0);  // 在 launch 同步完成后，剩余的引用会被释放，
                           // 析构函数将被执行（注意这是异步发生的）
// 如果析构回调中触发了某个同步对象，此时等待它是安全的
```

child graph node 中的 graph 所持有的引用**归属于对应的 child graph**，而不是父 graph。如果某个 child graph 被更新或删除，这些引用也会随之变化。如果通过 `cudaGraphExecUpdate` 或 `cudaGraphExecChildGraphNodeSetParams` 更新 executable graph 或 child graph，新 source graph 中的引用会被 clone，并替换 target graph 中原有的引用。在上述任意情况下，如果之前的 launch 尚未同步，所有本应被释放的引用都会一直保留，直到这些 launch 执行完成。

目前还**没有**通过 CUDA API 等待 user object 析构函数执行完成的机制。用户可以在析构函数中手动触发某个同步对象。此外，与 `cudaLaunchHostFunc` 的限制类似，在析构函数中调用 CUDA API 是不合法的，这是为了避免阻塞 CUDA 内部的共享线程，从而影响执行向前推进。如果依赖关系是单向的，并且执行 API 调用的线程不会阻塞 CUDA 工作的推进，那么通过通知另一个线程来执行 API 调用是合法的。

User object 通过 `cudaUserObjectCreate` 创建，这是查阅相关 API 的一个很好起点。
