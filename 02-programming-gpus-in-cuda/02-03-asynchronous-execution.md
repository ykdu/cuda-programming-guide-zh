# 2.3 Asynchronous Execution

### 2.3.1. 什么是 Asynchronous Concurrent Execution？

CUDA 允许多个任务并发执行，或者说以重叠的方式执行，具体包括：

* 在 host 上进行的计算
* 在 device 上进行的计算
* 从 host 到 device 的 memory 传输
* 从 device 到 host 的 memory 传输
* 在某个 device 的 memory 内部进行的 memory 传输
* device 之间的 memory 传输

这种并发性通过 asynchronous 接口来表达，在这种接口中，调度函数调用或 kernel launch 会立即返回。Asynchronous 调用通常在被调度的操作完成之前就返回，甚至可能在 asynchronous 操作真正开始之前就返回。此时，application 可以在原先调度的操作执行的同时，自由地执行其他任务。当需要最初调度的操作的最终结果时，application 必须执行某种形式的 synchronize，以确保相关操作已经完成。并发执行模式的一个典型示例是将 host 和 device 之间的 memory 传输与计算过程进行重叠，从而减少甚至消除这些操作的开销。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/cuda_streams.png)

*图 17 使用 CUDA streams 的 Asynchronous Concurrent Execution*

总体来说，asynchronous 接口通常提供三种主要方式，用于与已调度的操作进行 synchronize：

* blocking 方式，application 调用一个会阻塞的函数，等待操作完成
* non-blocking 方式，或称 polling 方式，application 调用一个立即返回的函数，并获得关于操作状态的信息
* callback 方式，在操作完成时执行一个预先注册的函数

尽管编程接口是 asynchronous 的，但实际能否并发执行各种操作，取决于所使用的 CUDA 版本以及硬件的 compute capability——这些细节将在本指南后续的章节中介绍（参见 Compute Capabilities）。

在 “Synchronizing CPU and GPU” 一节中，引入了 CUDA runtime 函数 `cudaDeviceSynchronize()`，这是一个 blocking 调用，用于等待之前发出的所有工作完成。之所以需要调用 `cudaDeviceSynchronize()`，是因为 kernel launch 是 asynchronous 的，并且会立即返回。CUDA 提供了同时支持 blocking 和 non-blocking synchronize 的 API，甚至还支持使用 host 侧的 callback 函数。

CUDA 中用于实现 asynchronous execution 的核心 API 组件是 CUDA Streams 和 CUDA Events。在本节的其余部分，我们将说明如何使用这些元素来表达 CUDA 中的 asynchronous execution。

一个相关的话题是 CUDA Graphs，它允许事先定义一个由 asynchronous 操作组成的 graph，然后以极低的开销反复执行。我们将在第 2.4.9.2 节 “Introduction to CUDA Graphs with Stream Capture” 中对 CUDA Graphs 进行非常基础的介绍，并在第 4.1 节 “CUDA Graphs” 中提供更为全面的讨论。

### 2.3.2. CUDA Streams

在最基本的层面上，CUDA stream 是一个抽象，它允许程序员表达一系列操作。stream 的行为类似一个工作队列，程序可以将诸如内存拷贝或 kernel launch 之类的操作加入队列，按顺序执行。对于某个 stream，队列最前面的操作会被执行，然后出队，使下一个排队的操作移到队首并被考虑执行。stream 中操作的执行顺序是顺序的，操作按它们被加入 stream 的顺序执行。

一个应用程序可以同时使用多个 streams。在这种情况下，runtime 会根据 GPU 资源的状态，从有可用工作的 streams 中选择一个任务去执行。streams 可以被分配一个优先级，这个优先级作为对 runtime 的调度提示，但不能保证特定的执行顺序。

在 stream 中的 API 函数调用和 kernel launches 对 host thread 来说都是 asynchronous 的。应用程序可以通过等待 stream 内没有任务来与 stream 同步，也可以在 device 级别进行同步。

CUDA 有一个默认的 stream，当操作和 kernel launch 没有指定具体的 stream 时，它们会被加入到这个默认 stream。示例代码中若不指定 stream，则隐式使用默认 stream。默认 stream 有一些特定的语义，这将在小节 “Blocking and non-blocking streams and the default stream” 中讨论。

#### 2.3.2.1. Creating and Destroying CUDA Streams

可以使用 `cudaStreamCreate()` 函数来创建 CUDA streams。该函数调用初始化 stream handle，这个 handle 可用于后续函数调用中标识该 stream。

```c++
cudaStream_t stream;        // stream handle
cudaStreamCreate(&stream);  // 创建一个新的 stream

// 基于 stream 的操作...

cudaStreamDestroy(stream);  // 销毁 stream
```

如果在应用程序调用 `cudaStreamDestroy()` 时设备仍在 stream `stream` 中执行工作，那么该 stream 会在被销毁前完成 stream 中的所有工作。

#### 2.3.2.2. Launching Kernels in CUDA Streams

可以使用通常的三角括号语法来将 kernel launch 到一个特定的 stream。stream 作为额外的参数传给 kernel launch。下面示例中，名为 `kernel` 的 kernel 被 launch 到 handle 为 `stream` 的 stream 中，该 stream 类型为 `cudaStream_t`，假定之前已经创建成功。

```c++
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

kernel launch 是 asynchronous 的，并且函数调用立即返回。假设 kernel launch 成功，kernel 会在 stream `stream` 中执行，同时应用程序可以在 CPU 上或 GPU 上的其他 streams 执行其他任务。

#### 2.3.2.3. Launching Memory Transfers in CUDA Streams

要在 stream 中启动内存传输，可以使用函数 `cudaMemcpyAsync()`。此函数类似于 `cudaMemcpy()`，但它多了一个参数用于指定用于内存传输的 stream。下面代码块将从 host memory 中由 `src` 指向的内存复制 `size` 字节到由 `dst` 指向的 device memory，并在 stream `stream` 中执行。

```c++
// 在 stream `stream` 中将 `size` 字节从 `src` 复制到 `dst`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

像其他 asynchronous 函数调用一样，此函数调用立即返回，而 `cudaMemcpy()` 会阻塞直到内存传输完成。为了安全访问传输结果，应用程序必须通过某种同步机制来确定操作已完成。

其他 CUDA 内存传输函数如 `cudaMemcpy2D()` 也有 asynchronous 变体。

> 注意
>
> **为了使涉及 CPU 内存的内存拷贝能够异步执行，host buffer 必须是 pinned 并 page-locked 的。**如果使用的 host memory 不是 pinned 并 page-locked，`cudaMemcpyAsync()` 虽然可以正常运行，但将退化为同步行为，不能与其他工作重叠，这可能抑制异步内存传输的性能优势。建议程序使用cudaMallocHost（）来分配缓冲区，这些缓冲区将用于从gpu发送或接收数据。

#### 2.3.2.4. Stream Synchronization

与 stream 同步的最简单方法是等待 stream 内的任务为空。这可以通过 `cudaStreamSynchronize()` 函数或 `cudaStreamQuery()` 函数实现。

`cudaStreamSynchronize()` 函数会阻塞直到 stream 中的所有工作完成。

```c++
// 等待 stream 内没有任务
cudaStreamSynchronize(stream);

// 此时 stream 已完成
// 我们可以安全访问 stream 操作的结果
```

如果不希望阻塞，但只是需要快速检查 stream 是否为空，可以使用 `cudaStreamQuery()`。

```c++
// 查看 stream 状态
// 如果 stream 为空返回 cudaSuccess
// 如果 stream 不为空返回 cudaErrorNotReady
cudaError_t status = cudaStreamQuery(stream);

switch (status) {
    case cudaSuccess:
        // stream 为空
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // stream 不为空
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // 出现错误 — 应进行处理
        break;
};
```

### 2.3.3. CUDA Events

CUDA events 是一种向 CUDA stream 插入标记的机制。它们本质上像是 tracer 粒子一样，可用于跟踪 stream 中任务的进度。想象在一个 stream 中启动两个 kernel。在没有这些追踪事件的情况下，我们只能确定 stream 是否为空。如果有一个操作依赖第一个 kernel 的输出，则在确认第一个 kernel 完成前无法安全开始该操作，而此时两个 kernel 都已经执行完。

使用 CUDA Events 可以更好地处理这个问题。通过在第一个 kernel 之后、第二个 kernel 之前将事件插入到 stream 中，我们可以等待这个事件移动到 stream 的最前端。然后我们就可以安全地开始依赖操作，知道第一个 kernel 已经完成，而第二个 kernel 还未开始。以这种方式使用 CUDA events 可以构建操作和 streams 之间的依赖图。这个图的类比直接对应于后面对 CUDA graphs 的讨论。

 CUDA streams 还保留时间信息，可以用于测量 kernel 启动和内存传输的执行时间。

#### 2.3.3.1. Creating and Destroying CUDA Events

可以使用 `cudaEventCreate()` 和 `cudaEventDestroy()` 函数来创建和销毁 CUDA Events。

```c++
cudaEvent_t event;

// 创建 event
cudaEventCreate(&event);

// 执行一些与 event 相关的工作

// 一旦工作完成并且 event 不再需要
// 我们可以销毁该 event
cudaEventDestroy(event);
```

应用负责在 event 不再需要时销毁它。

#### 2.3.3.2. Inserting Events into CUDA Streams

可以使用 `cudaEventRecord()` 函数将 CUDA Events 插入到 stream 中。

```c++
cudaEvent_t event;
cudaStream_t stream;

// 创建 event
cudaEventCreate(&event);

// 将 event 插入到 stream
cudaEventRecord(event, stream);
```

#### 2.3.3.3. Timing Operations in CUDA Streams

CUDA events 可以用于测量包括 kernels 在内的各种 stream 操作的执行时间。当 event 到达 stream 的最前端时它会记录一个时间戳。通过用两个 events 围绕一个 stream 中的 kernel，我们可以准确测量 kernel 执行的持续时间，如下面的代码片段所示：

```c++
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;
// 创建 events
cudaEventCreate(&start);
cudaEventCreate(&stop);

 // 记录开始 event
cudaEventRecord(start, stream);

// 启动 kernel
kernel<<<grid, block, 0, stream>>>(...);

// 记录结束 event
cudaEventRecord(stop, stream);

// 等待 stream 完成
// 两个 event 都会被触发
cudaStreamSynchronize(stream);
// 获取时间
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

// 清理
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

#### 2.3.3.4. Checking the Status of CUDA Events

与检查 stream 状态类似，我们可以以阻塞或非阻塞方式检查 events 的状态。

`cudaEventSynchronize()` 函数会阻塞直到 event 完成。在下面的代码片段中，我们在一个 stream 中启动一个 kernel，接着插入一个 event，然后再启动第二个 kernel。我们可以使用 `cudaEventSynchronize()` 等待第一个 kernel 完成，并在原则上在 kernel2 完成前启动一个依赖的任务。

```c++
cudaEvent_t event;
cudaStream_t stream;

// 创建 stream
cudaStreamCreate(&stream);
// 创建 event
cudaEventCreate(&event);

// 在 stream 中启动一个 kernel
kernel<<<grid, block, 0, stream>>>(...);

// 记录 event
cudaEventRecord(event, stream);

// 再在 stream 中启动一个 kernel
kernel2<<<grid, block, 0, stream>>>(...);

// 等待 event 完成
// Kernel1 将被保证已经完成
// 并且我们可以启动依赖的任务
cudaEventSynchronize(event);
dependentCPUtask();
// 等待 stream 为空
// Kernel2 将被保证已经完成
cudaStreamSynchronize(stream);

// 销毁 event
cudaEventDestroy(event);

// 销毁 stream
cudaStreamDestroy(stream);
```

可以通过 `cudaEventQuery()` 函数以非阻塞方式检查 events 是否完成。下面的示例中我们在一个 stream 中启动了两个 kernels。第一个 kernel1 生成一些数据，我们希望将其复制回 host，但我们还有一些 CPU 侧工作需要做。在代码中，我们在 stream1 中依次排入 kernel1、一个 event 和 kernel2。然后进入一个 CPU 工作循环，但偶尔检查 event 是否完成以确定 kernel1 是否完成。如果完成，则在 stream2 中排入一条 host 到 device 的复制。这样可以实现 CPU 侧工作与 GPU kernel 执行以及设备到 host 的复制的重叠。

```c++
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float *d_data;

// 创建数据
cudaMalloc(&d_data, size);
float *h_data = (float *)malloc(size);

// 创建 streams
cudaStreamCreate(&stream1);   // 处理 stream
cudaStreamCreate(&stream2);   // 复制 stream
bool copyStarted = false;

//  创建 event
cudaEventCreate(&event);
// 在 stream1 中启动 kernel1
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// 在 kernel1 后插入 event
cudaEventRecord(event, stream1);

// 在 stream1 中启动 kernel2
kernel2<<<grid, block, 0, stream1>>>();
// 当 kernels 正在运行时执行 CPU 工作
// 但检查 kernel1 是否完成，因为完成后我们将在 stream2 中启动复制
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();
    // 检查 kernel1 是否完成
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}

// 等待两个 streams 完成
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
// 销毁 event
cudaEventDestroy(event);

// 销毁 streams 并释放数据
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

### 2.3.4. Callback Functions from Streams

CUDA 提供了一种机制，可以从 stream 内部在 host 上启动函数。目前有两个函数可用于此目的：`cudaLaunchHostFunc()` 和 `cudaAddCallback()`。不过，`cudaAddCallback()` 计划弃用，因此应用应该使用 `cudaLaunchHostFunc()`。

使用 `cudaLaunchHostFunc()`

`cudaLaunchHostFunc()` 函数的函数签名如下：

```
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```

其中

- `stream`：要将 callback 函数加入的 stream。
- `func`：要启动的 callback 函数。
- `data`：传递给 callback 函数的数据指针。

host 函数本身是一个简单的 C 函数，函数签名如下：

```
void hostFunction(void *data);
```

其中 `data` 参数指向一个用户定义的数据结构，函数可以解释该结构。使用类似的 callback 函数时需要注意一些事项。特别是，**host 函数不得调用任何 CUDA API。**

为了配合 unified memory 使用，提供了如下执行保证：– 在该函数执行期间，stream 被认为是 idle。因此，例如，该函数可以始终使用附加到其排入 stream 的内存。– 函数执行的开始具有与在此函数之前立即在同一个 stream 中记录的事件同步的相同效果。因此，它同步了在此函数之前“join” 的 streams。– 向任何 stream 添加 device 工作并不会使该 stream 变为 active，直到所有之前的 host 函数和 stream callbacks 执行完毕。因此，例如，即使另一 stream 添加了工作，只要该工作在带事件的函数调用之后排序，该host函数仍然可以使用全局附加内存。– 函数完成不会使 stream 除上述情况外变为 active。如果没有 device 工作跟随函数，则 stream 将保持 idle，并且在连续的 host 函数或 stream callbacks 之间没有 device 工作时也将保持 idle。因此，例如，流同步可以通过流末端的主机函数发出信号来完成。

#### 2.3.4.1. Using `cudaStreamAddCallback()`

> 注意
>
> `cudaStreamAddCallback()` 函数计划弃用和移除，这里讨论它是为了完整性并因为它可能仍出现在现有代码中。应用应该使用或切换到使用 `cudaLaunchHostFunc()`。

`cudaStreamAddCallback()` 函数的函数签名如下：

```
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

其中

- `stream`：要将 callback 函数加入的 stream。
- `callback`：要启动的 callback 函数。
- `userData`：传递给 callback 函数的数据指针。
- `flags`：目前该参数必须为 0 以保证将来兼容。

与使用 `cudaLaunchHostFunc()` 时的情况相比，`callback` 函数的签名略有不同。在此情况下，callback 函数是一个 C 函数，函数签名如下：

```
void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);
```

其中函数现在传入

- `stream`：启动 callback 函数的 stream handle。
- `status`：触发 callback 的 stream 操作的状态。
- `userData`：传递给 callback 函数的数据指针。

特别地，`status` 参数将包含 stream 的当前错误状态，该状态可能由之前的操作设置。与 `cudaLaunchHostFunc()` 的情况类似，stream 在 host 函数完成之前不会变为 active 并推进到任务，同时 callback 内部不得调用 CUDA 函数。

#### 2.3.4.2. Asynchronous Error Handling

在 cuda stream 中，错误可能源自 stream 中的任何操作，包括 kernel 启动和 memory transfers。这些错误直到 stream 被同步（例如通过等待某个 event 或调用 `cudaStreamSynchronize()`）时可能才会在运行时传播回用户。有两种方法可以查知 stream 中可能发生的错误。

- 使用函数 `cudaGetLastError()` —— 此函数返回并清除当前 context 中任何 stream 遇到的最后一个错误。如果紧接着第二次调用 cudaGetLastError()，如果在两次调用之间没有其他错误发生，它将返回 `cudaSuccess`。
- 使用函数 `cudaPeekAtLastError()` —— 此函数返回当前 context 中的最后一个错误，但不会清除它。

这两个函数都会返回一个类型为 `cudaError_t` 的错误值。可以使用函数 cudaGetErrorName() 和 cudaGetErrorString() 生成错误的可打印名称。

下面的示例展示了如何使用这些函数：

```c++
    // Some work occurs in streams.
    cudaStreamSynchronize(stream);
    // Look at the last error but do not clear it
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("Error with name: %s\n", cudaGetErrorName(err));
        printf("Error description: %s\n", cudaGetErrorString(err));
    }
    // Look at the last error and clear it
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("Error with name: %s\n", cudaGetErrorName(err2));
        printf("Error description: %s\n", cudaGetErrorString(err2));
    }

    if (err2 != err) {
        printf("As expected, cudaPeekAtLastError() did not clear the error\n");
    }
    // Check again
    cudaError_t err3 = cudaGetLastError();
    if (err3 == cudaSuccess) {
        printf("As expected, cudaGetLastError() cleared the error\n");
    }
```
