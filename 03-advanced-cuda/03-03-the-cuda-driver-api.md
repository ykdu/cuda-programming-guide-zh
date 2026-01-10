# 3.3 The CUDA Driver API

本指南前面的章节已经介绍了 CUDA runtime。正如 CUDA Runtime API and CUDA Driver API 中所提到的，CUDA runtime 是构建在更底层的 CUDA driver API 之上的。本节将介绍 CUDA runtime 和 driver API 之间的一些差异，以及如何将二者混合使用。大多数应用即使完全不接触 CUDA driver API，也可以获得完整的性能。然而，一些新的接口有时会先在 driver API 中提供，而不是在 runtime API 中提供；此外，一些高级接口（例如 Virtual Memory Management）只在 driver API 中暴露。

CUDA driver API 实现于 `cuda` 动态库中（`cuda.dll` 或 `cuda.so`），该库会在安装 device driver 时被复制到系统中。它的所有入口函数都以 `cu` 作为前缀。

它是一个基于句柄、命令式的 API：大多数对象都通过不透明的句柄来引用，这些句柄可以作为参数传递给函数，用于对对象进行操作。

driver API 中可用的对象总结在 **表 6** 中。

**表 6. CUDA Driver API 中可用的对象**

| Object            | Handle      | Description                                                  |
| ----------------- | ----------- | ------------------------------------------------------------ |
| Device            | CUdevice    | 支持 CUDA 的 device                                          |
| Context           | CUcontext   | 大致等价于一个 CPU process                                   |
| Module            | CUmodule    | 大致等价于一个动态库                                         |
| Function          | CUfunction  | Kernel                                                       |
| Heap memory       | CUdeviceptr | 指向 device memory 的指针                                    |
| CUDA array        | CUarray     | device 上一维或二维数据的不透明容器，可通过 texture 或 surface reference 访问 |
| Texture object    | CUtexref    | 描述如何解释 texture memory 数据的对象                       |
| Surface reference | CUsurfref   | 描述如何读写 CUDA array 的对象                               |
| Stream            | CUstream    | 描述一个 CUDA stream 的对象                                  |
| Event             | CUevent     | 描述一个 CUDA event 的对象                                   |

在调用任何 driver API 函数之前，必须先通过 `cuInit()` 初始化 driver API。随后需要创建一个 CUDA context，将其绑定到某个特定的 device，并将其设为当前 host thread 的 current context，具体细节见 **Context**。在一个 CUDA context 中，kernel 需要由 host 代码显式地以 PTX 或 binary object 的形式加载，如 **Module** 中所述。因此，用 C++ 编写的 kernel 必须先单独编译为 PTX 或 binary object。Kernel 的 launch 则通过 **Kernel Execution** 中描述的 API 完成。

任何希望能够在未来 device 架构上运行的应用，都必须加载 PTX，而不是 binary code。这是因为 binary code 是架构相关的，无法兼容未来的架构；而 PTX 会在加载时由 device driver 编译为对应架构的 binary code。

下面是 **Kernels** 中示例使用 driver API 编写的 host 代码：

```c
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);
    // 在 host memory 中分配输入向量 h_A 和 h_B
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // 初始化输入向量
    ...

    // 初始化
    cuInit(0);

    // 获取支持 CUDA 的 device 数量
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit (0);
    }
    // 获取 device 0 的句柄
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // 创建 context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, 0, cuDevice);

    // 从 binary 文件创建 module
    CUmodule cuModule;
    cuModuleLoad(&cuModule, "VecAdd.ptx");
    // 在 device memory 中分配向量
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, size);
    CUdeviceptr d_B;
    cuMemAlloc(&d_B, size);
    CUdeviceptr d_C;
    cuMemAlloc(&d_C, size);

    // 将向量从 host memory 拷贝到 device memory
    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    // 从 module 中获取 function 句柄
    CUfunction vecAdd;
    cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");
    // 启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &d_A, &d_B, &d_C, &N };
    cuLaunchKernel(vecAdd,
                   blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                   0, 0, args, 0);

    ...
}
```

完整代码可以在 `vectorAddDrv` CUDA sample 中找到。

### 3.3.1 Context

CUDA context 类似于一个 CPU process。所有通过 driver API 执行的资源分配和操作都被封装在一个 CUDA context 内部，并且当 context 被销毁时，系统会自动清理这些资源。除了 module、texture reference 或 surface reference 等对象之外，每个 context 还拥有自己独立的 address space。因此，来自不同 context 的 `CUdeviceptr` 值指向的是不同的 memory 位置。

在任意时刻，一个 host thread 只能有一个 device context 处于 current 状态。当通过 `cuCtxCreate()` 创建 context 时，该 context 会被设为调用该函数的 host thread 的 current context。所有在 context 中执行的 CUDA 函数（也就是大多数不涉及 device 枚举或 context 管理的函数），如果当前 thread 没有一个有效的 current context，就会返回 `CUDA_ERROR_INVALID_CONTEXT`。

每个 host thread 都维护一个 current context 的栈。`cuCtxCreate()` 会将新创建的 context 压入栈顶。可以调用 `cuCtxPopCurrent()` 将该 context 从 host thread 上解绑。解绑之后，context 处于“floating”状态，可以被推送为任意 host thread 的 current context。`cuCtxPopCurrent()` 同时还会恢复之前的 current context（如果存在）。

每个 context 还维护一个 usage count。`cuCtxCreate()` 创建的 context，其 usage count 初始为 1。`cuCtxAttach()` 会递增 usage count，而 `cuCtxDetach()` 会递减 usage count。当调用 `cuCtxDetach()` 或 `cuCtxDestroy()` 并使 usage count 变为 0 时，context 会被销毁。

driver API 可以与 runtime API 互操作。可以通过 `cuDevicePrimaryCtxRetain()` 从 driver API 访问由 runtime 管理的 primary context（参见 Runtime Initialization）。

usage count 的设计使得第三方代码能够在同一个 context 中协同工作。例如，如果有三个 library 需要使用同一个 context，那么每个 library 都会调用 `cuCtxAttach()` 来递增 usage count，并在使用完成后调用 `cuCtxDetach()` 来递减 usage count。对于大多数 library 而言，通常期望 application 在加载或初始化 library 之前就已经创建好了 context；这样 application 可以使用自己的策略来创建 context，而 library 只需要在传入的 context 上运行即可。那些希望在 API client 不知情的情况下自行创建 context 的 library（API client 可能已经创建了 context，也可能没有），则会像下图所示那样，使用 `cuCtxPushCurrent()` 和 `cuCtxPopCurrent()`。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/library-context-management.png)

*Figure 20* Library Context Management

### 3.3.2. Module

Modules 是由 nvcc 输出的 device 代码和数据的动态加载包（参见 Compilation with NVCC），类似于 Windows 中的 DLL。所有符号的名称，包括 functions、全局变量以及 texture 或 surface references，都在 module 作用域内维护，以便由独立第三方编写的 modules 可以在同一个 CUDA context 中互操作。

以下代码示例加载了一个 module 并获取了某个 kernel 的 handle：

```c
CUmodule cuModule;
// 加载 module 文件
cuModuleLoad(&cuModule, "myModule.ptx");
CUfunction myKernel;
// 从 module 中获取 function handle
cuModuleGetFunction(&myKernel, cuModule, "MyKernel");
```

以下代码示例从 PTX 代码编译并加载一个新的 module，并解析编译错误：

```c
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* PTXCode = "some PTX code";
char error_log[BUFFER_SIZE];
int err;
// 设置错误日志缓冲区
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0] = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1] = (void*)BUFFER_SIZE;
// 根据当前 CUcontext 确定目标架构
options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
values[2] = 0;
// 加载 PTX 数据并进行 JIT 编译
err = cuModuleLoadDataEx(&cuModule, PTXCode, 3, options, values);
if (err != CUDA_SUCCESS) printf("Link error:\n%s\n", error_log);
```

以下代码示例从多个 PTX 代码进行编译、link 并加载一个新的 module，同时解析 link 和编译错误：

```c
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[6];
void* values[6];
float walltime;
char error_log[BUFFER_SIZE], info_log[BUFFER_SIZE];
char* PTXCode0 = "some PTX code";
char* PTXCode1 = "some other PTX code";
CUlinkState linkState;
int err;
void* cubin;
size_t cubinSize;
// 记录 wall time
options[0] = CU_JIT_WALL_TIME;
values[0] = (void*)&walltime;
// 设置信息日志缓冲区
options[1] = CU_JIT_INFO_LOG_BUFFER;
values[1] = (void*)info_log;
options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
values[2] = (void*)BUFFER_SIZE;
// 设置错误日志缓冲区
options[3] = CU_JIT_ERROR_LOG_BUFFER;
values[3] = (void*)error_log;
options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[4] = (void*)BUFFER_SIZE;
// 开启详细日志
options[5] = CU_JIT_LOG_VERBOSE;
values[5] = (void*)1;
// 创建 linker 实例
cuLinkCreate(6, options, values, &linkState);
// 添加第一个 PTX 数据
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)PTXCode0, strlen(PTXCode0) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS) printf("Link error:\n%s\n", error_log);
// 添加第二个 PTX 数据
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)PTXCode1, strlen(PTXCode1) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS) printf("Link error:\n%s\n", error_log);
// 完成 link 过程并生成 cubin
cuLinkComplete(linkState, &cubin, &cubinSize);
printf("Link completed in %fms. Linker Output:\n%s\n", walltime, info_log);
// 将生成的 cubin 加载到 module 中
cuModuleLoadData(cuModule, cubin);
// 销毁 linker 状态
cuLinkDestroy(linkState);
```

通过使用多 threads，可以加速 module link/加载过程的某些部分，包括加载 cubin 时。以下代码示例使用 CU_JIT_BINARY_LOADER_THREAD_COUNT 来提高 module 加载速度。

```c
#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* cubinCode = "some cubin code";
char error_log[BUFFER_SIZE];
int err;
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0] = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1] = (void*)BUFFER_SIZE;
// 使用与机器 CPU 核心数一样多的 threads
options[2] = CU_JIT_BINARY_LOADER_THREAD_COUNT;
values[2] = 0; 
// 加载 cubin 数据
err = cuModuleLoadDataEx(&cuModule, cubinCode, 3, options, values);
if (err != CUDA_SUCCESS) printf("Link error:\n%s\n", error_log);
```

完整代码可以在 ptxjit CUDA 示例中找到。

### 3.3.3. Kernel 执行

`cuLaunchKernel()` 使用给定的执行配置来启动一个 kernel。

参数可以通过两种方式传递： 一种方式是通过一个指针数组（`cuLaunchKernel()` 的倒数第二个参数），其中第 n 个指针对应第 n 个参数，并指向一块内存区域，kernel 参数会从该区域中拷贝；另一种方式是通过extra选项传递（`cuLaunchKernel()` 的最后一个参数）。

当参数通过extra选项传递时（使用 `CU_LAUNCH_PARAM_BUFFER_POINTER` 选项），参数以一个指向单一缓冲区的指针形式传入。该缓冲区中各个参数必须按照 device code 中对应参数类型的对齐要求，彼此正确地进行偏移排列。

device code 中内建向量类型（built-in vector types）的对齐要求列在表 42 中。对于所有其他基础类型，其在 device code 中的对齐要求与 host code 中一致，因此可以通过 `__alignof()` 获得。唯一的例外是：当 host 编译器将 `double` 和 `long long`（以及在 64 位系统上的 `long`）按照单字（one-word）边界对齐，而不是双字（two-word）边界对齐时（例如使用 gcc 的 `-mno-align-double` 编译选项），因为在 device code 中，这些类型始终按照双字边界对齐。

`CUdeviceptr` 是一个整数类型，但它表示的是一个指针，因此它的对齐要求是 `__alignof(void*)`。

下面的代码示例使用了一个宏（`ALIGN_UP()`）来调整每个参数的偏移，使其满足对应的对齐要求；并使用另一个宏（`ADD_TO_PARAM_BUFFER()`）将每个参数依次加入到传递给 `CU_LAUNCH_PARAM_BUFFER_POINTER` 选项的参数缓冲区中。

```c
#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

char paramBuffer[1024];
size_t paramBufferSize = 0;

#define ADD_TO_PARAM_BUFFER(value, alignment)                   \
    do {                                                        \
        paramBufferSize = ALIGN_UP(paramBufferSize, alignment); \
        memcpy(paramBuffer + paramBufferSize,                   \
               &(value), sizeof(value));                        \
        paramBufferSize += sizeof(value);                       \
    } while (0)

int i;
ADD_TO_PARAM_BUFFER(i, __alignof(i));
float4 f4;
ADD_TO_PARAM_BUFFER(f4, 16); // float4 的对齐要求是 16
char c;
ADD_TO_PARAM_BUFFER(c, __alignof(c));
float f;
ADD_TO_PARAM_BUFFER(f, __alignof(f));
CUdeviceptr devPtr;
ADD_TO_PARAM_BUFFER(devPtr, __alignof(devPtr));
float2 f2;
ADD_TO_PARAM_BUFFER(f2, 8); // float2 的对齐要求是 8

void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, paramBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &paramBufferSize,
    CU_LAUNCH_PARAM_END
};
cuLaunchKernel(cuFunction,
               blockWidth, blockHeight, blockDepth,
               gridWidth, gridHeight, gridDepth,
               0, 0, 0, extra);
```

一个结构体的对齐要求等于其所有字段中对齐要求最大的那个字段的对齐要求。因此，包含内建向量类型、`CUdeviceptr`，或者未对齐的 `double` 和 `long long` 的结构体，其在 device code 和 host code 中的对齐要求可能不同。这样的结构体在两端的填充方式也可能不同。

例如，下面这个结构体在 host code 中完全没有填充，但在 device code 中，在字段 `f` 之后会被填充 12 个字节，因为字段 `f4` 的对齐要求是 16。

```c
typedef struct {
    float  f;
    float4 f4;
} myStruct;
```

### 3.3.4. Runtime API 与 Driver API 之间的互操作性

一个应用程序可以在同一代码中混合使用 runtime API 和 driver API。

如果通过 driver API 创建并设为当前的 context，那么后续的 runtime API 调用将会使用这个 context，而不是再创建一个新的 context。

如果 runtime 已经完成初始化，则可以通过 `cuCtxGetCurrent()` 获取在初始化过程中创建的 context。随后，该 context 可以被 driver API 的调用所使用。

runtime 隐式创建的 context 被称为 primary context（参见 Runtime Initialization）。该 context 可以通过 driver API 提供的 Primary Context Management 函数进行管理。

device memory 可以通过任意一种 API 进行分配和释放。`CUdeviceptr` 可以与普通指针相互转换：

```
CUdeviceptr devPtr;
float* d_data;

// 使用 driver API 分配
cuMemAlloc(&devPtr, size);
d_data = (float*)devPtr;

// 使用 runtime API 分配
cudaMalloc(&d_data, size);
devPtr = (CUdeviceptr)d_data;
```

特别地，这意味着：使用 driver API 编写的应用程序可以调用使用 runtime API 编写的库（例如 cuFFT、cuBLAS 等）。

参考手册中 **device management** 和 **version management** 章节中的所有函数都可以在 runtime API 和 driver API 之间互换使用。
