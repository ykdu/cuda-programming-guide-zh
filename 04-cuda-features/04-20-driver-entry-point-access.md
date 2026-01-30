# 4.20 Driver Entry Point Access

### 4.20.1. 引言

**Driver Entry Point Access APIs** 提供了一种获取 CUDA driver 函数地址的方式。从 CUDA 11.3 开始，用户可以通过这些 API 返回的函数指针来调用可用的 CUDA driver API。

这些 API 提供的功能与其对应平台上的机制类似，例如 POSIX 平台上的 `dlsym` 以及 Windows 上的 `GetProcAddress`。这些 API 允许用户：

- 使用 **CUDA Driver API** 获取 driver 函数的地址。
- 使用 **CUDA Runtime API** 获取 driver 函数的地址。
- 请求 CUDA driver 函数的 **per-thread default stream** 版本。更多细节请参见 *Retrieve Per-thread Default Stream Versions*。
- 在旧版本的 toolkit 上，通过较新的 driver 访问新的 CUDA 特性。

### 4.20.2. 驱动函数类型定义（Typedefs）

为了帮助获取 CUDA Driver API 的入口点，CUDA Toolkit 提供了一组头文件，其中包含所有 CUDA driver API 的函数指针类型定义。这些头文件会随 CUDA Toolkit 一同安装，并位于 toolkit 的 `include/` 目录下。下表总结了每个 CUDA API 头文件所对应的 `typedefs` 头文件。

**表 27：CUDA driver API 的 typedefs 头文件**

| API header file  | API Typedef header file  |
| ---------------- | ------------------------ |
| `cuda.h`         | `cudaTypedefs.h`         |
| `cudaGL.h`       | `cudaGLTypedefs.h`       |
| `cudaProfiler.h` | `cudaProfilerTypedefs.h` |
| `cudaVDPAU.h`    | `cudaVDPAUTypedefs.h`    |
| `cudaEGL.h`      | `cudaEGLTypedefs.h`      |
| `cudaD3D9.h`     | `cudaD3D9Typedefs.h`     |
| `cudaD3D10.h`    | `cudaD3D10Typedefs.h`    |
| `cudaD3D11.h`    | `cudaD3D11Typedefs.h`    |

上述头文件本身并不定义实际的函数指针，而是定义了函数指针的 typedef。例如，`cudaTypedefs.h` 中包含了 driver API `cuMemAlloc` 的如下 typedef 定义：

```
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);
```

CUDA driver 符号采用基于版本的命名方案，除第一个版本外，其名称中都会带有 `_v*` 后缀。当某个 CUDA driver API 的函数签名或语义发生变化时，会递增对应 driver 符号的版本号。

以 `cuMemAlloc` driver API 为例，第一个 driver 符号名为 `cuMemAlloc`，下一个版本的符号名为 `cuMemAlloc_v2`。最初在 CUDA 2.0（2000）中引入的第一个版本，其 typedef 为 `PFN_cuMemAlloc_v2000`。在 CUDA 3.2（3020）中引入的下一个版本，其 typedef 为 `PFN_cuMemAlloc_v3020`。

这些 `typedefs` 可以用于在代码中更方便地定义合适类型的函数指针，例如：

```
PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;
```

如果用户关注的是某个特定版本的 API，上述方法是更推荐的做法。此外，这些头文件还为在 CUDA Toolkit 发布时可用的最新版本 driver 符号预定义了宏；这些 typedef 不包含 `_v*` 后缀。

对于 CUDA 11.3 toolkit，`cuMemAlloc_v2` 是当时的最新版本，因此也可以如下定义其函数指针：

```
PFN_cuMemAlloc pfn_cuMemAlloc;
```

### 4.20.3. 驱动函数获取

通过使用 **Driver Entry Point Access APIs** 以及相应的 typedef，可以获取任意 CUDA driver API 的函数指针。

#### 4.20.3.1. 使用 Driver API

Driver API 需要将 CUDA 版本作为参数传入，以获取与所请求 driver 符号 ABI 兼容的版本。CUDA Driver API 为每个函数定义了带有 `_v*` 后缀的 ABI 版本。

例如，下面展示了 `cuStreamBeginCapture` 的不同版本，以及它们在 `cudaTypedefs.h` 中对应的 typedef 定义：

```c
// cuda.h
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);

// cudaTypedefs.h
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);
```

从上述代码片段中的 typedef 可以看出，版本后缀 `_v10000` 和 `_v10010` 分别表示这些 API 是在 CUDA 10.0 和 CUDA 10.1 中引入的。

```c
#include <cudaTypedefs.h>

// Declare the entry points for cuStreamBeginCapture
PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;

// Get the function pointer to the cuStreamBeginCapture driver symbol
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
// Get the function pointer to the cuStreamBeginCapture_v2 driver symbol
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
```

参考上述代码示例，如果要获取 driver API `cuStreamBeginCapture` 的 `_v1` 版本地址，CUDA 版本参数必须精确指定为 10.0（10000）。类似地，如果要获取 `_v2` 版本的地址，则 CUDA 版本应指定为 10.1（10010）。

为获取某个 driver API 的特定版本而指定更高的 CUDA 版本并不总是具有可移植性。例如，在这里如果使用 11030，仍然会返回 `_v2` 符号；但如果在 CUDA 11.3 中假设发布了 `_v3` 版本，当与 CUDA 11.3 driver 配合使用时，`cuGetProcAddress` API 将开始返回更新的 `_v3` 符号。

由于 `_v2` 与 `_v3` 符号的 ABI 和函数签名可能不同，使用为 `_v2` 符号设计的 `_v10010` typedef 去调用 `_v3` 函数将会导致未定义行为。

为了获取给定 CUDA Toolkit 中某个 driver API 的最新版本，也可以将 `CUDA_VERSION` 作为 `version` 参数传入，并使用未带版本后缀的 typedef 来定义函数指针。

由于在 CUDA 11.3 中，`cuStreamBeginCapture` driver API 的最新版本是 `_v2`，下面的代码展示了另一种获取方式：

```c
// Assuming we are using CUDA 11.3 Toolkit
#include <cudaTypedefs.h>

// Declare the entry point
PFN_cuStreamBeginCapture pfn_cuStreamBeginCapture_latest;

// Initialize the entry point. Specifying CUDA_VERSION will give the function pointer to the
// cuStreamBeginCapture_v2 symbol since it is the latest version on CUDA 11.3.
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_latest, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
```

需要注意的是，请求一个带有无效 CUDA 版本的 driver API 将返回错误 `CUDA_ERROR_NOT_FOUND`。在上述代码示例中，传入小于 10000（CUDA 10.0）的版本号都是无效的。

#### 4.20.3.2. 使用 Runtime API

Runtime API `cudaGetDriverEntryPoint` 使用 CUDA runtime 版本来获取与所请求 driver 符号 ABI 兼容的版本。在下面的代码示例中，所需的最小 CUDA runtime 版本为 CUDA 11.2，因为 `cuMemAllocAsync` 是在该版本中引入的。

```c
#include <cudaTypedefs.h>

// Declare the entry point
PFN_cuMemAllocAsync pfn_cuMemAllocAsync;

// Initialize the entry point. Assuming CUDA runtime version >= 11.2
cudaGetDriverEntryPoint("cuMemAllocAsync", &pfn_cuMemAllocAsync, cudaEnableDefault, &driverStatus);

// Call the entry point
if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
    pfn_cuMemAllocAsync(...);
}
```

Runtime API `cudaGetDriverEntryPointByVersion` 使用用户提供的 CUDA 版本来获取与所请求 driver 符号 ABI 兼容的版本。这使得用户可以对所请求的 ABI 版本进行更加精确的控制。

#### 4.20.3.3. 获取 Per-thread Default Stream 版本

某些 CUDA driver API 可以被配置为具有 **default stream** 或 **per-thread default stream** 语义。具有 **per-thread default stream** 语义的 Driver API，其名称中会带有 `_ptsz` 或 `_ptds` 后缀。例如，`cuLaunchKernel` 的 per-thread default stream 版本命名为 `cuLaunchKernel_ptsz`。通过 **Driver Entry Point Access APIs**，用户可以请求获取 driver API `cuLaunchKernel` 的 **per-thread default stream** 版本，而不是 default stream 版本。将 CUDA driver API 配置为 **default stream** 或 **per-thread default stream** 语义会影响其同步行为。更多细节可参见此处。

driver API 的 default stream 或 per-thread default stream 版本可以通过以下方式之一获得：

- 使用编译选项 `--default-stream per-thread`，或定义宏 `CUDA_API_PER_THREAD_DEFAULT_STREAM`，以获得 per-thread default stream 行为。
- 通过以下标志强制指定 default stream 或 per-thread default stream 行为：
  - `CU_GET_PROC_ADDRESS_LEGACY_STREAM` / `cudaEnableLegacyStream`
  - `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM` / `cudaEnablePerThreadDefaultStream`
     分别对应 default stream 和 per-thread default stream 行为。

#### 4.20.3.4. 访问新的 CUDA 特性

通常建议安装最新的 CUDA toolkit 以访问新的 CUDA driver 特性。但如果由于某些原因，用户不希望升级或无法使用最新的 CUDA toolkit，则可以仅通过更新 CUDA driver，并使用这些 API 来访问新的 CUDA 特性。

为了说明这一用法，假设用户正在使用 CUDA 11.3，并希望使用在 CUDA 12.0 driver 中引入的新 driver API `cuFoo`。下面的代码示例展示了这一使用场景：

```c
int main()
{
    // Assuming we have CUDA 12.0 driver installed.

    // Manually define the prototype as cudaTypedefs.h in CUDA 11.3 does not have the cuFoo typedef
    typedef CUresult (CUDAAPI *PFN_cuFoo)(...);
    PFN_cuFoo pfn_cuFoo = NULL;
    CUdriverProcAddressQueryResult driverStatus;

    // Get the address for cuFoo API using cuGetProcAddress. Specify CUDA version as
    // 12000 since cuFoo was introduced then or get the driver version dynamically
    // using cuDriverGetVersion
    int driverVersion;
    cuDriverGetVersion(&driverVersion);
    CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

    if (status == CUDA_SUCCESS && pfn_cuFoo) {
        pfn_cuFoo(...);
    }
    else {
        printf("Cannot retrieve the address to cuFoo - driverStatus = %d. Check if the latest driver for CUDA 12.0 is installed.\n", driverStatus);
        assert(0);
    }

    // rest of code here

}
```

### 4.20.4. 使用 cuGetProcAddress 的潜在影响

下面给出了一组关于使用 `cuGetProcAddress` 和 `cudaGetDriverEntryPoint` 可能带来问题的具体示例和理论示例。

#### 4.20.4.1. cuGetProcAddress 与隐式链接的影响差异

`cuDeviceGetUuid` 在 CUDA 9.2 中引入。该 API 在 CUDA 11.4 中引入了一个更新版本 `cuDeviceGetUuid_v2`。为了保持次版本兼容性，`cuDeviceGetUuid` 在 `cuda.h` 中直到 CUDA 12.0 才被版本升级为 `cuDeviceGetUuid_v2`。这意味着，通过 `cuGetProcAddress` 获取该 API 的函数指针并调用，可能会表现出不同的行为。

下面是直接使用该 API 的示例：

```
#include <cuda.h>

CUuuid uuid;
CUdevice dev;
CUresult status;

status = cuDeviceGet(&dev, 0); // Get device 0
// handle status

status = cuDeviceGetUuid(&uuid, dev); // Get uuid of device 0
```

在该示例中，假设用户使用 CUDA 11.4 进行编译。注意，这里执行的是 `cuDeviceGetUuid` 的行为，而不是 `_v2` 版本。

下面给出使用 `cuGetProcAddress` 的示例：

```
#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // Get device 0
// handle status

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if (CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
    // pfn_cuDeviceGetUuid points to ???
}
```

在该示例中，假设用户使用 CUDA 11.4 进行编译。此时获取到的是 `cuDeviceGetUuid_v2` 的函数指针。调用该函数指针将会执行新的 `_v2` 版本函数，而不是前一个示例中所示的 `cuDeviceGetUuid` 行为。

#### 4.20.4.2. cuGetProcAddress 中编译期与运行期版本使用差异

我们在前一个问题的基础上做一个小的改动。上一个示例中使用的是编译期常量 `CUDA_VERSION` 来决定获取哪个函数指针。如果用户改为在运行期通过 `cuDriverGetVersion` 或 `cudaDriverGetVersion` 动态查询 driver 版本，并将该版本传递给 `cuGetProcAddress`，则会引入更多复杂情况。示例如下：

```
#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // Get device 0
// handle status

status = cuDriverGetVersion(&cudaVersion);
// handle status

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuid;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if (CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
    // pfn_cuDeviceGetUuid points to ???
}
```

在该示例中，假设用户使用 CUDA 11.3 进行编译。用户会基于已知行为（获取 `cuDeviceGetUuid`，而非 `_v2` 版本）来调试、测试并部署该应用。由于 CUDA 保证了次版本之间的 ABI 兼容性，该应用在 driver 升级到 CUDA 11.4 后（不更新 toolkit 和 runtime）仍然应该能够运行，且无需重新编译。

然而，此时行为将变为未定义。这是因为 `PFN_cuDeviceGetUuid` 仍然对应原始版本的函数签名，但由于 `cudaVersion` 已变为 11040（CUDA 11.4），`cuGetProcAddress` 将返回 `_v2` 版本的函数指针。这意味着使用原始 typedef 调用该函数，可能会产生未定义行为。

在这种情况下，原始（非 `_v2`）版本的 typedef 定义如下：

```
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);
```

而 `_v2` 版本的 typedef 定义如下：

```
typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);
```

因此，在该示例中，API/ABI 本身是相同的，运行期调用通常不会导致直接的问题——唯一可能的差异在于返回的 uuid 内容不可预期。在 **Implications to API/ABI** 一节中，将讨论一个更具问题性的 API/ABI 兼容性案例。

#### 4.20.4.3. 带显式版本检查的 API 版本升级

上面给出了一个具体的示例。现在我们再看一个理论示例，该示例在不同 driver 版本之间仍然存在兼容性问题。例如：

```c
CUresult cuFoo(int bar);              // Introduced in CUDA 11.4
CUresult cuFoo_v2(int bar);           // Introduced in CUDA 11.5
CUresult cuFoo_v3(int bar, void* jazz); // Introduced in CUDA 11.6

typedef CUresult (CUDAAPI *PFN_cuFoo_v11040)(int bar);
typedef CUresult (CUDAAPI *PFN_cuFoo_v11050)(int bar);
typedef CUresult (CUDAAPI *PFN_cuFoo_v11060)(int bar, void* jazz);
```

可以看到，自 CUDA 11.4 初始引入以来，该 API 已被修改了两次，而最新的 CUDA 11.6 还修改了该函数的 API/ABI 接口。下面给出一个针对 CUDA 11.5 编译的用户代码使用示例：

```c
#include <cuda.h>
#include <cudaTypedefs.h>

CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDriverGetVersion(&cudaVersion);
// handle status

PFN_cuFoo_v11040 pfn_cuFoo_v11040;
PFN_cuFoo_v11050 pfn_cuFoo_v11050;

if (cudaVersion < 11050) {
    // We know to get the CUDA 11.4 version
    status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11040, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    // Handle status and validating pfn_cuFoo_v11040
}
else {
    // Assume >= CUDA 11.5 version we can use the second version
    status = cuGetProcAddress("cuFoo", &pfn_cuFoo_v11050, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
    // Handle status and validating pfn_cuFoo_v11050
}
```

在该示例中，如果应用没有针对 CUDA 11.6 新增的 typedef 进行更新，也没有使用这些新 typedef 重新编译并补充对应的分支处理逻辑，那么应用将会获取到 `cuFoo_v3` 的函数指针。任何对该函数的调用都会导致未定义行为。

该示例的目的在于说明：即便对 `cuGetProcAddress` 进行了显式版本检查，也无法安全地覆盖 CUDA 主版本内部的次版本升级。

#### 4.20.4.4. 使用 Runtime API 的问题

前面的示例主要聚焦于使用 Driver API 获取 driver API 函数指针时可能出现的问题。现在我们讨论使用 Runtime API（`cudaApiGetDriverEntryPoint`）时可能存在的潜在问题。

我们从一个与前面类似的 Runtime API 示例开始：

```c
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

CUresult status;
cudaError_t error;
int driverVersion, runtimeVersion;
CUdriverProcAddressQueryResult driverStatus;

// Ask the runtime for the function
PFN_cuDeviceGetUuidRuntime pfn_cuDeviceGetUuidRuntime;
error = cudaGetDriverEntryPoint("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault);
if (cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
    // pfn_cuDeviceGetUuidRuntime points to ???
}
```

该示例中的函数指针情况比前面仅使用 driver 的示例更加复杂，因为无法控制要获取的是哪个版本的函数；它始终会返回当前 CUDA Runtime 版本所对应的 API。更多信息请参见下表：

|                          | Static Runtime Version Linkage |
| ------------------------ | ------------------------------ |
| Driver Version Installed | V11.3                          |
| V11.3                    | v1                             |
| V11.4                    | v1                             |

```
V11.3 => 11.3 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
V11.4 => 11.4 CUDA Runtime and Toolkit (includes header files cuda.h and cudaTypedefs.h)
v1    => cuDeviceGetUuid
v2    => cuDeviceGetUuid_v2
x     => Implies the typedef function pointer won't match the returned function pointer.
         In these cases, the typedef at compile time using a CUDA 11.4 runtime would
         match the _v2 version, but the returned function pointer would be the original
         (non _v2) function.
```

表中所示的问题出现在使用较新的 CUDA 11.4 Runtime 和 Toolkit、但 driver 较旧（CUDA 11.3）的组合情况下，在上表中标记为 **v1x**。在这种组合中，driver 返回的是旧版本函数（非 `_v2`），而应用中使用的 typedef 却是新版本函数指针类型。

#### 4.20.4.5. Runtime API 与动态版本选择的问题

当我们考虑一个应用在不同的 CUDA 版本组合下运行时，会出现更多复杂情况：包括应用编译时使用的 CUDA 版本、CUDA runtime 版本，以及应用动态链接到的 CUDA driver 版本之间的不同组合。

```c
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

CUresult status;
cudaError_t error;
int driverVersion, runtimeVersion;
CUdriverProcAddressQueryResult driverStatus;
enum cudaDriverEntryPointQueryResult runtimeStatus;

PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriver;
status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriver, CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriver) {
    // pfn_cuDeviceGetUuidDriver points to ???
}

// Ask the runtime for the function
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidRuntime;
error = cudaGetDriverEntryPoint ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidRuntime, cudaEnableDefault, &runtimeStatus);
if(cudaSuccess == error && pfn_cuDeviceGetUuidRuntime) {
    // pfn_cuDeviceGetUuidRuntime points to ???
}

// Ask the driver for the function based on the driver version (obtained via runtime)
error = cudaDriverGetVersion(&driverVersion);
PFN_cuDeviceGetUuid pfn_cuDeviceGetUuidDriverDriverVer;
status = cuGetProcAddress ("cuDeviceGetUuid", &pfn_cuDeviceGetUuidDriverDriverVer, driverVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuidDriverDriverVer) {
    // pfn_cuDeviceGetUuidDriverDriverVer points to ???
}
```

下面给出预期的函数指针矩阵：

> 说明：图片中的矩阵（3=> CUDA 11.3，4=> CUDA 11.4）这里用 Markdown 复刻。
>  行：不同获取方式的函数指针变量；列：`应用编译版本/Runtime 动态链接版本/Driver 版本`。

| Function Pointer                     | 3/3/3 | 3/3/4     | 3/4/3 | 3/4/4     | 4/3/3 | 4/3/4 | 4/4/3 | 4/4/4 |
| ------------------------------------ | ----- | --------- | ----- | --------- | ----- | ----- | ----- | ----- |
| `pfn_cuDeviceGetUuidDriver`          | t1/v1 | t1/v1     | t1/v1 | t1/v1     | N/A   | N/A   | t2/v1 | t2/v2 |
| `pfn_cuDeviceGetUuidRuntime`         | t1/v1 | t1/v1     | t1/v1 | **t1/v2** | N/A   | N/A   | t2/v1 | t2/v2 |
| `pfn_cuDeviceGetUuidDriverDriverVer` | t1/v1 | **t1/v2** | t1/v1 | **t1/v2** | N/A   | N/A   | t2/v1 | t2/v2 |

```
tX -> Typedef version used at compile time
vX -> Version returned/used at runtime
```

如果应用是基于 CUDA Version 11.3 编译的，它会使用原始函数的 typedef；但如果基于 CUDA Version 11.4 编译，它会使用 `_v2` 函数的 typedef。正因如此，请注意这里出现了多种 “typedef 与实际返回/使用的版本不匹配” 的情况。

#### 4.20.4.6. Runtime API 允许指定 CUDA 版本带来的问题

除非另有说明，CUDA runtime API `cudaGetDriverEntryPointByVersion` 会具有与 driver 入口点 `cuGetProcAddress` 类似的影响，因为它允许用户请求一个特定的 CUDA driver 版本。

#### 4.20.4.7. 对 API/ABI 的影响

在上面使用 `cuDeviceGetUuid` 的示例中，API 不匹配带来的影响相对较小，而且对许多用户来说可能并不明显，因为 `_v2` 是为支持 Multi-Instance GPU (MIG) mode 而添加的。因此，在一个没有 MIG 的系统上，用户甚至可能不会意识到自己拿到的是一个不同的 API。

更成问题的是那种会改变函数调用签名（从而改变 ABI）的 API，例如 `cuCtxCreate`。`_v2` 版本是在 CUDA 3.2 中引入的，目前在使用 `cuda.h` 时作为默认的 `cuCtxCreate` 使用，但现在在 CUDA 11.4 中又引入了一个更新的版本 `cuCtxCreate_v3`。该 API 的签名也发生了修改，现在会额外接收一些参数。因此，在上面某些情况下，当函数指针的 typedef 与返回的函数指针不匹配时，就可能出现不明显的 ABI 不兼容，从而导致未定义行为。

例如，假设下面这段代码是基于 CUDA 11.3 toolkit 编译的，但系统安装了 CUDA 11.4 driver：

```c
PFN_cuCtxCreate cuUnknown;
CUdriverProcAddressQueryResult driverStatus;

status = cuGetProcAddress("cuCtxCreate", (void**)&cuUnknown, cudaVersion, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
if(CUDA_SUCCESS == status && cuUnknown) {
    status = cuUnknown(&ctx, 0, dev);
}
```

在这段代码中，如果 `cudaVersion` 被设置为任何 `>= 11040` 的值（表示 CUDA 11.4），那么就可能因为没有为 `cuCtxCreate_v3` 版本的 `cuCtxCreate_v3` API 充分提供所需参数而产生未定义行为。

### 4.20.5. 确定 cuGetProcAddress 失败原因

在使用 `cuGetProcAddress` 时，错误主要分为两类：(1) API / 使用方式错误，以及 (2) 无法找到所请求的 driver API。

第一类错误会通过 `CUresult` 返回值从 API 中直接返回。例如，将 `pfn` 变量传入 `NULL`，或者传入非法的 `flags`，都会导致这一类错误。

第二类错误会编码在 `CUdriverProcAddressQueryResult *symbolStatus` 中，用于帮助区分 driver 无法找到所请求符号时的具体原因。

下面给出一个示例：

```c
// cuDeviceGetExecAffinitySupport was introduced in release CUDA 11.4
#include <cuda.h>
CUdriverProcAddressQueryResult driverStatus;
cudaVersion = ...;
status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
if (CUDA_SUCCESS == status) {
    if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
        printf("We can use the new feature when you upgrade cudaVersion to 11.4, but CUDA driver is good to go!\n");
        // Indicating cudaVersion was < 11.4 but run against a CUDA driver >= 11.4
    }
    else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
        printf("Please update both CUDA driver and cudaVersion to at least 11.4 to use the new feature!\n");
        // Indicating driver is < 11.4 since string not found, doesn't matter what cudaVersion was
    }
    else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
        printf("You're using cudaVersion and CUDA driver >= 11.4, using new feature!\n");
        pfn();
    }
}
```

当返回码为 `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` 时，表示在 CUDA driver 中能够找到该 `symbol`，但该 API 是在所提供的 `cudaVersion` 之后才引入的。

在上述示例中，如果 `cudaVersion` 为 11030 或更低，并且程序运行在 CUDA driver ≥ 11.4 的环境中，就会得到 `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`。这是因为 `cuDeviceGetExecAffinitySupport` 是在 CUDA 11.4（11040）中引入的。

当返回码为 `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND` 时，表示在 CUDA driver 中搜索该 `symbol` 时未找到。这种情况可能由多种原因导致，例如：driver 版本过旧，不支持该 CUDA API，或者只是字符串拼写错误。

对于后一种情况，类似于上一个示例，如果用户将 `symbol` 写成 `CUdeviceGetExecAffinitySupport`（注意字符串开头的大写 `CU`），由于字符串不匹配，`cuGetProcAddress` 将无法找到对应的 API。

对于前一种情况，一个典型示例是：用户在开发时基于支持新 API 的 CUDA driver 进行开发，但在部署时运行在较旧的 CUDA driver 上。例如，开发者在 CUDA 11.4 或更新版本的环境中开发应用，`cuGetProcAddress` 在开发阶段是成功的；但当应用部署到仅安装 CUDA 11.3 driver 的系统上运行时，该调用将不再成功，并且 `driverStatus` 中会返回 `CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`。
