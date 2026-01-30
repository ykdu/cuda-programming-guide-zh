# 4.7 Lazy Loading

### 4.7.1. Introduction

Lazy loading 通过等到真正需要时才加载 CUDA modules，来减少程序初始化时间。对于那种只会用到其包含的少量 kernels 的程序（使用 libraries 时很常见），lazy loading 尤其有效。只要遵循 CUDA programming model，lazy loading 的设计目标就是让用户几乎感知不到。Potential Hazards 会更详细地说明这一点。

从 CUDA 12.3 开始，所有平台默认启用 lazy Loading，但可以通过 `CUDA_MODULE_LOADING` 环境变量进行控制。

### 4.7.2. Change History

表 17 按 CUDA Version 列出的 Lazy Loading 变更

| CUDA Version | Change                                         |
| ------------ | ---------------------------------------------- |
| 12.3         | Lazy loading 性能改进。现在 Windows 默认启用。 |
| 12.2         | Linux 默认启用 lazy loading。                  |
| 11.7         | 首次引入 lazy loading，默认关闭。              |

### 4.7.3. Requirements for Lazy Loading

Lazy loading 是 CUDA runtime 和 driver 的联合特性。只有在 runtime 和 driver 都满足版本要求时，lazy loading 才可用。

#### 4.7.3.1. CUDA Runtime Version Requirement

从 CUDA runtime 11.7 开始支持 lazy loading。由于 CUDA runtime 通常会被静态链接进程序和 libraries，只有来自或使用 CUDA 11.7+ toolkit 编译的程序与 libraries 才能从 lazy loading 中获益。使用更旧 CUDA runtime versions 编译出来的 libraries 会把所有 modules 都按 eager 的方式加载。

#### 4.7.3.2. CUDA Driver Version Requirement

lazy loading 需要 driver version 515 或更新版本。即使使用 CUDA toolkit 11.7 或更新版本，driver versions 低于 515 时也无法使用 lazy loading。

#### 4.7.3.3. Compiler Requirements

lazy loading 不需要任何 compiler 支持。使用 pre-11.7 compilers 编译出来的 SASS 和 PTX，也都可以在启用 lazy loading 的情况下被加载，并且能获得该特性的完整收益。不过，仍然需要 11.7+ 的 CUDA runtime（如上所述）。

#### 4.7.3.4. Kernel Requirements

lazy loading 不会影响包含 managed variables 的 modules，这类 modules 仍会以 eager 的方式加载。

### 4.7.4. Usage

#### 4.7.4.1. Enabling & Disabling

把 `CUDA_MODULE_LOADING` 环境变量设为 `LAZY` 即可启用 lazy loading。把 `CUDA_MODULE_LOADING` 环境变量设为 `EAGER` 即可关闭 lazy loading。从 CUDA 12.3 开始，所有平台默认启用 lazy loading。

#### 4.7.4.2. Checking if Lazy Loading is Enabled at Runtime

在 CUDA driver API 中，可以用 `cuModuleGetLoadingMode` API 来判断是否启用了 lazy loading。注意：在运行这个函数之前必须先初始化 CUDA。下面的 snippet 展示了一个用法示例。

```c
#include "<cuda.h>"
#include "<assert.h>"
#include "<iostream>"

int main() {
        CUmoduleLoadingMode mode;
        assert(CUDA_SUCCESS == cuInit(0));
        assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

        std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

        return 0;
}
```

#### 4.7.4.3. Forcing a Module to Load Eagerly at Runtime

kernels 和 variables 的加载会自动发生，不需要显式加载。即使不执行 kernels，也可以通过下面方式显式触发它们被加载：

- 调用 `cuModuleGetFunction()` 会导致一个 module 被加载到 device memory 中
- 调用 `cudaFuncGetAttributes()` 会导致一个 kernel 被加载到 device memory 中

> Note
>
> `cuModuleLoad()` 不保证 module 会立刻被加载。

### 4.7.5. Potential Hazards

lazy loading 的设计目标是：使用它不应该要求应用做任何修改。不过，仍然有一些注意点，尤其是在应用没有完全遵循 CUDA programming model 的情况下，如下所述。

#### 4.7.5.1. Impact on Concurrent Kernel Execution

有些程序会错误地假设 concurrent kernel execution 是有保证的。如果需要 cross-kernel synchronization，但 kernel execution 被序列化了，就可能发生 deadlock。为了尽量减少 lazy loading 对 concurrent kernel execution 的影响，可以这样做：

- 在 launch 之前先 preload 所有希望并发执行的 kernels，或者
- 以 `CUDA_MODULE_LOADING = EAGER` 运行应用，强制数据按 eager 方式加载。而 function 本身就是按需加载的

#### 4.7.5.2. Large Memory Allocations

lazy loading 会把 CUDA modules 的 memory allocation 从程序初始化阶段推迟到更接近执行阶段。如果应用在启动时就分配了全部 VRAM，CUDA 可能在运行时为 modules 分配内存失败。可选的解决方案包括：

- 使用 `cudaMallocAsync()`，而不是那种在启动时就分配全部 VRAM 的 allocator
- 预留一些 buffer，用来补偿 kernels 延迟加载带来的额外占用
- 在尝试初始化 allocator 之前，先 preload 程序会用到的所有 kernels

#### 4.7.5.3. Impact on Performance Measurements

lazy loading 可能会把 CUDA module initialization 移到被测量的执行窗口里，从而让性能测量结果出现偏差。为避免这种情况：

- 在测量前至少做一次 warmup iteration
- 在 launch 之前先 preload 被 benchmark 的 kernel
