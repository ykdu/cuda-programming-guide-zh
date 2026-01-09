# 2.5 NVCC: The NVIDIA CUDA Compiler

NVIDIA CUDA Compiler `nvcc` 是 NVIDIA 提供的一套用于编译 CUDA C/C++ 以及 PTX code 的工具链。该工具链属于 CUDA Toolkit 的一部分，由多个工具组成，包括 compiler、linker 以及 PTX 和 Cubin assembler。最上层的 `nvcc` 工具负责协调整个编译流程，在每个编译阶段调用合适的工具。

`nvcc` 驱动的是 CUDA code 的离线编译，与 CUDA runtime compiler nvrtc 所驱动的在线或 Just-in-Time（JIT）编译形成对比。

本章介绍构建应用程序时最常用、也是最关键的 `nvcc` 用法和细节。关于 `nvcc` 的完整内容，请参考 nvcc 官方文档。

### 2.5.1 CUDA Source Files 和 Headers

使用 `nvcc` 编译的 source files 可以同时包含在 CPU 上执行的 host code，以及在 GPU 上执行的 device code。`nvcc` 支持常见的 C/C++ source file 扩展名 `.c`、`.cpp`、`.cc`、`.cxx`，这些文件仅包含 host-only code；而扩展名为 `.cu` 的文件用于包含 device code，或者 host 和 device code 的混合内容。包含 device code 的 header 通常使用 `.cuh` 扩展名，以便与仅包含 host-only code 的 header（如 `.h`、`.hpp`、`.hh`、`.hxx` 等）区分开来。

| 文件扩展名                  | 描述              | 内容                                                         |
| --------------------------- | ----------------- | ------------------------------------------------------------ |
| `.c`                        | C source file     | Host-only code                                               |
| `.cpp`, `.cc`, `.cxx`       | C++ source file   | Host-only code                                               |
| `.h`, `.hpp`, `.hh`, `.hxx` | C/C++ header file | Device code、host code，或 host/device 混合 code // 译者：写错了吧 |
| `.cu`                       | CUDA source file  | Device code、host code，或 host/device 混合 code             |
| `.cuh`                      | CUDA header file  | Device code、host code，或 host/device 混合 code             |

### 2.5.2 NVCC Compilation Workflow

在编译的初始阶段，`nvcc` 会将 device code 与 host code 分离，并分别将它们的编译任务交给 GPU compiler 和 host compiler。

为了编译 host code，CUDA compiler `nvcc` 需要系统中存在一个兼容的 host compiler。CUDA Toolkit 为 Linux 和 Windows 平台定义了支持的 host compiler 策略。

仅包含 host code 的文件既可以直接使用 `nvcc` 编译，也可以直接使用 host compiler 编译。生成的 object files 可以在 link 阶段与由 `nvcc` 生成、包含 GPU code 的 object files 一起进行链接。

GPU compiler 会将 C/C++ device code 编译为 PTX assembly code。对于在编译命令行中指定的每一种 virtual machine instruction set architecture（例如 `compute_90`），GPU compiler 都会单独运行一次。

随后，每一份 PTX code 会被传递给 `ptxas` 工具，由它生成面向目标硬件 ISA 的 Cubin。硬件 ISA 通过其 SM version 进行标识。

可以将多个 PTX 和 Cubin target 嵌入到一个应用程序或 library 中的单一 Fatbin container 中，这样同一个 binary 就能够支持多种 virtual ISA 和目标硬件 ISA。

上述各个工具的调用和协调均由 `nvcc` 自动完成。可以使用 `-v` 选项来显示完整的编译流程以及各工具的调用情况；使用 `-keep` 选项可以将编译过程中生成的中间文件保存在当前目录，或者通过 `--keep-dir` 指定的目录中。

下面的示例展示了对一个 CUDA source file `example.cu` 的编译流程：

```
// ----- example.cu -----
#include <stdio.h>

__global__ void kernel() {
    printf("Hello from kernel\n");
}

void kernel_launcher() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

int main() {
    kernel_launcher();
    return 0;
}
```

`nvcc` 的基本编译流程如下所示。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/nvcc-flow.png)

`nvcc` 在同时生成多个 PTX 和 Cubin architecture 时的编译流程如下所示。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/nvcc-flow-multi-archs.png)

关于 `nvcc` 编译流程的更详细说明，请参考 compiler 文档。

### 2.5.3. NVCC Basic Usage

使用 `nvcc` 编译 CUDA source file 的基本命令如下：

```
nvcc <source_file>.cu -o <output_file>
```

`nvcc` 支持常见的 compiler flags，用于指定 include directories（`-I <path>`）、library paths（`-L <path>`）、链接其它 libraries（`-l<library>`），以及定义 macros（`-D<macro>=<value>`）。

```
nvcc example.cu -I path_to_include/ -L path_to_library/ -lcublas -o <output_file>
```

#### 2.5.3.1. NVCC PTX and Cubin Generation

默认情况下，`nvcc` 会为 CUDA Toolkit 所支持的**最早的 GPU architecture**（即最低的 `compute_XY` 和 `sm_XY` 版本）生成 PTX 和 Cubin，以最大化兼容性。

- 可以使用 `-arch` 选项为指定的 GPU architecture 生成 PTX 和 Cubin。
- 可以使用 `-gencode` 选项为多个 GPU architectures 生成 PTX 和 Cubin。

可以通过传入 `--list-gpu-code` 和 `--list-gpu-arch` flags 分别获取支持的 real GPU architectures 和 virtual GPU architectures 的完整列表，或者参考 `nvcc` 文档中的 Virtual Architecture List 和 GPU Architecture List 章节。

```bash
nvcc --list-gpu-code # 列出所有支持的 real GPU architectures
nvcc --list-gpu-arch # 列出所有支持的 virtual GPU architectures
nvcc example.cu -arch=compute_<XY> # 例如 -arch=compute_80，适用于 NVIDIA Ampere GPUs 及之后的架构
                                   # 仅生成 PTX，GPU forward compatible
nvcc example.cu -arch=sm_<XY>      # 例如 -arch=sm_80，适用于 NVIDIA Ampere GPUs 及之后的架构
                                   # 生成 PTX 和 Cubin，GPU forward compatible
nvcc example.cu -arch=native       # 自动检测当前 GPU 并生成对应的 Cubin
                                   # 不生成 PTX，不具备 GPU forward compatibility
nvcc example.cu -arch=all          # 为所有支持的 GPU architectures 生成 Cubin
                                   # 同时包含最新的 PTX，用于 GPU forward compatibility
nvcc example.cu -arch=all-major    # 为所有主要的 GPU architectures 生成 Cubin，例如 sm_80、sm_90
                                   # 同时包含最新的 PTX，用于 GPU forward compatibility
```

更高级的用法允许分别指定 PTX 和 Cubin 的 target：

```bash
# 为 virtual architecture compute_80 生成 PTX，并将其编译为 real architecture sm_86 的 Cubin，同时保留 compute_80 的 PTX
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,compute_80 #（生成 PTX 和 Cubin）

# 为 virtual architecture compute_80 生成 PTX，并将其编译为 real architecture sm_86、sm_89 的 Cubin
nvcc example.cu -arch=compute_80 -gpu-code=sm_86,sm_89      #（不保留 PTX）
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89   # 与上一条命令等价

#（1）为 virtual architecture compute_80 生成 PTX，并将其编译为 real architecture sm_86、sm_89 的 Cubin
#（2）为 virtual architecture compute_90 生成 PTX，并将其编译为 real architecture sm_90 的 Cubin
nvcc example.cu -gencode=arch=compute_80,code=sm_86,sm_89 -gencode=arch=compute_90,code=sm_90
```

关于用于控制 GPU code generation 的 `nvcc` command-line options 的完整说明，可以参考 `nvcc` 文档。

#### 2.5.3.2. Host Code Compilation Notes

不包含 device code 或 device symbols 的 compilation units（即 source file 及其 headers），可以直接使用 host compiler 进行编译。如果某个 compilation unit 使用了 CUDA runtime API functions，则 application 必须链接 CUDA runtime library。CUDA runtime 同时提供 static library 和 shared library，分别为 `libcudart_static` 和 `libcudart`。默认情况下，`nvcc` 会链接 static CUDA runtime library。如果需要使用 shared CUDA runtime library，可以在 compile 或 link 命令中向 `nvcc` 传入 `--cudart=shared` flag。

`nvcc` 允许通过 `-ccbin <compiler>` 参数指定用于 host functions 的 host compiler。也可以通过定义环境变量 `NVCC_CCBIN` 来指定 `nvcc` 使用的 host compiler。`nvcc` 的 `-Xcompiler` 参数可以将 arguments 直接传递给 host compiler。例如，在下面的示例中，`-O3` 参数被 `nvcc` 传递给了 host compiler。

```bash
nvcc example.cu -ccbin=clang++
export NVCC_CCBIN='gcc'
nvcc example.cu -Xcompiler=-O3
```

#### 2.5.3.3. Separate Compilation of GPU Code

`nvcc` 默认采用 whole-program compilation，这要求使用到的所有 GPU code 和 symbols 都必须出现在同一个 compilation unit 中。CUDA device functions 可以调用定义在其它 compilation units 中的 device functions，或者访问其中定义的 device variables，但必须在 `nvcc` command line 中指定 `-rdc=true` 或其别名 `-dc`，以启用来自不同 compilation units 的 device code 的链接。

能够从不同 compilation units 中链接 device code 和 symbols 的能力称为 separate compilation。

Separate compilation 允许更灵活的代码组织方式，可以改善 compile time，并且可能生成更小的 binaries。与 whole-program compilation 相比，separate compilation 在 build 阶段可能会引入一定复杂度。由于 device code linking 可能影响性能，因此默认情况下不会启用。Link-Time Optimization（LTO）可以帮助降低 separate compilation 带来的性能开销。

Separate compilation 需要满足以下条件：

- 在某个 compilation unit 中定义的非 `const` device variables，必须在其它 compilation units 中通过 `extern` 关键字进行引用。
- 所有 `const` device variables 必须通过 `extern` 关键字进行定义和引用。
- 所有 CUDA source files（`.cu`）都必须使用 `-dc` 或 `-rdc=true` flags 进行编译。

Host functions 和 device functions 默认具有 external linkage，因此不需要使用 `extern` 关键字。需要注意的是，从 CUDA 13 开始，`__global__` functions 以及 `__managed__` / `__device__` / `__constant__` variables 默认具有 internal linkage。

在下面的示例中，`definition.cu` 定义了一个 variable 和一个 function，而 `example.cu` 引用了它们。两个文件分别进行编译，随后被链接成最终的 binary。

```c++
// ----- definition.cu -----
extern __device__ int device_variable = 5;
__device__        int device_function() { return 10; }
```

```c++
// ----- example.cu -----
extern __device__ int  device_variable;
__device__        int device_function();

__global__ void kernel(int* ptr) {
    device_variable = 0;
    *ptr            = device_function();
}
```

```bash
nvcc -dc definition.cu -o definition.o
nvcc -dc example.cu    -o example.o
nvcc definition.o example.o -o program
```

### 2.5.4. Common Compiler Options

本节介绍了可与 `nvcc` 一起使用的最相关编译器选项，涵盖语言特性、优化、调试、profiling 以及构建相关内容。所有选项的完整说明可在 nvcc 文档中找到。

#### 2.5.4.1. Language Features

`nvcc` 支持 C++ 核心语言特性，从 C++03 到 C++20。可以使用 `-std` 标志指定要使用的语言标准：

- `--std={c++03|c++11|c++14|c++17|c++20}`

此外，`nvcc` 还支持以下语言扩展：

- `-restrict`：断言所有 kernel 指针参数都是 restrict 指针。
- `-extended-lambda`：允许在 lambda 声明中使用 `__host__` 和 `__device__` 注解。
- `-expt-relaxed-constexpr`：（实验性标志）允许 host 代码调用 `__device__ constexpr` 函数，同时允许 device 代码调用 `__host__ constexpr` 函数。

关于这些特性的更多细节，可以参考 extended lambda 和 constexpr 相关章节。

#### 2.5.4.2. Debugging Options

`nvcc` 支持以下用于生成调试信息的选项：

- `-g`：为 host 代码生成调试信息。`gdb`、`lldb` 等工具依赖这些信息进行 host 代码调试。
- `-G`：为 device 代码生成调试信息。cuda-gdb 依赖这些信息进行 device 代码调试。该标志还会定义宏 `__CUDACC_DEBUG__`。
- `-lineinfo`：为 device 代码生成行号信息。该选项不会影响执行性能，并且在与 compute-sanitizer 工具结合使用时，有助于追踪 kernel 的执行。

`nvcc` 默认对 GPU 代码使用最高优化级别 `-O3`。调试标志 `-G` 会阻止部分编译器优化，因此调试代码的性能通常低于非调试代码。可以定义 `-DNDEBUG` 标志来禁用运行时断言，因为这些断言同样会降低执行性能。

#### 2.5.4.3. Optimization Options

`nvcc` 提供了大量用于性能优化的选项。本节旨在简要介绍一些开发者可能会用到的选项，并提供进一步信息的指引。完整内容可参考 nvcc 文档。

- `-Xptxas`：将参数传递给 PTX 汇编器工具 `ptxas`。nvcc 文档中列出了多个对 `ptxas` 有用的参数。例如，`-Xptxas=-maxrregcount=N` 用于指定每个 thread 可使用的最大寄存器数量。
- `-extra-device-vectorization`：启用更激进的 device 代码向量化。
- 其他用于对浮点行为进行细粒度控制的标志，分别在 Floating-Point Computation 章节以及 nvcc 文档中进行说明。

以下标志可以让编译器输出有助于高级优化的信息：

- `-res-usage`：在编译后打印资源使用报告，其中包括每个 kernel 函数使用的寄存器数量、shared memory、constant memory 和 local memory。
- `-opt-info=inline`：打印有关函数内联的信息。
- `-Xptxas=-warn-lmem-usage`：当使用 local memory 时给出警告。
- `-Xptxas=-warn-spills`：当寄存器溢出到 local memory 时给出警告。

#### 2.5.4.4. Link-Time Optimization (LTO)

分离编译可能会导致性能低于整体程序编译，因为跨文件优化机会有限。链接时优化（LTO）通过在链接阶段对分离编译的文件进行跨文件优化来解决这一问题，但代价是增加编译时间。LTO 能在保持分离编译灵活性的同时，恢复整体程序编译的大部分性能。

要启用 LTO，`nvcc` 需要使用 `-dlto` 标志，或使用 `lto_<SM version>` 作为链接时优化目标：

```bash
nvcc -dc -dlto -arch=sm_100 definition.cu -o definition.o
nvcc -dc -dlto -arch=sm_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program

nvcc -dc -arch=lto_100 definition.cu -o definition.o
nvcc -dc -arch=lto_100 example.cu    -o example.o
nvcc -dlto definition.o example.o -o program
```

#### 2.5.4.5. Profiling Options

可以使用 Nsight Compute 和 Nsight Systems 工具直接对 CUDA 应用进行 profiling，而无需在编译过程中指定额外的标志。不过，`nvcc` 生成的一些附加信息可以通过将源文件与生成的代码关联起来，从而辅助 profiling：

- `-lineinfo`：为 device 代码生成行号信息，使 profiling 工具能够显示源代码。profiling 工具要求原始源代码必须位于与编译时相同的位置。
- `-src-in-ptx`：将原始源代码保留在 PTX 中，从而避免上述 `-lineinfo` 的限制。该选项需要同时指定 `-lineinfo`。

#### 2.5.4.6. Fatbin Compression

`nvcc` 默认会对存储在应用程序或库二进制文件中的 fatbins 进行压缩。可以使用以下选项来控制 fatbin 的压缩行为：

- `-no-compress`：禁用 fatbin 压缩。
- `--compress-mode={default|size|speed|balance|none}`：设置压缩模式。`speed` 侧重于快速解压，`size` 侧重于减小 fatbin 大小，`balance` 在速度和大小之间提供折中。默认模式为 `speed`，`none` 表示不进行压缩。

#### 2.5.4.7. Compiler Performance Controls

`nvcc` 提供了一些选项，用于分析和加速编译过程本身：

- `-t <N>`：在为多个 GPU 架构编译单个编译单元时，用于并行编译的 CPU 线程数量。
- `-split-compile <N>`：用于并行执行优化阶段的 CPU 线程数量。
- `-split-compile-extended <N>`：更激进的分离编译形式，需要链接时优化支持。
- `-Ofc <N>`：device 代码的编译速度级别。
- `-time <filename>`：生成一个 CSV 表，记录各个编译阶段所花费的时间。
- `-fdevice-time-trace`：为 device 代码编译生成时间跟踪信息。
