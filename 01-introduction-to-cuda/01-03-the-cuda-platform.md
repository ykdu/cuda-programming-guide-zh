# 1.3 The CUDA platform

> 原文：CUDA Programming Guide v13.1  
> https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

TODO: 翻译内容





NVIDIA CUDA platform 由许多软件和硬件组件组成，并包含许多关键技术，这些技术的开发旨在支持异构系统上的计算。本章介绍 CUDA platform 的一些基本概念和组件，这些内容对应用开发者理解 CUDA 非常重要。与 *Programming Model* 一样，本章不特定于任何编程语言，而适用于所有使用 CUDA platform 的场景。

#### 1.3.1. 计算能力和 SM 版本

每个 NVIDIA GPU 都有一个计算能力（CC）编号，表示该 GPU 支持哪些特性，并指定该 GPU 的某些硬件参数。具体规格记录在 *Section 5.1* 附录中。所有 NVIDIA GPUs 及其计算能力的列表维护在 CUDA GPU 计算能力页面上。

计算能力通过主版本号和次版本号的格式 *X.Y* 表示，其中 *X* 是主版本号，*Y* 是次版本号。例如，CC 12.0 的主版本号是 12，次版本号是 0。计算能力直接对应于 SM 的版本号。例如，具有 CC 12.0 的 GPU 中的 SMs 具有 SM 版本 ***sm_120***。该版本用于标记二进制文件。

*Section 5.1.1* 显示了如何查询和确定系统中 GPU 的计算能力。

#### 1.3.2. CUDA Toolkit 和 NVIDIA Driver

**NVIDIA Driver 可以被看作是 GPU 的操作系统**。NVIDIA Driver 是一个必须安装在主机操作系统上的软件组件，对于所有 GPU 的使用（包括显示和图形功能）都是必要的。NVIDIA Driver 是 CUDA platform 的基础。除了 CUDA 外，NVIDIA Driver 还提供其它使用 GPU 的方法，例如 Vulkan 和 Direct3D。NVIDIA Driver 有如 *r580* 这样的版本号。

**CUDA Toolkit 是用于编写、构建和分析利用 GPU 计算的软件的库、头文件和工具的集合。**CUDA Toolkit 是与 NVIDIA driver 分离的软件产品。

CUDA runtime 是由 CUDA Toolkit 提供的一个库的特殊情况。CUDA runtime 提供了一个 API 和一些语言扩展，用于处理常见任务，如分配内存、在 GPUs 或 GPU 和 CPUs 之间复制数据以及启动 kernels。CUDA runtime 的 API 部分称为 CUDA runtime API。

CUDA Compatibility 文档提供了不同 GPU、NVIDIA Drivers 和 CUDA Toolkit 版本之间兼容性的完整细节。

##### 1.3.2.1. CUDA Runtime API 和 CUDA Driver API

CUDA runtime API 构建在名为 CUDA driver API 的更底层 API 之上，该 API 由 NVIDIA Driver 暴露。本指南重点介绍 CUDA runtime API 暴露的 API。如果需要，也可以仅使用 driver API 实现相同的功能。有些特性仅能通过 driver API 获取。应用程序可以使用任意一个 API 或者两者互操作。*Section The CUDA Driver API* 讲解了 runtime 和 driver API 之间的互操作。

CUDA runtime API 函数的完整 API 参考可在 *CUDA Runtime API Documentation* 中找到。

CUDA driver API 的完整 API 参考可在 *CUDA Driver API Documentation* 中找到。

#### 1.3.3. PTX

CUDA platform 的一个基本但有时不显眼的层是PTX ISA。PTX 是一种为 NVIDIA GPUs 设计的高级汇编语言。PTX 提供了一个抽象层，覆盖真实 GPU 硬件的物理 ISA。像其它平台一样，应用程序可以直接用这种汇编语言编写，尽管这样做会给软件开发增加不必要的复杂性和难度。

为高级语言设计的领域特定语言和编译器可以将代码生成为 PTX 作为中间表示（IR），然后使用 NVIDIA 的 AOT 或 JIT 编译工具生成可执行的 GPU 二进制代码。这使得 CUDA platform 可以从除了 NVIDIA 提供的工具（如 *NVCC: The NVIDIA CUDA Compiler*）支持的语言之外编程。

由于 GPU 能力随时间变化，PTX 虚拟 ISA 规范是有版本的。**PTX 版本，像 SM 版本一样，对应于计算能力。**例如，支持计算能力 8.0 所有特性的 PTX 称为 ***compute_80***。

有关 PTX 的完整文档可在 *PTX ISA* 中找到。

#### 1.3.4. Cubins 和 Fatbins

CUDA 应用程序和库通常用如 C++ 这样的高级语言编写。该高级语言被编译为 PTX，然后 PTX 被编译为物理 GPU 的真实二进制代码，称为 CUDA binary，简称 *cubin*。cubin 对特定 SM 版本（如 *sm_120*）具有特定的二进制格式。

使用 GPU 计算的可执行文件和库二进制文件包含 CPU 和 GPU 代码。GPU 代码存储在名为 *fatbin* 的容器中。Fatbins 可以包含多种不同目标的 cubins 和 PTX。例如，应用程序可以构建具有多个不同 GPU 架构（即不同 SM 版本）的二进制文件。当应用程序运行时，它的 GPU 代码会加载到特定的 GPU 上，并从 fatbin 中选择最佳二进制代码。

![](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/fatbin.png)

*图 8 可执行文件或库的二进制文件包含 CPU 二进制代码和用于 GPU 代码的 fatbin 容器。fatbin 可以同时包含 cubin GPU 二进制代码和 PTX 虚拟 ISA 代码。PTX 代码可以为未来的目标进行 JIT 编译。*

Fatbins 还可以同时包含多个版本的 PTX code。*图 8* 显示了一个包含多个 cubin 版本的 GPU 代码和一个 PTX 代码版本的应用程序或库二进制文件示例。

##### 1.3.4.1. CUBIN 兼容性

NVIDIA GPUs 在某些情况下保证二进制兼容性。具体而言，在计算能力的主版本号范围内，具有较高或等于目标 cubin 版本的 minor 计算能力的 GPUs 可以加载并执行该 cubin。例如，如果一个应用程序包含为计算能力 8.6 编译的 cubin，那么该 cubin 可以在计算能力为 8.6 或 8.9 的 GPUs 上加载和执行。然而，无法在计算能力为 8.0 的 GPUs 上加载该 cubin，因为 GPU 的 CC minor 版本 0 低于该代码的 minor 版本 6。

NVIDIA GPUs 之间的主计算能力版本之间是二进制不兼容的。也就是说，为计算能力 8.6 编译的 cubin 代码不能在计算能力为 9.0 的 GPUs 上加载。

在讨论二进制代码时，二进制代码通常被称为具有如 *sm_86* 的版本号。就像上面例子中所说的，这意味着该二进制文件是为计算能力 8.6 构建的。这个简写通常被使用，因为它是开发者指定该二进制构建目标给 NVIDIA CUDA 编译器 *nvcc* 的方式。

> **注意：** 二进制兼容性仅对由 NVIDIA 工具（如 `nvcc`）创建的二进制文件承诺。手动编辑或为 NVIDIA GPUs 生成二进制代码不受支持。如果二进制文件以任何方式被修改，兼容性承诺将无效。

##### 1.3.4.2. PTX 兼容性

GPU 代码可以以二进制或 PTX 形式存储在可执行文件中，如 *Cubins and Fatbins* 所述。当应用程序存储 GPU 代码的 PTX 版本时，该 PTX 可以在运行时**为等于或高于该 PTX 代码计算能力的任何 GPU** 进行 JIT 编译。例如，如果应用程序包含 *compute_80* 的 PTX 代码，则该 PTX 代码可以在运行时 JIT 编译为更高版本的 SM（如 *sm_120*）。这使得应用程序和库能够向前兼容未来的 GPU，无需重新构建。

##### 1.3.4.3. 即时编译

应用程序在运行时加载的 PTX 代码由设备驱动程序编译为二进制代码，这称为即时（JIT）编译。JIT 编译增加了应用程序的加载时间，但允许应用程序受益于随每个新设备驱动程序发布的编译器改进。它还使应用程序能够在编译时尚不存在的设备上运行。

当设备驱动程序为应用程序 JIT 编译 PTX 代码时，它会自动缓存生成的二进制代码的副本，以避免在后续调用应用程序时重复编译。这个缓存——称为计算缓存（compute cache）——在设备驱动程序升级时会自动失效，以便应用程序能够从新的 JIT 编译器的改进中受益。

PTX 在运行时如何以及何时进行 JIT 编译，自 CUDA 的早期版本以来已有放宽，允许在何时以及是否 JIT 编译某些或所有 kernels 上具有更多的灵活性。*Lazy Loading* 部分描述了可用的选项以及如何控制 JIT 行为。还有一些环境变量可以控制 JIT 编译行为，如 *CUDA 环境变量* 中所述。

作为使用 `nvcc` 编译 CUDA C++ 设备代码的替代方案，可以使用 NVRTC 在运行时将 CUDA C++ 设备代码编译为 PTX。NVRTC 是一个用于 CUDA C++ 的运行时编译库；更多信息可以在 NVRTC 用户指南中找到。
