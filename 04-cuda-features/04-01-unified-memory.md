# 4.1 Unified Memory

> 原文：CUDA Programming Guide v13.1  
> https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

TODO: 翻译内容



本节将详细说明 unified memory 各种可用范式的具体行为和使用方式。前文关于 unified memory 的章节已经介绍了如何判断适用哪一种 unified memory 范式，并对每一种范式做了简要说明。

如前所述，unified memory 编程共有四种范式：

- 对显式 managed memory allocation 的完全支持
- 对所有 allocation 的完全支持，并采用 software coherence
- 对所有 allocation 的完全支持，并采用 hardware coherence
- 受限的 unified memory 支持

前三种涉及完全 unified memory 支持的范式在行为和编程模型上非常相似，将统一在 4.1.1 中进行介绍，并在其中指出它们之间的差异。

最后一种 unified memory 支持受限的范式，将在 4.1.2 中进行详细讨论。

### 4.1.1. 在具备完整 CUDA Unified Memory 支持的设备上使用 Unified Memory

这类系统包括 hardware-coherent memory system，例如 NVIDIA Grace Hopper，以及启用了 Heterogeneous Memory Management（HMM）的现代 Linux 系统。HMM 是一种基于 software 的 memory management system，它提供了与 hardware-coherent memory system 相同的编程模型。

Linux HMM 需要 Linux kernel 版本 6.1.24+、6.2.11+ 或 6.3+，需要 compute capability 为 7.5 或更高的 device，并且要求安装带有 Open Kernel Modules 的 CUDA driver 535+。

> **Note**
>  我们将 CPU 和 GPU 共享同一套 page table 的系统称为 hardware-coherent system。
>  CPU 和 GPU 各自使用独立 page table 的系统被称为 software-coherent system。

像 NVIDIA Grace Hopper 这样的 hardware-coherent system 为 CPU 和 GPU 提供了一套逻辑上统一的 page table，参见 *4.1.1.2.1.2. CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency*。下面的小节仅适用于 hardware-coherent system：

> * Access Counter Migration



------

#### 4.1.1.1. Unified Memory：深入示例

具备完整 CUDA unified memory 支持的系统（参见 *Overview of Unified Memory Paradigms* 表）允许 device 访问 host process 所拥有的任意 memory，只要该 process 正在与 device 进行交互。

本节将展示一些较为高级的使用场景，示例中使用的 kernel 只是将输入的字符数组中的前 8 个字符打印到标准输出流中：

```
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

下面的各个选项卡展示了在使用 system-allocated memory 的情况下，调用该 kernel 的多种方式：









































#### 4.1.1.2.1.2. CPU 和 GPU Page Tables：Hardware Coherency 与 Software Coherency 的本质区别

在支持 hardware coherency 的系统中（例如 NVIDIA Grace Hopper），CPU 与 GPU 在逻辑上共享同一套 page table。也就是说，无论是 CPU 还是 GPU，在访问同一段 virtual address 时，看到的都是同一套地址映射关系。GPU 在访问 system-allocated memory 时，直接使用由 CPU 建立的 page table entry，而不需要额外的地址转换或 ownership 切换。这种设计的一个重要影响在于 page size 的选择。如果 CPU 为该 memory 使用的是默认的 page size（例如 4KiB 或 64KiB），那么当 GPU 访问大规模、连续的 virtual memory 区域时，会因为 page 粒度过小而频繁触发 TLB miss，从而造成明显的性能下降。但即便如此，CPU 与 GPU 之间并不存在 page ownership 的切换，访问始终是共享且一致的。

与之相对，在 software-coherent 的系统中，CPU 和 GPU 各自维护独立的 page table。为了在这种架构下保证 memory coherency，当 CPU 和 GPU 访问同一段 virtual memory 时，系统并不是通过共享 cache-line 来维持一致性，而是通过 page 粒度的 ownership 转移 来实现。

在这种模型下，当某个 processor（CPU 或 GPU）访问的 virtual address 实际上映射到另一 processor 当前“拥有”的 physical memory 时，系统会触发一次 page fault。这种 page fault 并不只是简单的“缺页”，而是意味着系统需要完成一次**整页 memory 所有权的转移**。具体来说，这个过程包括三个紧密相关的步骤。

首先，系统必须撤销当前拥有该 page 的 processor 对这块 memory 的访问权限。也就是说，原 processor 对应的 page table entry 会被删除或标记为无效，从而确保它之后无法再访问这块 page。
 随后，系统需要为发起访问请求的 processor 建立对该 page 的访问权限，通过创建新的 page table entry，或更新已有 entry，使该 page 在该 processor 的 page table 中变为有效。
 最后，支撑该 virtual page 的 physical page 会被实际迁移到发起访问请求的 processor 所在的位置。由于迁移的基本单位是 page，这一步的成本与 page size 成正比，page 越大，迁移所需的数据移动和同步开销也越高。

正因为 software-coherent 系统是以 **page 为最小一致性单位**，在 CPU 和 GPU 频繁并发访问同一段 memory 的场景下，会出现大量 page fault、page ownership 来回切换以及随之而来的 migration 开销。这在性能上是非常不利的。

相比之下，hardware-coherent 系统在 CPU 与 GPU 之间以 **cache-line 粒度** 实现一致性。当 CPU 和 GPU 同时访问同一 page 中的不同 cache-line 时，不会产生任何冲突；即便多个 processor 竞争同一个 cache-line，也只需要在 processor 之间交换该 cache-line，而不是迁移整个 page。这使得 hardware-coherent 系统在并发访问场景下显著减少了 page fault、降低了 contention，并避免了大规模 memory migration。

这种差异在以下典型场景中尤为明显：

* 当 CPU 与 GPU 同时对同一 address 执行 atomic update 时，software-coherent 系统往往需要反复迁移 page，而 hardware-coherent 系统只需在 cache-line 层面协调；
* 当 CPU thread 与 GPU thread 之间通过共享 memory 进行 signaling 时，software-coherent 系统同样会因 page ownership 切换而引入额外延迟，而 hardware-coherent 系统则可以在 cache-line 粒度下高效完成同步。
