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

#### 4.1.1.1. Unified Memory：深入示例

具备完整 CUDA unified memory 支持的系统（参见 *Overview of Unified Memory Paradigms* 表）允许 device 访问 host process 所拥有的任意 memory，只要该 process 正在与 device 进行交互。

本节将展示一些较为高级的使用场景，示例中使用的 kernel 只是将输入的字符数组中的前 8 个字符打印到标准输出流中：

```c
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

以下标签页展示了使用 system 分配的 memory 调用该 kernel 的多种方式：

Malloc

```c
void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with %s", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}
```

Managed

```c
void test_managed() {
  const char test_string[] = "Hello World";
  char* data;
  cudaMallocManaged(&data, sizeof(test_string));
  strncpy(data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("managed", data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  cudaFree(data);
}
```

Stack variable

```c
void test_stack() {
  const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("stack", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

File-scope static variable

```c
void test_static() {
  static const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("static", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

Global-scope variable

```c
const char global_string[] = "Hello World";

void test_global() {
  kernel<<<1, 1>>>("global", global_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

Global-scope extern variable

```c
// declared in separate file, see below
extern char* ext_data;

void test_extern() {
  kernel<<<1, 1>>>("extern", ext_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

```c
/** This may be a non-CUDA file */
char* ext_data;
static const char global_string[] = "Hello World";

void __attribute__ ((constructor)) setup(void) {
  ext_data = (char*)malloc(sizeof(global_string));
  strncpy(ext_data, global_string, sizeof(global_string));
}

void __attribute__ ((destructor)) tear_down(void) {
  free(ext_data);
}
```

请注意，对于 extern 变量，它可以由第三方库声明，其 memory 由该库拥有并管理，而该库本身可能完全不与 CUDA 交互。

还要注意，stack 变量以及 file-scope 和 global-scope 变量只能通过 pointer 被 GPU 访问。在这个具体示例中，这一点比较方便，因为字符数组已经被声明为 pointer：`const char*`。不过，请看下面这个使用 global-scope 整数变量的示例：

```c
// 该变量声明在 global scope
int global_variable;

__global__ void kernel_uncompilable() {
  // 这会导致编译错误：global (__host__) 变量
  // 不允许在 __device__ / __global__ 代码中访问
  printf("%d\n", global_variable);
}

// 在 pageableMemoryAccess 被设置为 1 的系统上，我们可以访问
// global 变量的地址。下面的 kernel 将该地址作为参数传入
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}

int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}
```

在上面的示例中，我们必须确保向 kernel 传递 global 变量的 pointer，而不是在 kernel 中直接访问该 global 变量。这是因为没有使用 `__managed__` 修饰符的 global 变量默认被声明为仅 `__host__` 可见，因此目前大多数编译器不允许在 device 代码中直接使用这些变量。

##### 4.1.1.1.1. File-backed Unified Memory

由于具备完整 CUDA unified memory 支持的系统允许 device 访问 host 进程拥有的任何 memory，因此它们可以直接访问 file-backed memory。

这里我们展示了对上一节中初始示例的一个修改版本，使用 file-backed memory，从输入文件中直接读取数据，并在 GPU 上打印字符串。在下面的示例中，memory 由一个物理文件支持，但该示例同样适用于 memory-backed 文件。

```c
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - 前 %d 个字符: ", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("\n");
}
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "无效的文件句柄");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "无效的文件状态信息");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "无法将文件映射到 memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
         "CUDA failed with %s", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "无法解除文件映射");
  ASSERT(close(fd) == 0, "无法关闭文件");
}
```

请注意，在不具备 `hostNativeAtomicSupported` 属性（参见 Host Native Atomics）的系统上，包括启用了 Linux HMM 的系统，对 file-backed memory 的 atomic 访问不受支持。

##### 4.1.1.1.2. 使用 Unified Memory 的 Inter-Process Communication (IPC)

> **Note**
>  目前，使用 unified memory 进行 IPC 可能会带来显著的性能影响。

许多应用更倾向于每个进程管理一个 GPU，但仍然需要使用 unified memory，例如用于 over-subscription，并且需要从多个 GPU 访问该 memory。

CUDA IPC（参见 Interprocess Communication）不支持 managed memory：这种类型 memory 的 handle 不能通过本节讨论的任何机制进行共享。在具备完整 CUDA unified memory 支持的系统上，system 分配的 memory 是支持 IPC 的。一旦 system 分配的 memory 访问权限被共享给其他进程，就可以使用相同的编程模型，类似于 File-backed Unified Memory。

关于在 Linux 下创建支持 IPC 的 system 分配 memory 的各种方式，请参考以下资料：

- 使用 MAP_SHARED 的 mmap
- POSIX IPC APIs
- Linux memfd_create

请注意，无法通过这种技术在不同 host 及其 device 之间共享 memory。

// 如果你想跨进程共享给 GPU 用，别用 cudaMallocManaged，用操作系统分配的内存。

#### 4.1.1.2. 性能调优

为了在 unified memory 上获得良好的性能，重要的是：

- 理解系统上的 paging 工作机制，以及如何避免不必要的 page faults
- 理解各种机制，使数据保持在访问它的 processor 本地
- 根据系统的 memory 传输粒度对应用进行调优

一般建议，性能提示（参见 Performance Hints）可能会提升性能，但如果使用不当，可能比默认行为表现更差。还要注意，任何提示在 host 上都会带来性能成本，因此有用的提示至少必须带来足够的性能提升来抵消这一成本。

##### 4.1.1.2.1. Memory Paging and Page Sizes

为了更好地理解 unified memory 的性能影响，有必要理解 virtual addressing、memory pages 以及 page sizes。本小节将定义所有必要术语，并解释为什么 paging 对性能至关重要。

目前所有支持 unified memory 的系统都使用 virtual address space：这意味着应用使用的 memory address 表示的是一个 virtual 位置，该位置可能被映射到 memory 实际所在的 physical 位置。

目前所有支持的 processor，包括 CPU 和 GPU，都使用 memory paging。由于所有系统都使用 virtual address space，因此存在两种 memory page：

- Virtual pages：表示每个进程的一段固定大小、连续的 virtual memory，由操作系统跟踪，并可以映射到 physical memory。请注意，virtual page 与映射相关联：例如，一个 virtual address 可能通过不同的 page size 被映射到 physical memory。
- Physical pages：表示 processor 的主要 Memory Management Unit (MMU) 支持的一段固定大小、连续的 memory，virtual page 可以映射到其中。

目前，所有 x86_64 CPU 默认使用 4KiB 的 physical page size。Arm CPU 根据具体型号支持 4KiB、16KiB、32KiB 和 64KiB 等多种 physical page size。NVIDIA GPU 也支持多种 physical page size，但更倾向于使用 2MiB 或更大的 physical page。请注意，这些大小可能会随未来硬件变化。

virtual page 的默认 page size 通常对应于 physical page size，但只要操作系统和硬件支持，应用可以使用不同的 page size。通常，支持的 virtual page size 必须是 2 的幂，并且是 physical page size 的整数倍。

用于跟踪 virtual page 到 physical page 映射关系的逻辑实体称为 page table，而给定 virtual page（及其 virtual size）到 physical page 的每个映射称为 Page Table Entry (PTE)。所有支持的 processor 都为 page table 提供专门的缓存，以加速 virtual address 到 physical address 的转换。这些缓存称为 Translation Lookaside Buffers (TLBs)。

应用进行性能调优时有两个重要方面：

- virtual page size 的选择
- 系统是否提供 CPU 和 GPU 共享的 combined page table，还是每个 CPU 和 GPU 分别使用独立的 page table

###### 4.1.1.2.1.1. 选择合适的 Page Size

一般而言，小 page size 会减少（virtual）memory 碎片，但会导致更多 TLB misses；较大的 page size 会增加 memory 碎片，但会减少 TLB misses。此外，与小 page size 相比，较大的 page size 会使 memory migration 更昂贵，因为通常会迁移整个 memory page。这可能会在使用大 page size 的应用中引发更大的 latency 峰值。有关 page fault 的更多细节，请参见下一节。

性能调优中的一个重要方面是，TLB misses 在 GPU 上通常比在 CPU 上代价更高。这意味着，如果 GPU thread 频繁访问使用较小 page size 映射的 unified memory 中的随机位置，那么相比使用较大 page size 映射的 unified memory，这种访问可能会显著更慢。虽然 CPU thread 在使用小 page size 映射的大块 memory 中随机访问时也可能出现类似现象，但性能下降通常不那么明显，因此应用可能需要在减少 memory 碎片与性能之间进行权衡。

请注意，通常应用不应根据某个 processor 的 physical page size 来进行性能调优，因为 physical page size 可能会随硬件变化。上述建议仅适用于 virtual page size。

###### 4.1.1.2.1.2. CPU 和 GPU Page Tables：Hardware Coherency vs. Software Coherency

硬件一致（hardware-coherent）的系统，例如 NVIDIA Grace Hopper，为 CPU 和 GPU 提供逻辑上合并的 page table。这一点非常重要，因为当 GPU 访问 system 分配的 memory 时，它会使用 CPU 为该 memory 创建的 page table entry。如果该 entry 使用默认的 CPU page size（4KiB 或 64KiB），那么对大块 virtual memory 区域的访问将导致大量 TLB misses，从而带来显著的性能下降。

另一方面，在软件一致（software-coherent）的系统上，CPU 和 GPU 各自拥有独立的逻辑 page table，此时需要考虑不同的性能调优因素：为了保证一致性，当某个 processor 访问映射到另一 processor 的 physical memory 时，系统通常会使用 page faults。此类 page fault 意味着：

- 必须确保当前拥有该 physical page 的 processor 不再访问该 page，可以通过删除或更新其 page table entry 实现。
- 必须确保请求访问的 processor 能够访问该 page，可以通过创建新的 page table entry 或更新现有 entry，使其变为有效/激活状态。
- 支撑该 virtual page 的 physical page 必须迁移到请求访问的 processor 上：这是一个代价较高的操作，其工作量与 page size 成正比。

总体而言，在 CPU 和 GPU threads 频繁并发访问同一 memory page 的场景下，hardware-coherent 系统相比 software-coherent 系统具有显著性能优势：

- 更少的 page-fault：无需通过 page-fault 来模拟一致性或迁移 memory
- 更少的 contention：系统在 cache-line 粒度上保持一致，而非 page-size 粒度。也就是说，当多个 processor 在同一 cache line 内产生竞争时，仅交换该 cache line，其大小远小于最小 page size；而当不同 processor 访问同一 page 内不同 cache line 时，则不会产生竞争。

这会影响以下场景的性能：

- CPU 和 GPU 同时对同一地址进行 atomic 更新
- 从 CPU thread 向 GPU thread 发信号，或反之。

##### 4.1.1.2.2. host 直接访问 Unified Memory

一些 device 在硬件上支持 host 对 GPU-resident unified memory 进行一致性的 reads、stores 以及 atomic 访问。这些 device 的属性 `cudaDevAttrDirectManagedMemAccessFromHost` 被设置为 1。请注意，所有 hardware-coherent 系统在其 NVLink 连接的 device 上都会设置该属性。在这些系统上，host 可以在不发生 page faults 和 data migration 的情况下，直接访问 GPU-resident memory。请注意，对于 CUDA managed memory，需要使用带有 location type `cudaMemLocationTypeHost` 的 `cudaMemAdviseSetAccessedBy` 提示，才能启用这种不经过 page faults 的直接访问，见下例。

System Allocator：

```c
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // 对于 shared page table 系统，下面这个 hint 不是必须的
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages 在 GPU memory 中被填充
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU 直接访问 GPU memory，不发生 migrations
                                                  // directManagedMemAccessFromHost=0: CPU 触发 fault 并触发 device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU 访问 GPU memory，不发生 migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU 触发 fault 并触发 host-to-device migrations
  free(ret);
}
```

Managed：

```c
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // 设置 direct access hint

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages 在 GPU memory 中被填充
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU 直接访问 GPU memory，不发生 migrations
                                                  // directManagedMemAccessFromHost=0: CPU 触发 fault 并触发 device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU 访问 GPU memory，不发生 migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU 触发 fault 并触发 host-to-device migrations
  cudaFree(ret);
}
```

在 `write` kernel 完成之后，`ret` 将在 GPU memory 中被创建并初始化。接下来，CPU 会访问 `ret`，随后 `append` kernel 会再次使用同一块 `ret` memory。根据系统架构以及对 hardware coherency 的支持情况，这段代码会表现出不同的行为：

- 在 `directManagedMemAccessFromHost=1` 的系统上：CPU 对 managed buffer 的访问不会触发任何 migrations；数据将保持 resident 在 GPU memory 中，之后的 GPU kernels 也可以继续直接访问它，而不会引发 faults 或 migrations
- 在 `directManagedMemAccessFromHost=0` 的系统上：CPU 对 managed buffer 的访问会触发 page fault 并启动 data migration；任何 GPU kernel 在首次尝试访问同一数据时都会触发 page fault，并将 pages 迁移回 GPU memory。

// ![image-20260204213701041](/Users/duyunkai/Library/Application Support/typora-user-images/image-20260204213701041.png)

##### 4.1.1.2.3. Host Native Atomics

一些 device（包括 hardware-coherent 系统中通过 NVLink 连接的 device）支持对 CPU-resident memory 的硬件加速 atomic 访问。这意味着对 host memory 的 atomic 访问不必通过 page fault 来进行模拟。对于这些 device，属性 `cudaDevAttrHostNativeAtomicsSupported` 被设置为 1。

##### 4.1.1.2.4. Atomic 访问与同步原语

CUDA unified memory 支持 host 与 device threads 可用的所有 atomic 操作，使所有 threads 可以通过并发访问同一个共享内存位置来协作。libc++ 库提供了许多面向 host 与 device threads 并发使用而优化的异构同步原语，包括 `cuda::atomic`、`cuda::atomic_ref`、`cuda::barrier`、`cuda::semaphore` 等。

在 software-coherent 系统上，device 对 **file-backed host memory** 的 atomic 访问是不支持的。下面的示例代码在 hardware-coherent 系统上是合法的，但在其他系统上会产生未定义行为：

```c
#include <cuda/atomic>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>

#define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);
}

int main() {
  // 默认在退出时关闭/删除
  FILE* tmp_file = tmpfile();
  // 需要为文件分配空间，这里使用 posix_fallocate
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  if (status != 0) ERR("Failed to allocate space in temp file\n");
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

  // 初始化 file-backed memory 中的值
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);

  // device 与 host thread 并发访问 ptr，使用 cuda::atomic_ref
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  // 这里将始终为 2
  printf("Atom value: %d\n", *ptr);

  return EXIT_SUCCESS;
}
```

在 software-coherent 系统上，对 unified memory 的 atomic 访问可能会触发 page fault，从而带来显著的延迟。需要注意的是，这并不适用于这些系统上所有 GPU 对 CPU memory 的 atomic 操作；通过 `nvidia-smi -q | grep "Atomic Caps Outbound"` 列出的操作可能可以避免 page fault。

在 hardware-coherent 系统上，host 与 device 之间的 atomic 操作不需要 page fault，但仍然可能因其他原因触发 fault，就像任何内存访问都可能发生 fault 一样。

##### 4.1.1.2.5. Unified Memory 下的 Memcpy()/Memset() 行为

`cudaMemcpy*()` 和 `cudaMemset*()` 接受任何 unified memory 指针作为参数。

对于 `cudaMemcpy()`，通过 `cudaMemcpyKind` 指定的方向只是一个性能提示。如果任一参数是 unified memory 指针，这个提示可能会对性能产生较大影响。

因此，建议遵循以下性能建议：

- 当 unified memory 的物理位置已知时，使用准确的 `cudaMemcpyKind` 提示。
- 相比错误的 `cudaMemcpyKind` 提示，优先使用 `cudaMemcpyDefault`。
- 始终使用已填充（已初始化）的 buffer：避免使用这些 API 来初始化内存。
- 如果两个指针都指向 system-allocated memory，避免使用 `cudaMemcpy*()`；应启动一个 kernel 或使用 CPU 内存拷贝算法（例如 `std::memcpy`）。

##### 4.1.1.2.6. Unified Memory 的内存分配器概览

在支持完整 CUDA unified memory 的系统上，可以使用多种不同的分配器来分配 unified memory。下表展示了一些分配器及其各自特性的概览。请注意，本节中的所有信息在未来的 CUDA 版本中可能会发生变化。

**表 7：不同分配器对 unified memory 支持情况概览**

| API                                                          | 放置策略            | 可访问方 | 是否基于访问迁移 [2] | 页大小 [4][5]                              |
| ------------------------------------------------------------ | ------------------- | -------- | -------------------- | ------------------------------------------ |
| malloc, new, mmap                                            | 首次触达 / hint [1] | CPU, GPU | 是 [3]               | 系统页大小或 huge page [6]                 |
| cudaMallocManaged                                            | 首次触达 / hint     | CPU, GPU | 是                   | CPU resident：系统页大小 GPU resident：2MB |
| cudaMalloc                                                   | GPU                 | GPU      | 否                   | GPU 页大小：2MB                            |
| cudaMallocHost, cudaHostAlloc, cudaHostRegister              | CPU                 | CPU, GPU | 否                   | CPU 映射：系统页大小 GPU 映射：2MB         |
| Memory pool（location type host）：cuMemCreate, cudaMemPoolCreate | CPU                 | CPU, GPU | 否                   | CPU 映射：系统页大小 GPU 映射：2MB         |
| Memory pool（location type device）：cuMemCreate, cudaMemPoolCreate, cudaMallocAsync | GPU                 | GPU      | 否                   | 2MB                                        |

[1] 对于 mmap，file-backed memory 默认放置在 CPU 上，除非通过 cudaMemAdviseSetPreferredLocation（或 mbind，见下文条目）进行指定。

[2] 此特性可以通过 cudaMemAdvise 覆盖。即使禁用了基于访问的迁移，如果底层 backing memory 空间已满，内存仍可能发生迁移。

[3] file-backed memory 不会基于访问进行迁移。

[4] 在大多数系统上，默认系统页大小为 4KiB 或 64KiB，除非显式指定 huge page（例如使用 mmap MAP_HUGETLB / MAP_HUGE_SHIFT）。在这种情况下，系统配置的任意 huge page 大小都受支持。

[5] GPU resident memory 的页大小在未来 CUDA 版本中可能发生变化。

[6] 当前在将内存迁移到 GPU 或在 GPU 上通过首次触达放置时，huge page 大小可能不会被保留。

// ![image-20260205173927390](/Users/duyunkai/Library/Application Support/typora-user-images/image-20260205173927390.png)

该表展示了多种分配器在 unified memory 语义上的差异，这些分配器可用于分配可同时被多个处理器访问的数据，包括 host 和 device。关于 cudaMemPoolCreate 的更多细节，请参阅 Memory Pools 章节；关于 cuMemCreate 的更多细节，请参阅 Virtual Memory Management 章节。

在硬件一致（hardware-coherent）系统上，当 device memory 作为一个 NUMA domain 暴露给系统时，可以使用诸如 numa_alloc_on_node 等特殊分配器将内存固定到指定的 NUMA node（host 或 device）。这类内存可以同时被 host 和 device 访问，并且不会发生迁移。类似地，可以使用 mbind 将内存固定到指定的 NUMA node(s)，并且可以在首次访问之前，将 file-backed memory 放置到指定的 NUMA node(s)。

// 在 hardware-coherent 系统上，unified memory 退化成 NUMA 问题，你不再考虑迁移。只考虑：本地访问or远端访问。

以下内容适用于共享内存的分配器：

- 系统分配器（如 mmap）允许使用 MAP_SHARED 标志在进程之间共享内存。该特性在 CUDA 中受支持，可用于在同一 host 上连接的不同 device 之间共享内存。然而，目前尚不支持在多个 host 以及多个 device 之间共享内存。详见 Inter-Process Communication (IPC) with Unified Memory。
- 若需通过网络在多个 host 上访问 unified memory 或其他 CUDA memory，请查阅所使用通信库的文档，例如 NCCL、NVSHMEM、OpenMPI、UCX 等。

##### 4.1.1.2.7. Access Counter Migration

在 hardware-coherent 系统上，access counters 功能会跟踪 GPU 对位于其它 processor 上的 memory 的访问频率。这是为了确保 memory pages 被移动到最频繁访问该页面的 processor 的物理 memory 中。它可以引导 CPU 与 GPU 之间、以及 peer GPUs 之间的迁移，这种过程称为 access counter migration。

从 CUDA 12.4 开始，access counters 支持 system-allocated memory。需要注意的是，file-backed memory 不会基于访问行为进行迁移。对于 system-allocated memory，可以通过对相应 device 使用 `cudaMemAdviseSetAccessedBy` 提示来开启 access counters migration。如果启用了 access counters，可以通过将 `cudaMemAdviseSetPreferredLocation` 设置为 host 来阻止迁移。默认情况下，`cudaMallocManaged` 采用 fault-and-migrate 机制进行迁移。[7]

driver 还可以利用 access counters 在 thrashing 缓解或 memory oversubscription 场景下进行更高效的迁移决策。

[7] 当前系统在设置了 accessed-by device 提示时，允许 managed memory 使用 access-counter migration。这属于实现细节，不应依赖其作为未来兼容性的保证。

##### 4.1.1.2.8. 避免 CPU 频繁写入 GPU-Resident Memory

如果 host 访问 unified memory，cache miss 可能会在 host 与 device 之间引入比预期更多的流量。许多 CPU 架构要求所有 memory 操作（包括写操作）都必须经过 cache 层级结构。如果 system memory 当前驻留在 GPU 上，那么 CPU 对这块 memory 的频繁写入可能会导致 cache miss，从而在真正写入目标地址之前，先将数据从 GPU 传输回 CPU。在 software-coherent 系统上，这可能会引入额外的 page fault；而在 hardware-coherent 系统上，则可能增加 CPU 操作之间的延迟。因此，为了让 device 读取 host 产生的数据，应考虑将数据写入 CPU-resident memory，并让 device 直接读取这些值。下面的代码展示了如何在 unified memory 中实现这一点。

System Allocator：

```c
size_t data_size = sizeof(int);
int* data = (int*)malloc(data_size);
// 确保 data 保持在 host 本地，避免发生 fault
cudaMemLocation location = {.type = cudaMemLocationTypeHost};
cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

// 频繁交换小数据：如果 CPU 写入 CPU-resident memory，
// 并且 GPU 直接访问该数据，则可以避免在两次写入之间若数据被逐出时
// CPU cache 重新加载数据
for (int i = 0; i < 10; ++i) {
  *data = 42 + i;
  kernel<<<1, 1>>>(data);
  cudaDeviceSynchronize();
  // 此处 CPU cache 可能已经逐出该数据
}
free(data);
```

Managed：

```c
int* data;
size_t data_size = sizeof(int);
cudaMallocManaged(&data, data_size);
// 确保 data 保持在 host 本地，避免发生 fault
cudaMemLocation location = {.type = cudaMemLocationTypeHost};
cudaMemAdvise(data, data_size, cudaMemAdviseSetPreferredLocation, location);
cudaMemAdvise(data, data_size, cudaMemAdviseSetAccessedBy, location);

// 频繁交换小数据：如果 CPU 写入 CPU-resident memory，
// 并且 GPU 直接访问该数据，则可以避免在两次写入之间若数据被逐出时
// CPU cache 重新加载数据
for (int i = 0; i < 10; ++i) {
  *data = 42 + i;
  kernel<<<1, 1>>>(data);
  cudaDeviceSynchronize();
  // 此处 CPU cache 可能已经逐出该数据
}
cudaFree(data);
```

##### 4.1.1.2.9. 利用对 System Memory 的异步访问

如果应用程序需要将 device 上的计算结果与 host 共享，有以下几种可选方式：

1. device 将结果写入 GPU-resident memory，然后使用 `cudaMemcpy*` 进行传输，host 读取传输后的数据。
2. device 直接将结果写入 CPU-resident memory，host 读取该数据。
3. device 将结果写入 GPU-resident memory，host 直接访问该数据。

如果在 host 传输或访问结果的同时，device 上可以调度独立工作，则优先选择方案 1 或 3。如果 device 必须等待 host 访问完结果才能继续工作，则方案 2 可能更合适。这是因为 device 通常具有比 host 更高的写入带宽，除非使用多个 host threads 来读取数据。

1. 显式拷贝（Explicit Copy）

```c
void exchange_explicit_copy(cudaStream_t stream) {
  int* data, *host_data;
  size_t n_bytes = sizeof(int) * 16;
  // 分配接收缓冲区
  host_data = (int*)malloc(n_bytes);
  // 分配内存，由于首先在 device 上访问，因此将驻留在 GPU
  cudaMallocManaged(&data, n_bytes);
  kernel<<<1, 16, 0, stream>>>(data);
  // 在 device 上启动独立工作
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // 传输到 host
  cudaMemcpyAsync(host_data, data, n_bytes, cudaMemcpyDeviceToHost, stream);
  // 同步 stream，确保数据已传输完成
  cudaStreamSynchronize(stream);
  // 读取传输后的数据
  printf("Got values %d - %d from GPU\n", host_data[0], host_data[15]);
  cudaFree(data);
  free(host_data);
}
```

2. Device 直接写入（Device Direct Write）

```c
void exchange_device_direct_write(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // 分配接收缓冲区
  cudaMallocManaged(&data, n_bytes);
  // 确保 data 映射并驻留在 host 上
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, location);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, location);
  kernel<<<1, 16, 0, stream>>>(data);
  // 同步 stream，确保数据已写入完成
  cudaStreamSynchronize(stream);
  // 读取数据
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}
```

3. Host 直接读取（Host Direct Read）

```c
void exchange_host_direct_read(cudaStream_t stream) {
  int* data;
  size_t n_bytes = sizeof(int) * 16;
  // 分配接收缓冲区
  cudaMallocManaged(&data, n_bytes);
  // 确保 data 映射并驻留在 device 上
  cudaMemLocation device_loc = {};
  cudaGetDevice(&device_loc.id);
  device_loc.type = cudaMemLocationTypeDevice;
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetPreferredLocation, device_loc);
  cudaMemAdvise(data, n_bytes, cudaMemAdviseSetAccessedBy, device_loc);
  kernel<<<1, 16, 0, stream>>>(data);
  // 在 GPU 上启动独立工作
  // other_kernel<<<1024, 256, 0, stream>>>(other_data, ...);
  // 同步 stream，确保数据已经由 device 写入
  cudaStreamSynchronize(stream);
  // host 直接读取数据
  printf("Got values %d - %d from GPU\n", data[0], data[15]);
  cudaFree(data);
}
```

最后，在上述 Explicit Copy 示例中，除了使用 `cudaMemcpy*` 进行传输外，也可以通过 host 或 device kernel 显式完成数据传输。对于连续数据，优先使用 CUDA copy-engines，因为 **copy-engines 执行的操作可以与 host 和 device 的工作重叠**。`cudaMemcpy*` 和 `cudaMemPrefetchAsync` 可能使用 copy-engines，但并不保证一定使用。基于同样的原因，对于足够大的数据，显式拷贝优于直接 host 读取：如果 host 和 device 的工作都未达到各自 memory 系统的饱和状态，copy-engines 可以在 host 和 device 执行工作时并行完成传输。

copy-engines 通常用于 host 与 device 之间以及 NVLink 互联的 peer devices 之间的传输。由于 copy-engines 数量有限，某些系统上 `cudaMemcpy*` 的带宽可能低于使用 device 显式执行传输的方式。如果数据传输位于应用程序的关键路径上，可能更适合使用显式的 device 传输。

### 4.1.2. 仅支持 CUDA Managed Memory 的设备上的 Unified Memory

对于 compute capability 6.x 或更高、但不支持 pageable memory access 的设备（参见 *Overview of Unified Memory Paradigms* 表格），CUDA managed memory 是完全支持并保持一致性的，但 GPU 无法访问 system-allocated memory。

Unified memory 的编程模型和性能调优方式，与前文 *Unified Memory on Devices with Full CUDA Unified Memory Support* 一节中描述的模型基本相同，但有一个重要区别：不能使用 system allocator 来分配 memory。因此，下面列出的子章节内容在此类设备上不适用：

- Unified Memory: In-Depth Examples
- CPU and GPU Page Tables: Hardware Coherency vs. Software Coherency
- Atomic Accesses and Synchronization Primitives
- Access Counter Migration
- Avoid Frequent Writes to GPU-Resident Memory from the CPU
- Exploiting Asynchronous Access to System Memory

### 4.1.3. Unified Memory on Windows, WSL, and Tegra

略









### **4.1.4 性能提示**

性能提示允许程序员为 CUDA 提供更多有关统一内存使用的信息。CUDA 使用性能提示更有效地管理内存，并提高应用程序的性能。性能提示从不影响应用程序的正确性。性能提示仅影响性能。

> 注意
>
> 如果性能提示不能提高性能，则应用程序不应使用统一内存性能提示。

性能提示可以用于任何统一内存分配，包括 CUDA 管理的内存。在支持完全 CUDA 统一内存的系统上，性能提示可以应用于所有系统分配的内存。

#### **4.1.4.1 数据预取**

 `cudaMemPrefetchAsync` API 是一个异步流排序 API，可能将数据迁移到更靠近指定处理器的位置。在预取过程中，可以访问这些数据。迁移直到所有流中的先前操作完成后才会开始，并在流中任何后续操作之前完成。

```c
cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream=0);
```

包含 [devPtr, devPtr + count] 的内存区域可能会迁移到目标设备的位置，如果 `location.type` 是 `cudaMemLocationTypeDevice`，或者如果 `location.type` 是 `cudaMemLocationTypeHost`，当预取任务在给定的 `stream` 中执行时，它将迁移到 CPU。有关标志的详细信息，请参阅当前的 CUDA Runtime API 文档。

以下是简单的代码示例：

System Allocator：

```c
void test_prefetch_sam(const cudaStream_t& s) {
    // 在 CPU 上初始化数据
    char *data = (char*)malloc(dataSizeBytes);
    init_data(data, dataSizeBytes);
    cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

    // 鼓励数据在使用前迁移到 GPU
    const unsigned int flags = 0;
    cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

    // 在 GPU 上使用数据
    const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
    mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

    // 鼓励数据迁移回 CPU
    location = {.type = cudaMemLocationTypeHost};
    cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

    cudaStreamSynchronize(s);

    // 在 CPU 上使用数据
    use_data(data, dataSizeBytes);
    free(data);
}
```

Managed：

```c
void test_prefetch_managed(const cudaStream_t& s) {
    // 在 CPU 上初始化数据
    char *data;
    cudaMallocManaged(&data, dataSizeBytes);
    init_data(data, dataSizeBytes);
    cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

    // 鼓励数据在使用前迁移到 GPU
    const unsigned int flags = 0;
    cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

    // 在 GPU 上使用数据
    const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
    mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

    // 鼓励数据迁移回 CPU
    location = {.type = cudaMemLocationTypeHost};
    cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);
    
    cudaStreamSynchronize(s);

    // 在 CPU 上使用数据
    use_data(data, dataSizeBytes);
    cudaFree(data);
}
```

#### **4.1.4.2 Data Usage Hints**

 当多个处理器同时访问相同的数据时，可以使用 `cudaMemAdvise` 来提示如何访问位于 [devPtr, devPtr + count] 的数据：

```c
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

以下示例展示了如何使用 `cudaMemAdvise`：

```c
    init_data(data, dataSizeBytes);
    cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};

    // 鼓励数据在使用前迁移到 GPU
    const unsigned int flags = 0;
    cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

    // 在 GPU 上使用数据
    const unsigned num_blocks = (dataSizeBytes + threadsPerBlock - 1) / threadsPerBlock;
    mykernel<<<num_blocks, threadsPerBlock, 0, s>>>(data, dataSizeBytes);

    // 鼓励数据迁移回 CPU
    location = {.type = cudaMemLocationTypeHost};
    cudaMemPrefetchAsync(data, dataSizeBytes, location, flags, s);

    cudaStreamSynchronize(s);

    // 在 CPU 上使用数据
    use_data(data, dataSizeBytes);
    cudaFree(data);
}
// test-prefetch-managed-end

static const int maxDevices = 1;
static const int maxOuterLoopIter = 3;
static const int maxInnerLoopIter = 4;

// test-advise-managed-begin
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * threadsPerBlock;  // 16 KiB
```

`advice` 可以采取以下值：

- `cudaMemAdviseSetReadMostly`：
   这意味着数据主要是要被读取，而仅偶尔被写入。通常，这可以用来在这个区域上通过牺牲读取带宽来换取写入带宽。
- `cudaMemAdviseSetPreferredLocation`：
   此提示设置数据的首选位置为指定设备的物理内存。此提示鼓励系统保持数据在首选位置，但不会保证它。传入 `cudaMemLocationTypeHost` 的值为 `location.type` 将数据放置在 CPU 内存中。其他提示如 `cudaMemPrefetchAsync` 可能会覆盖此提示，并允许内存迁移到首选位置以外的位置。
- `cudaMemAdviseSetAccessedBy`：
   在某些系统上，在访问数据之前建立内存映射可能对性能有利。此提示告诉系统数据将由指定处理器频繁访问。它启用系统假设创建这些映射会带来回报。此提示并不意味着数据一定要存储在这些映射中，但可以与 `cudaMemAdviseSetPreferredLocation` 一起使用来指定位置。在硬件一致系统中，此提示与访问计数器迁移开关一起使用，具体信息见访问计数器迁移。

每个 `advice` 也可以通过使用以下函数之一来取消设置：`cudaMemAdviseUnsetReadMostly`、`cudaMemAdviseUnsetPreferredLocation` 和 `cudaMemAdviseUnsetAccessedBy`。

System Allocator 标签页：

```c
void test_advise_managed(cudaStream_t stream) {
    char *dataPtr;
    size_t dataSize = 64 * threadsPerBlock; // 16 KiB

    // 使用 malloc 分配内存
    dataPtr = (char*)malloc(dataSize);

    // 为内存区域设置提示
    cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
    cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

    int outerLoopIter = 0;
    while (outerLoopIter < maxOuterLoopIter) {
        // 数据由 CPU 每次外部循环迭代时写入
        init_data(dataPtr, dataSize);

        // 通过预取使数据对所有 GPU 可用
        // 预取操作导致数据的读取复制
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        for (int device = 0; device < maxDevices; device++) {
            location.id = device;
            const unsigned int flags = 0;
            cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
        }

        // 内循环中仅读取此数据
        int innerLoopIter = 0;
        while (innerLoopIter < maxInnerLoopIter) {
            mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
            innerLoopIter++;
        }
        outerLoopIter++;
    }

    free(dataPtr);
}
```

Managed 标签页：

```c
void test_advise_managed(cudaStream_t stream) {
    char *dataPtr;
    size_t dataSize = 64 * threadsPerBlock; // 16 KiB

    // 使用 cudaMallocManaged 分配内存（仅在支持完整 CUDA 统一内存的系统上）
    cudaMallocManaged(&dataPtr, dataSize);

    // 设置内存区域的提示
    cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
    cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);

    int outerLoopIter = 0;
    while (outerLoopIter < maxOuterLoopIter) {
        // 数据由 CPU 每次外部循环迭代时写入
        init_data(dataPtr, dataSize);

        // 通过预取使数据对所有 GPU 可用
        // 预取操作导致数据的读取复制
        cudaMemLocation location;
        location.type = cudaMemLocationTypeDevice;
        for (int device = 0; device < maxDevices; device++) {
            location.id = device;
            const unsigned int flags = 0;
            cudaMemPrefetchAsync(dataPtr, dataSize, location, flags, stream);
        }

        // 内循环中仅读取此数据
        int innerLoopIter = 0;
        while (innerLoopIter < maxInnerLoopIter) {
            mykernel<<<32, threadsPerBlock, 0, stream>>>((const char *)dataPtr, dataSize);
            innerLoopIter++;
        }
        outerLoopIter++;
    }

    cudaFree(dataPtr);
}
```

#### 4.1.4.3 Querying Data Usage Attributes on Managed Memory

 程序可以通过以下 API 查询通过 `cudaMemAdvise` 或 `cudaMemPrefetchAsync` 在 CUDA 管理内存上分配的内存范围属性：

```
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

此函数查询从 `devPtr` 开始的、大小为 `count` 字节的内存范围的属性。内存范围必须是通过 `cudaMallocManaged` 分配或通过 `__managed__` 变量声明的 CUDA 管理内存。可以查询以下属性：

- `cudaMemRangeAttributeReadMostly`：如果整个内存范围都设置了 `cudaMemAdviseSetReadMostly` 属性，则返回 1，否则返回 0。
- `cudaMemRangeAttributePreferredLocation`：返回值将是一个 GPU 设备 ID，如果整个内存范围的内存位置与对应的处理器的位置匹配；否则返回 `cudaInvalidDeviceId`。应用程序可以使用此查询 API 来决定是否通过 CPU 或 GPU 进行数据迁移，具体取决于管理指针的首选位置属性。请注意，在查询时，内存范围的实际位置可能与首选位置不同。
- `cudaMemRangeAttributeAccessedBy`：将返回具有该提示的设备列表。
- `cudaMemRangeAttributeLastPrefetchLocation`：返回最后一次通过 `cudaMemPrefetchAsync` 明确预取的内存范围位置。请注意，这仅返回应用程序请求将内存范围预取到的最后位置，它并不表示预取操作是否已经完成或开始。
- `cudaMemRangeAttributePreferredLocationType`：返回首选位置类型，其值如下：
  - `cudaMemLocationTypeDevice`：如果内存范围中的所有页面都将相同的 GPU 作为其首选位置。
  - `cudaMemLocationTypeHost`：如果内存范围中的所有页面都将 CPU 作为其首选位置。
  - `cudaMemLocationTypeHostNuma`：如果内存范围中的所有页面都将相同的主机 NUMA 节点 ID 作为其首选位置。
  - `cudaMemLocationTypeInvalid`：如果内存范围中的任何页面没有相同的首选位置，或者某些页面根本没有首选位置。
- `cudaMemRangeAttributePreferredLocationId`：如果查询的相同地址范围返回 `cudaMemLocationTypeDevice`，则返回设备的设备序号。如果首选位置类型是主机 NUMA 节点，则返回主机 NUMA 节点 ID，否则忽略该 ID。
- `cudaMemRangeAttributeLastPrefetchLocationType`：返回所有内存范围页面最后一次明确通过 `cudaMemPrefetchAsync` 预取的位置类型，返回以下值：
  - `cudaMemLocationTypeDevice`：如果内存范围中的所有页面都预取到同一 GPU。
  - `cudaMemLocationTypeHost`：如果内存范围中的所有页面都预取到 CPU。
  - `cudaMemLocationTypeHostNuma`：如果内存范围中的所有页面都预取到相同的主机 NUMA 节点 ID。
  - `cudaMemLocationTypeInvalid`：如果内存范围中的所有页面都没有预取到相同位置，或者某些页面从未预取过。
- `cudaMemRangeAttributeLastPrefetchLocationId`：如果 `cudaMemRangeAttributeLastPrefetchLocationType` 查询返回 `cudaMemLocationTypeDevice`，则返回有效的设备设备序号，或者如果返回 `cudaMemLocationTypeHost` 或 `cudaMemLocationTypeHostNuma`，则返回有效的主机 NUMA 节点 ID，否则忽略该 ID。

此外，可以使用相应的 `cudaMemRangeGetAttributes` 函数查询多个属性。

#### 4.1.4.4 GPU Memory Oversubscription

统一内存使得应用程序可以超分配任何个别处理器的内存：换句话说，程序可以分配并共享比任何个别处理器内存容量更大的数组，这使得系统能够处理那些无法完全适应单一 GPU 的数据集，而无需为编程模型增加额外的复杂性。

此外，可以使用相应的 `cudaMemRangeGetAttributes` 函数查询多个属性。
