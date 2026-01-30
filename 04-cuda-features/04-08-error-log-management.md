# 4.8 Error Log Management

Error Log Management 机制允许将 CUDA API 错误以易懂的英文格式报告给开发者，描述问题的原因。

### 4.8.1. 背景

传统上，唯一表明 CUDA API 调用失败的方式是返回非零代码。截至 CUDA Toolkit 12.9，CUDA Runtime 为错误情况定义了 100 多种不同返回代码，但其中许多都是通用的，在调试原因时并不能为开发者提供帮助。

### 4.8.2. 激活

设置环境变量 CUDA_LOG_FILE。可接受的值是 stdout、stderr 或系统上的有效路径以写入文件。即使在程序执行之前没有设置 CUDA_LOG_FILE，也可以通过 API 将日志缓冲区转储出来。注意：无错误执行可能不会打印任何日志。

### 4.8.3. 输出

日志按以下格式输出：

```
[Time][TID][Source][Severity][API Entry Point] Message
```

下面这一行是实际的错误消息，当开发者尝试将 Error Log Management 日志转储到未分配的缓冲区时生成：

```
[22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL
```

在此之前，开发者最多只能从返回代码中得到 CUDA_ERROR_INVALID_VALUE，并且在调用 cuGetErrorString 时可能看到 “invalid argument”。

### 4.8.4. API 描述

CUDA Driver 提供了两类用于与 Error Log Management 功能交互的 API。

此功能允许开发者注册回调函数，当生成错误日志时调用，回调函数签名如下：

```
void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)
```

使用以下 API 注册回调：

```
CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)
```

其中 userData 会原样传递给回调函数。callback_out 应由调用者保存，以便后续用于 cuLogsUnregisterCallback。

```
CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)
```

另一组 API 函数用于管理日志输出。一个重要的概念是日志迭代器，它指向缓冲区的当前末尾：

```
CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)
```

当不希望转储整个日志缓冲区时，调用方的软件可以保存迭代器位置。目前 flags 参数必须为 0，保留其他选项以供未来 CUDA 发布使用。

在任何时候，都可以使用这些函数将错误日志缓冲区转储到文件或内存：

```
CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)
```

如果 iterator 是 NULL，则将转储整个缓冲区，最多 100 条日志条目。如果 iterator 不为 NULL，则从该条目开始转储日志，并将 iterator 的值更新为当前日志末尾，就好像调用了 cuLogsCurrent。一旦缓冲区中有超过 100 条日志条目，在转储开始处会添加一行注明这一情况。

flags 参数必须为 0，其他选项为未来 CUDA 发布保留。

cuLogsDumpToMemory 函数有额外注意事项：

1. 缓冲区本身将以 null 结尾，但每个单独的日志条目仅由换行符 (n) 分隔。
2. 缓冲区的最大大小为 25600 字节。
3. 如果提供的 size 值不足以存储所有所需日志，则会在首条记录中添加一个说明，并且不适合的最旧条目将不会被转储。
4. 返回后，size 将包含写入所提供缓冲区的实际字节数。

### 4.8.5. 限制和已知问题

1. 日志缓冲区限制为 100 条目。达到该限制后，最旧的条目将被替换，日志转储将包含一行注明滚动情况。
2. 并非所有 CUDA API 都已涵盖。这是一个持续进行的项目，旨在为所有 API 提供更好的使用错误报告。
3. Error Log Management 日志位置（如果指定）在生成日志之前/除非生成日志，否则不会测试其有效性。
4. Error Log Management API 目前仅通过 CUDA Driver 可用。等效 API 将在未来的发布中添加到 CUDA Runtime。
5. 日志消息未本地化为任何语言，所有提供的日志均为美国英语。
