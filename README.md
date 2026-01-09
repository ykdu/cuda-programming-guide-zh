# CUDA Programming Guide 13.1 中文翻译（非官方）

本仓库是 NVIDIA 官方文档 **CUDA C++ Programming Guide** 的非官方中文翻译与整理工程，目标是提供一个**结构化、可维护、可引用**的中文版本（以 GitHub 形式长期迭代）。

- 官方原文（Table of Contents 及正文）：  
  https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- 本项目定位：学习与技术交流（非 NVIDIA 官方发布）

---

## 快速开始

- 📚 **总目录（推荐从这里读）**：[`BOOK.md`](BOOK.md)
- 🧭 每章目录在对应章节文件夹的 `README.md`（由脚本自动生成）
- 🧾 术语对照表：[`TERMINOLOGY.md`](TERMINOLOGY.md)
- ⚠️ 免责声明：[`DISCLAIMER.md`](DISCLAIMER.md)
- 🏷️ 对齐的官方版本说明：[`VERSION.md`](VERSION.md)

---

## 翻译进度

> 说明：  
> - ✅ 已完成：已完成初版翻译，可阅读  
> - 🚧 进行中：正在翻译或校对  
> - ⏳ 计划中：仅有结构，占位尚未翻译

### 第 1 章 Introduction to CUDA
- ✅ 1.1 Introduction
- ✅ 1.2 Programming Model
- ✅ 1.3 The CUDA Platform

### 第 2 章 Programming GPUs in CUDA
- ✅ 2.1 Intro to CUDA C++
- ✅ 2.2 Writing CUDA SIMT Kernels
- ✅ 2.3 Asynchronous Execution
- ✅ 2.4 Unified and System Memory
- ✅ 2.5 NVCC: The NVIDIA CUDA Compiler

### 第 3 章 Advanced CUDA
- ✅ 3.1 Advanced CUDA APIs and Features
- ✅ 3.2 Advanced Kernel Programming
- ⏳ 3.3 The CUDA Driver API
- ⏳ 3.4 Programming Systems with Multiple GPUs
- ⏳ 3.5 A Tour of CUDA Features

### 第 4 章 CUDA Features
- ⏳ 4.1 Unified Memory
- ✅ 4.2 CUDA Graphs
- ⏳ 4.3 Stream-Ordered Memory Allocator
- ⏳ 4.4 Cooperative Groups
- ⏳ 4.5 Programmatic Dependent Launch and Synchronization
- ✅ 4.6 Green Contexts
- ⏳ 4.7 Lazy Loading
- ⏳ 4.8 Error Log Management
- ⏳ 4.9 Asynchronous Barriers
- ⏳ 4.10 Pipelines
- ⏳ 4.11 Asynchronous Data Copies
- ⏳ 4.12 Work Stealing with Cluster Launch Control
- ⏳ 4.13 L2 Cache Control
- ⏳ 4.14 Memory Synchronization Domains
- ⏳ 4.15 Interprocess Communication
- ✅ 4.16 Virtual Memory Management
- ⏳ 4.17 Extended GPU Memory
- ⏳ 4.18 CUDA Dynamic Parallelism
- ⏳ 4.19 CUDA Interoperability with APIs
- ⏳ 4.20 Driver Entry Point Access

### 第 5 章 Technical Appendices
- ⏳ 5.1 Compute Capabilities
- ⏳ 5.2 CUDA Environment Variables
- ⏳ 5.3 C++ Language Support
- ⏳ 5.4 C/C++ Language Extensions
- ⏳ 5.5 Floating-Point Computation
- ⏳ 5.6 Device-Callable APIs and Intrinsics

### 第 6 章 Notices
- ⏳ Notices

---

## 仓库结构说明

- 每个一级章节一个目录（如 `04-cuda-features/`）
- 每个官方小节对应一个 Markdown 文件（如 `04-06-green-contexts.md`）
- 每章目录下的 `README.md` 是章节入口与目录页（脚本生成）
- 根目录 `BOOK.md` 是整本文档的导航索引（脚本生成）

---

## 如何维护目录与进度

当你：
- 新增章节文件
- 完成某一节翻译
- 调整文件标题

只需要：

```bash
./gen_toc_readmes.sh

