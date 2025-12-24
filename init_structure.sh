#!/usr/bin/env bash
set -euo pipefail

# Create root meta files if missing
touch README.md DISCLAIMER.md TERMINOLOGY.md VERSION.md

mkfile () {
  local path="$1"
  local title="$2"
  if [[ ! -f "$path" ]]; then
    mkdir -p "$(dirname "$path")"
    cat > "$path" <<EOF
# $title

> 原文：CUDA Programming Guide v13.1  
> https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

TODO: 翻译内容
EOF
  fi
}

# Chapter 1
mkdir -p 01-introduction-to-cuda
mkfile 01-introduction-to-cuda/README.md "第 1 章 Introduction to CUDA"
mkfile 01-introduction-to-cuda/01-01-introduction.md "1.1 Introduction"
mkfile 01-introduction-to-cuda/01-02-programming-model.md "1.2 Programming Model"
mkfile 01-introduction-to-cuda/01-03-the-cuda-platform.md "1.3 The CUDA platform"

# Chapter 2
mkdir -p 02-programming-gpus-in-cuda
mkfile 02-programming-gpus-in-cuda/README.md "第 2 章 Programming GPUs in CUDA"
mkfile 02-programming-gpus-in-cuda/02-01-intro-to-cuda-cpp.md "2.1 Intro to CUDA C++"
mkfile 02-programming-gpus-in-cuda/02-02-writing-cuda-simt-kernels.md "2.2 Writing CUDA SIMT Kernels"
mkfile 02-programming-gpus-in-cuda/02-03-asynchronous-execution.md "2.3 Asynchronous Execution"
mkfile 02-programming-gpus-in-cuda/02-04-unified-and-system-memory.md "2.4 Unified and System Memory"
mkfile 02-programming-gpus-in-cuda/02-05-nvcc-the-nvidia-cuda-compiler.md "2.5 NVCC: The NVIDIA CUDA Compiler"

# Chapter 3
mkdir -p 03-advanced-cuda
mkfile 03-advanced-cuda/README.md "第 3 章 Advanced CUDA"
mkfile 03-advanced-cuda/03-01-advanced-cuda-apis-and-features.md "3.1 Advanced CUDA APIs and Features"
mkfile 03-advanced-cuda/03-02-advanced-kernel-programming.md "3.2 Advanced Kernel Programming"
mkfile 03-advanced-cuda/03-03-the-cuda-driver-api.md "3.3 The CUDA Driver API"
mkfile 03-advanced-cuda/03-04-programming-systems-with-multiple-gpus.md "3.4 Programming Systems with Multiple GPUs"
mkfile 03-advanced-cuda/03-05-a-tour-of-cuda-features.md "3.5 A Tour of CUDA Features"

# Chapter 4
mkdir -p 04-cuda-features
mkfile 04-cuda-features/README.md "第 4 章 CUDA Features"
mkfile 04-cuda-features/04-01-unified-memory.md "4.1 Unified Memory"
mkfile 04-cuda-features/04-02-cuda-graphs.md "4.2 CUDA Graphs"
mkfile 04-cuda-features/04-03-stream-ordered-memory-allocator.md "4.3 Stream-Ordered Memory Allocator"
mkfile 04-cuda-features/04-04-cooperative-groups.md "4.4 Cooperative Groups"
mkfile 04-cuda-features/04-05-programmatic-dependent-launch-and-synchronization.md "4.5 Programmatic Dependent Launch and Synchronization"
mkfile 04-cuda-features/04-06-green-contexts.md "4.6 Green Contexts"
mkfile 04-cuda-features/04-07-lazy-loading.md "4.7 Lazy Loading"
mkfile 04-cuda-features/04-08-error-log-management.md "4.8 Error Log Management"
mkfile 04-cuda-features/04-09-asynchronous-barriers.md "4.9 Asynchronous Barriers"
mkfile 04-cuda-features/04-10-pipelines.md "4.10 Pipelines"
mkfile 04-cuda-features/04-11-asynchronous-data-copies.md "4.11 Asynchronous Data Copies"
mkfile 04-cuda-features/04-12-work-stealing-with-cluster-launch-control.md "4.12 Work Stealing with Cluster Launch Control"
mkfile 04-cuda-features/04-13-l2-cache-control.md "4.13 L2 Cache Control"
mkfile 04-cuda-features/04-14-memory-synchronization-domains.md "4.14 Memory Synchronization Domains"
mkfile 04-cuda-features/04-15-interprocess-communication.md "4.15 Interprocess Communication"
mkfile 04-cuda-features/04-16-virtual-memory-management.md "4.16 Virtual Memory Management"
mkfile 04-cuda-features/04-17-extended-gpu-memory.md "4.17 Extended GPU Memory"
mkfile 04-cuda-features/04-18-cuda-dynamic-parallelism.md "4.18 CUDA Dynamic Parallelism"
mkfile 04-cuda-features/04-19-cuda-interoperability-with-apis.md "4.19 CUDA Interoperability with APIs"
mkfile 04-cuda-features/04-20-driver-entry-point-access.md "4.20 Driver Entry Point Access"

# Chapter 5
mkdir -p 05-technical-appendices
mkfile 05-technical-appendices/README.md "第 5 章 Technical Appendices"
mkfile 05-technical-appendices/05-01-compute-capabilities.md "5.1 Compute Capabilities"
mkfile 05-technical-appendices/05-02-cuda-environment-variables.md "5.2 CUDA Environment Variables"
mkfile 05-technical-appendices/05-03-cpp-language-support.md "5.3 C++ Language Support"
mkfile 05-technical-appendices/05-04-ccpp-language-extensions.md "5.4 C/C++ Language Extensions"
mkfile 05-technical-appendices/05-05-floating-point-computation.md "5.5 Floating-Point Computation"
mkfile 05-technical-appendices/05-06-device-callable-apis-and-intrinsics.md "5.6 Device-Callable APIs and Intrinsics"

# Chapter 6
mkdir -p 06-notices
mkfile 06-notices/README.md "第 6 章 Notices"

echo "Done. Structure created."

