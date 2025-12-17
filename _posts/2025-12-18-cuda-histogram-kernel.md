---
layout: post
title: CUDA Histogram Kernel
date: 2025-12-17
comments: false
tags: [cuda, optimization]
archive: false
---

I recently revisited CUDA and optimized a histogram kernel that I wrote last year. This document catalogs the incremental improvements that led to an efficient histogram computation for 8‑bit integers

### Version 1: Basic Kernel

A very simple kernel to compute the histogram of 8‑bit integers in CUDA looks like this:

```cpp
__global__ void histogramKernelv1(
    const uint8_t* __restrict__ a,
    int* __restrict__ bins,
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&bins[a[i]], 1);
    }
}
```

In this kernel, each element of the array `a` is processed by a separate thread. Each thread reads a value and updates the corresponding bin in the global memory histogram using CUDA’s `atomicAdd` API to prevent race conditions. The `__restrict__` keyword informs the compiler that these pointers are the only references to the data, which can help with optimization

Profiling this kernel on an array of size $2^{20}$ yields an execution time of approximately **230 µs**:

```bash
>>> nvcc -o main -arch=sm_86 histogram.cu && nsys nvprof ./main
...
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)
 --------  ---------------  ---------  --------  --------  --------  --------
    100.0       230914,129          1   4,129.0   4,129.0     4,129     4,129 
...
```

### Version 2: Using Shared Memory

The primary bottleneck in Version 1 is the frequent global memory writes, which leads to write conflicts. To alleviate this, we can create a sub‑histogram for each thread block by using shared memory. Shared memory (allocated with the `__shared__` keyword) is fast and accessible to all threads in a block, though its size is limited (48 KB on my RTX 4040). Fortunately, our kernel only requires about 1 KB (256 bins × 4 bytes per bin)

```cpp
__global__ void histogramKernelv2(
    const uint8_t* __restrict__ a,
    int* __restrict__ bins,
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_bins[256];
    // Initialize the shared histogram to 0
    if (threadIdx.x < 256) {
        s_bins[threadIdx.x] = 0;
    }
    __syncthreads();

    if (i < N) {
        atomicAdd(&s_bins[a[i]], 1);
    }
    __syncthreads();
    // Merge the shared histogram into global memory
    if (threadIdx.x < 256) {
        atomicAdd(&bins[threadIdx.x], s_bins[threadIdx.x]);
    }
}
```

Here, `__syncthreads()` is used for block-level synchronization. The first barrier ensures that all shared bins are initialized to zero before computation begins, and the second barrier ensures that the histogram computation is complete before merging the sub‑histograms back into global memory. This optimization reduces the execution time to approximately **38.304 µs**.

### Version 3: Loading Multiple Elements per Thread

To further improve performance, we can have each thread load multiple elements instead of one. This approach reduces synchronization overhead. In the following kernel, each thread processes a fixed number of elements (denoted by `stride`):

```cpp
__global__ void histogramKernelv3(
    const uint8_t* __restrict__ a,
    int* __restrict__ bins,
    const int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_bins[256];
    if (threadIdx.x < 256) {
        s_bins[threadIdx.x] = 0;
    }
    __syncthreads();

    constexpr size_t stride = 16;
    if (i * stride + stride - 1 < N) [[likely]] {
        #pragma unroll
        for (int j = 0; j < stride; j++) {
            atomicAdd(&s_bins[a[i * stride + j]], 1);
        }
    } else {
        for (int j = i * stride; j < N; j++) {
            atomicAdd(&s_bins[a[j]], 1);
        }
    }
    
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&bins[threadIdx.x], s_bins[threadIdx.x]);
    }
}
```

The `#pragma unroll` directive instructs the compiler to unroll the loop that loads consecutive elements. The `[[likely]]` attribute hints that this branch of the conditional is expected to be executed most of the time, allowing the compiler to optimize it further. If the array length is not a multiple of `stride`, the remaining elements are processed one by one. This version achieves an execution time of approximately **11.136 µs**.

### Version 4: Using 128-Bit Read/Write Operations

For the final optimization, we leverage 128‑bit read/write operations to load consecutive elements of the array at once. To do this, we first cast the array pointer to an `int4*` to load 128 bits in one go. We then cast the loaded section to a `uint8_t*` to process individual bytes.

```cpp
__global__ void histogramKernelv4(
    const uint8_t* __restrict__ a,
    int* __restrict__ bins,
    const size_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int s_bins[256];
    if (threadIdx.x < 256) {
        s_bins[threadIdx.x] = 0;
    }
    __syncthreads();

    constexpr size_t stride = 16;
    if (i * stride + stride - 1 < N) [[likely]] {
        int4 ain = ((int4*)a)[i];
        uint8_t* ax = (uint8_t*)&ain;
        #pragma unroll
        for (int j = 0; j < stride; j++) {
            atomicAdd_block(&s_bins[ax[j]], 1);
        }
    } else {
        for (int j = i * stride; j < N; j++) {
            atomicAdd_block(&s_bins[a[j]], 1);
        }
    }

    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&bins[threadIdx.x], s_bins[threadIdx.x]);
    }
}
```

By using 128‑bit loads, this version further reduces memory access overhead and achieves an execution time of approximately **4.160 µs**