---
layout: post
title: Median Filter over Arbitrary Datatypes
date: 2025-09-08
comments: false
tags: [algorithms, computer-vision, optimization, cpp]
archive: false
---

Median filter is one of most common filters in computer vision applications. Many variants of median filter that can be applied to arbitrary datatypes have been proposed throughout the years and a lot of progress is made to make it more efficient. I find all these research very fascinating and was not able to find a good source that aggregates and explains them all. So here it is now!

The write up first introduces the baseline median filter (V1) then optimizes it to improve the per pixel median computation (V2) achieve 4.2 times speedup, enable multi threading (V3) gives an even more speedup of 16 times, finally an median filter over the ordinal transform of the image (V4) is discussed which gives 420 times speedup. Finally if the restriction of generalized datatypes is removed algorithm which enable an even greater speedup is discussed.

**Table of Contents:**
- [V1: Median Filter (Baseline)](#v1-median-filter-baseline)
- [V2: Linear Time Median Finding](#v2-linear-time-median-finding)
- [V3: Making it Multithreaded](#v3-making-it-multithreaded)
- [V4: Median over Ordinal Transform](#v4-median-over-ordinal-transform)
- [Limitations](#limitations)

## V1: Median Filter (Baseline)

The main idea behind applying a median filter over a 2D image is to replace each pixel with the median of the values in a window around that pixel. The window can be square or circular. I will be focusing on a square window but it can be easily extended to a circular or hexagonal window by adapting how we add and remove pixels inside our window. I also use a 2D black & white image to avoid verbose code but can be easily modified for general multi channel images by applying the filter on each channel separately.

Implementing the simplest possible median filter would be to loop over all pixels of the image and for each pixel we store the values in the surrounding window in a buffer. We then sort the values in the window and take the middle one as the median. If the window size is odd, that's just the single middle element. If it's even, we take the average of the two middle elements, for integer types this becomes the floor of that average because of integer division. This is how implementing it looks like:

```cpp
#include <algorithm>
#include <vector>

template <typename T>
void median_filter_v1(
    const T* input, T* output,
    int ny, int nx, int hy, int hx
) {
    T *pixels = (T *)malloc((2 * hy + 1) * (2 * hx + 1) * sizeof(T));

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {

            int len = 0;
            for (int i = std::max(y - hy, 0); i < std::min(y + hy + 1, ny); i++) {
                for (int j = std::max(x - hx, 0); j < std::min(x + hx + 1, nx); j++) {
                    pixels[len++] = input[nx * i + j];
                }
            }

            const int mid = len / 2;
            std::sort(pixels, pixels + len);
            if (len % 2 == 1) {
                output[nx * y + x] = pixels[mid];
            } else {
                // For integral types median is the
                // floor of the mean of the two middle values
                output[nx * y + x] = (pixels[mid] + pixels[mid - 1]) / T(2);
            }

        }
    }

    free(pixels);
}
```

Let's benchmark this for an images of size 1000 * 1000 and uint8, float and double dataypes with varying kernel sizes and taking an average of 5 runs. I perform these benchmarks on an AMD EPVC 4564P 16 core processor:

```
kernel_size             uint8                float                double
5 * 5               77.10 ± 0.07 ms      95.31 ± 0.17 ms      94.38 ± 0.41 ms
10 * 10             549.81 ± 1.42 ms     693.87 ± 1.03 ms     690.97 ± 0.51 ms
15 * 15            1142.64 ± 2.15 ms    1449.58 ± 1.97 ms    1442.46 ± 0.59 ms
20 * 20            2495.58 ± 4.78 ms    3160.45 ± 4.05 ms    3163.54 ± 0.88 ms
25 * 25            3699.39 ± 3.45 ms    4701.75 ± 2.10 ms    4713.27 ± 1.47 ms
30 * 30            5946.51 ± 7.35 ms    7668.03 ± 5.78 ms    7659.16 ± 10.32 ms
```

Apart from noticing that this is very slow we can also reason that the average runtime scales O(r² + r²log(r²)) per pixel. This is expected because for each pixel we read r² values and sort them which takes r² + r²log(r²) per pixel operations. This also seems to match the trend that runtime grows superlinearly on top of the quadratic increase due to the kernel.

Second thing to notice is the huge gap between uint8 and float/double. The L1 cache on my AMD processor is 512 KiB:

```
L1d cache:                            512 KiB
L1i cache:                            512 KiB
L2 cache:                             16 MiB
L3 cache:                             64 MiB
```

For an window of 30*30 doubles that's 6KiB and our entire buffer can fit in the cache lines. So every time the std::sort loads a value for comparison it does not the pay the full price of loading it from global memory. And since our runtime here is mostly dominated by the comparisons of std::sort comparing uint8 is fast which utilizes `_Z13compare_uint8hh` instructions which has 1 cycle latency compared to comparing float/double using `_Z13compare_floatff` which has 4 cycle latency.

## V2: Linear Time Median Finding

Previously we sorted the entire buffer even though for median computation we only need the middle element(s). We can do better by selecting instead of sorting. The idea is simple: repeatedly partition the buffer around a pivot so that values smaller than the pivot move to the left and values larger move to the right. If the pivot lands in the middle, we are done. If not, we recurse only on the side that contains the middle. Because each partition discards half of the search space in expectation, the selection finishes in linear time on average. This is also called the quick selection algorithm. There is also a deterministic linear time selection using the median of medians scheme, but for our purpose the average case behavior is more than enough for understanding it.

### Proof of Average Linear Time Complexity

To prove the average case linear time complexity. We can model the quick select with a simple recurrence. One partition of the algorithm scans the entire array once so it costs O(n). After partitioning we only scan the side of the array with the kth element. With a random (uniformly choosen) pivot the expected size of that side is n/2. So we have the recurrence T(n) = T(n/2) + O(n). Unrolling this gives us T(n) = cn + cn/2 + cn/4 + … = 2cn = O(n), or could use [master theorem](https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms)) with a=1, b=2 and f(n)=cn giving us the upper bound of O(n). Another way to see it is each level does linear work and on average the subproblem halves so the geometric series converges to linear time. A more careful analysis shows the expected number of comparisons is of the order of 2n with a random pivot. The worst case happens when the pivot is always the smallest or largest element and keep picking an extreme pivot, which gives T(n) = T(n-1) + O(cn), T(n) = O(n²). That case is rare with randomized or median of three pivots, and if we really need to rule it out, using [median-of-medians](https://en.wikipedia.org/wiki/Median_of_medians) as pivot selection guarantees linear time by ensuring the pivot rank always falls in a fixed middle interval like [3n/10, 7n/10]. In short it's guaranteed to have expected linear run time on average, a more strict and formal proof using median of medians can be found in [Time bounds for selection](https://people.csail.mit.edu/rivest/pubs/BFPRT73.pdf).

In C++ this routine is implemented with the `std::nth_element`. This works for the case when the buffer has odd number of elements but for even elements as discussed the function also ensures that all elements on the left of pivot are less than pivot and we can just iterate over the left half to find the next greatest element while still maintaining linear time median finding per pixel:

```cpp
// Move the element in the middle as if the array was sorted
std::nth_element(pixels, pixels + mid, pixels + len);

if (len & 1) {
    output[nx*y + x] = pixels[mid];
} else {
    // even count, find the max in the lower half
    T hi = pixels[mid];
    T lo = pixels[mid - 1];
    for(T *p=pixels; p<pixels + mid - 1; p++) lo = std::max(lo, *p);
    output[nx*y + x] = (lo + hi) / T(2);
}
```

Benchmarking this gives us:

```
kernel_size             uint8                float                double
5 * 5               57.75 ± 0.13 ms      68.46 ± 0.10 ms      68.03 ± 0.07 ms
10 * 10             222.32 ± 0.52 ms     265.32 ± 0.09 ms     266.47 ± 0.26 ms
15 * 15             387.34 ± 0.64 ms     464.77 ± 0.90 ms     466.43 ± 0.23 ms
20 * 20             726.90 ± 0.44 ms     871.55 ± 0.48 ms     881.61 ± 2.87 ms
25 * 25            1009.66 ± 2.23 ms    1208.53 ± 0.93 ms    1222.44 ± 0.31 ms
30 * 30            1518.99 ± 3.47 ms    1824.89 ± 1.43 ms    1834.24 ± 1.41 ms
```

This is a clear win over fully sorting. The cost per pixel is still dominated by reading r² values from memory, but the median selection step is now linear. On large windows this compounds into a noticeable speedup. On my machine this approach is about **4.2× faster** than the baseline for large kernels while producing the exact same results.

## V3: Making it Multithreaded

The next low hanging fruit is make utilize multi threading. So far everything runs on a single core. Median filtering is embarrassingly parallel over output pixels, so we should use all cores. A simple strategy is to split the image into a regular grid of blocks and assign blocks to threads. Each thread processes its own block tile by tile. This avoids synchronization on the hot path and keeps per thread working sets small. We also make a thread operate on one block at a time so it can reuse a local window buffer in the next improvement ;)

```cpp
int Sx = nx / Nx + 1;
int Sy = ny / Ny + 1;

// Split the image into blocks of size Sx x Sy
// and process each block in parallel
// yg -> grid index of the block in y
// xg -> grid index of the block in x
#pragma omp parallel for collapse(2) schedule(dynamic)
for(int yg=0; yg<ny; yg+=Sy) {
    for(int xg=0; xg<nx; xg+=Sx) {

        // median filter over input[yg:yg+Sy, xg:xg+Sx]

    }
}
```

In cpp [OpenMP](https://en.wikipedia.org/wiki/OpenMP) makes this straightforward. We pick a grid size (Nx, Ny), derive the block sizes (Sx, Sy), and parallelize the nested loops with `collapse(2)` which parallelizes the nested for loop iterations across threads. The window buffer is local to the loop iteration (and therefore thread-safe).

On my AMD Processor which has 16 cores with a [hyperthreading](https://en.wikipedia.org/wiki/Hyper-threading) of 2. Hyperthreading is a way in which modern CPU arch virtualize multiple threads on a single core:

```
CPU(s):                               32
On-line CPU(s) list:                  0-31
Thread(s) per core:                   2
Core(s) per socket:                   16
Socket(s):                            1
```

Benchmarking this gives us a **16× speedup**:

```
kernel_size             uint8                float                double
5 * 5               15.68 ± 0.17 ms      18.55 ± 0.87 ms      18.19 ± 0.07 ms
10 * 10              60.67 ± 0.93 ms      72.07 ± 1.36 ms      71.45 ± 0.20 ms
15 * 15             106.52 ± 1.80 ms     128.59 ± 5.41 ms     126.34 ± 1.02 ms
20 * 20             199.56 ± 3.94 ms     234.77 ± 0.44 ms     238.22 ± 3.84 ms
25 * 25             275.47 ± 1.99 ms     330.32 ± 3.13 ms     329.48 ± 1.35 ms
30 * 30             417.48 ± 3.98 ms     500.04 ± 4.30 ms     500.43 ± 6.84 ms
```

The speedup is now dominated by core count and memory bandwidth rather than the selection routine itself. We still read r² values per output pixel, but the total wall-clock time drops substantially because multiple blocks are processed in parallel. This is the point where smarter data reuse and window update strategies start to matter we will get there when we stop treating each pixel independently and restructure the computation. That is the motivation for the next section.

## V4: Median over Ordinal Transform

There is lot of overlap between the pixels in the adjacent windows, so instead of rereading the pixels for each window and sorting them every time, we can make a data structure that allows us to update and delete new pixel values while also able to efficiently search the median in the current window.

But if store ranks(ordinal positions) of the pixels rather than raw values, and represent the current window as a compact bitset over those ranks. Then the median is just "the k-th set bit." This turns median lookup into just fast bit twiddling operation.

### Ordinal Transform

We first take a block around the region we want the filter to be applied. Sort all pixel values in this block once, and assign each pixel a rank from 0 to B-1 where B is the number of the pixels in the block. That mapping is our ordinal transform. The window is now represented in which ranks are present and not which values are present:

```
Raw 5×5 values      Sorted pairs (value, id)      Rank image (same shape)
┌─────────────┐     value,id (by value asc)        id = (row*W + col)
|  8  2  5  5  9 |  ( 2,  1) → rank 0              ranks[id] = rank
|  3  7  4  6  1 |  ( 3,  5) → rank 1              ┌─────────────┐
|  9  0  2  2  8 |  ( 4,  6) → rank 2              | 10  0  4  3 20 |
|  1  5  7  6  3 |  ( 5,  2) → rank 3              |  5 15  7 12  4 |
|  2  8  9  1  0 |  ( 5, 11) → rank 4              | 21 24  9  8 18 |
└─────────────┘     ...                            |  6 11 16 13  2 |
                                                   |  1 17 22 14 23 |
                                                   └─────────────┘
```

When the our sliding window moves, we just toggle the ranks that leave/enter the window. This way we avoid both the cost of rereading the previous pixel values and also the cost of resorting the entire window. The median is just the (count - 1) / 2th bit which is set, and if we have even bit sets it's just the mean of the middle two.

### Data Structure for the Window

We encode window membership as a bitset buff over ranks. Each uint64_t word holds 64 ranks. We also keep a cursor p pointing near the median's word, and two prefix sums psum[0] and psum[1] that count how many set bits lie in words &lt;p and &gt;=p. This lets us jump to the correct word quickly when the target rank changes a little.

```cpp
template <typename T>
struct Block {

    int nx, ny;
    int bx, by;
    int hy, hx;
    int x0b, x1b, y0b, y1b;
    int x0i, y0i, x1i, y1i;
    int x0, y0, x1, y1;
    int words, p;
    int psum[2];
    std::vector<std::pair<T, int>> sorted;
    std::vector<int> ranks;
    std::vector<uint64_t> buff;

    Block(int ny, int nx, int hy, int hx, const T *in, int x0i, int y0i, int x1i, int y1i)
    : nx(nx), ny(ny), hy(hy), hx(hx), x0i(x0i), y0i(y0i), x1i(x1i), y1i(y1i) {

        // The boundaries of the block
        x0b = std::max(x0i - hx, 0);
        y0b = std::max(y0i - hy, 0);
        x1b = std::min(x1i + hx, nx - 1);
        y1b = std::min(y1i + hy, ny - 1);

        x0 = x0i - x0b;
        y0 = y0i - y0b;
        x1 = x1i - x0b;
        y1 = y1i - y0b;

        // The number of pixels in the block along x and y
        bx = (x1b - x0b + 1);
        by = (y1b - y0b + 1);

        sorted.resize(bx * by);
        ranks.resize(bx * by);

        for (int dy = 0; dy < by; dy++) for (int dx = 0; dx < bx; dx++) {
            sorted[dy * bx + dx] = { in[(y0b + dy) * nx + (x0b + dx)], dy * bx + dx };
        }

        std::sort(
            sorted.begin(), sorted.end(),
            [](auto &a, auto &b){
                return a.first < b.first;
            }
        );
     
        for (int i = 0; i < bx * by; i++) {
            ranks[sorted[i].second] = i;
        }

        words = (bx * by + 63) / 64;
        psum[0] = psum[1] = 0;
        p = words / 2;
        buff = std::vector<uint64_t>(words, 0);
    }
```

### Window Update Operation

To add/remove a pixel at any place (ix, jy) we look up its ranks[jy * bx + ix], compute its word and bit, flip the bit, and fix the prefix sums relative to p. Because our scan pattern removes/adds each boundary exactly once using toggle ^= is very cheap operation:

```cpp
    // all indices are local block coordinates
    inline void add_rank(int ix, int jy) {
        if (ix < 0 || ix >= bx || jy < 0 || jy >= by) return;
        int rank = ranks[jy * bx + ix];
        int i = rank >> 6;
        buff[i] ^= (uint64_t(1) << (rank & 63));
        psum[i >= p] += 1;
    }

    inline void remove_rank(int ix, int jy) {
        if (ix < 0 || ix >= bx || jy < 0 || jy >= by) return;
        int rank = ranks[jy * bx + ix];
        int i = rank >> 6;
        buff[i] ^= (uint64_t(1) << (rank & 63));
        psum[i >= p] -= 1;
    }
```

### Median Search

If count = psum[0] + psum[1]. The median index in the ordered multiset is t=(count - 1)/2 (and t = count/2 for the even case). We first make sure p points to the correct 64-bit word that contains the tth set bit. We do that by moving p left/right and updating psum using popcount of the words we cross. Then we locate the exact bit inside that word. For popcount we use the [__builtin_popcountll](https://stackoverflow.com/questions/38113284/whats-the-difference-between-builtin-popcountll-and-mm-popcnt-u64) intrinsic in AMD processor which returns the number of bit sets in the current word:

```cpp
    inline int pop(int idx) const {
        return __builtin_popcountll(buff[idx]);
    }

    inline int search(int target) {

        // localize the target chunk in buffer
        while (psum[0] > target) {
            p--;
            psum[0] -= pop(p);
            psum[1] += pop(p);
        }
        while (psum[0] + pop(p) <= target) {
            psum[0] += pop(p);
            psum[1] -= pop(p);
            p++;
        }
        int n = target - psum[0];

        // courtesy of:
        // https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int
        uint64_t x = _pdep_u64(uint64_t(1) << n, buff[p]);
        int bitpos = __builtin_ctzll(x);
        return (p << 6) | bitpos;
    }
```

Once we are on the right word, we need the n-th set bit inside that 64-bit value, where n = target - W.psum[0]. We use a BMI2 intrinsic [_pdep_u64](https://www.felixcloutier.com/x86/pdep) to scatter a single 1 into the position of the n-th set bit of the mask and then [__builtin_ctzll](https://stackoverflow.com/questions/76705816/how-can-i-use-the-builtin-function-builtin-ctzll-in-a-vs-c-project) intrinsic to get its index.

```cpp
    inline T get_median() {
        int sum = psum[0] + psum[1];
        int i1 = search((sum - 1) / 2);
        if (sum % 2 == 1) {
            return sorted[i1].first;
        } else {
            int i2 = search(sum / 2);
            return (sorted[i1].first + sorted[i2].first) / T(2);
        }
    }
```

### Zig Zag Filter

When we slide left to right across a row, only a new right column enters and a left column leaves. When we go to the next row, instead of resetting the window, we switch direction and scan right to left. The zip zag pattern helps us reuse most of the window and only touch a thin boundary each step:

```cpp
    inline void compute_median(T *out) {

        for (int ix = x0 - hx; ix < x0 + hx; ix++) 
            for (int jy = y0 - hy; jy <= y0 + hy; jy++) 
                add_rank(ix, jy);

        int x = x0 - 1;
        while (x <= x1) {

            // Remove the left vertical boundary of the window and add the right
            for (int jy = y0 - hy; jy <= y0 + hy; jy++) remove_rank(x - hx, jy);
            x++; if (x > x1) break;
            for (int jy = y0 - hy; jy <= y0 + hy; jy++) add_rank(x + hx, jy);

            int y = y0;
            while (y < y1) {

                out[(x + x0b) + nx * (y + y0b)] = get_median();
     
                // remove the upper horizontal boundary and add lower
                if (y - hy >= 0) 
                    for (int ix = -hx; ix <= hx; ix++) remove_rank(x + ix, y - hy);
                y++;
                if (y + hy < ny) 
                    for (int ix = -hx; ix <= hx; ix++) add_rank(x + ix, y + hy);
            }
     
            out[(x + x0b) + nx * (y + y0b)] = get_median();
     
            // Remove the left vertical boundary of the window
            for (int jy = y - hy; jy <= y + hy; jy++) remove_rank(x - hx, jy);
            x++; if (x > x1) break;
            // Add right vertical boundary of the window
            for (int jy = y - hy; jy <= y + hy; jy++) add_rank(x + hx, jy);
     
            y = y1;
            while (y > y0) {
     
                out[(x + x0b) + nx * (y + y0b)] = get_median();
     
                // remove the lower horizontal boundary
                if (y + hy < ny) 
                    for (int ix = -hx; ix <= hx; ix++) remove_rank(x + ix, y + hy);
                y--;
                // add the upper horizontal boundary
                if (y - hy >= 0) 
                    for (int ix = -hx; ix <= hx; ix++) add_rank(x + ix, y - hy);
            }
     
            out[(x + x0b) + nx * (y + y0b)] = get_median();
        }
    }
};
```

Benchmarking this gives us a **420 times speed up**:

```
kernel_size             uint8                float                double
5 * 5                6.92 ± 0.28 ms       8.51 ± 0.08 ms       8.69 ± 0.16 ms
10 * 10               7.72 ± 0.07 ms       9.66 ± 0.05 ms       9.62 ± 0.12 ms
15 * 15               8.77 ± 0.11 ms      10.73 ± 0.15 ms      10.78 ± 0.21 ms
20 * 20              10.52 ± 0.12 ms      12.57 ± 0.06 ms      12.56 ± 0.26 ms
25 * 25              11.83 ± 0.17 ms      12.70 ± 1.19 ms      13.89 ± 0.24 ms
30 * 30              13.61 ± 0.09 ms      15.86 ± 0.31 ms      15.66 ± 0.17 ms
```

Every step only touch O(hx + hy) pixels and not the entire window. We now have only single block sort O(BlogB), when a window moves by one pixel toggle at most O(r) pixels and also have p tracking the median of each word we move p by at most 0-1 words. This is also the point where we see the gap in performance difference between datatypes to be the least, as the comparison operations gets pushed to one block sort at the initial.

We can also parallelize this over multiple threads now and make use the parallelization in V3. Here is the complete implementation with parallelization:

```cpp
#include <vector>
#include <cstdint>
#include <iostream>
#include <x86intrin.h>
#include <algorithm>
#include <omp.h>
#include <cmath>

template <typename T>
void median_filterv4(
    const T *input, T *output,
    int ny, int nx, int hy, int hx
) {

    int num_threads = omp_get_max_threads();
    int target_blocks = std::max(num_threads * 3, 4);
    int blocks_per_dim = std::max(1, (int)std::sqrt((double)target_blocks));
     
    int Bx = std::max(32, (nx + blocks_per_dim - 1) / blocks_per_dim);
    int By = std::max(32, (ny + blocks_per_dim - 1) / blocks_per_dim);
     
    if (nx <= 64 && ny <= 64) {
        Bx = nx;
        By = ny;
    }
     
    Bx = std::min(Bx, std::max(nx / 2, 64));
    By = std::min(By, std::max(ny / 2, 64));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y0 = 0; y0 < ny; y0 += By) {
        for (int x0 = 0; x0 < nx; x0 += Bx) {

            int x1 = std::min(x0 + Bx - 1, nx - 1);
            int y1 = std::min(y0 + By - 1, ny - 1);

            Block<T> block(
                ny, nx, hy, hx, input,
                x0, y0, x1, y1
            );
         
            block.compute_median(output);
        }
    }

}
```

## Limitations

This writeup only discusses fast median filter algorithms for generalized datatypes. However if the restrictions of general datatypes is removed and only considered uint8 datatype which most real world images are stored in even more speedup can be gained. For example [Huang's Algorithm](https://ieeexplore.ieee.org/document/1163188) uses a 256 bit histogram for storing the median buffer of V4 which can be very easily optimized to have constant time median search. [Median filtering by localization of median value](https://www.ripublication.com/ijaer18/ijaerv13n12_107.pdf) takes this a step further and also introduces a column histogram while [Perreault and Hebert](http://mesh.brown.edu/engn1610/refs/PerreaultHebert-tip2007.pdf) also uses a similar variant of column histogram to result in a constant time median filtering algorithm but only applicable to 256 bit images due to the nature of lookup based median computation. Finally [Moruto](https://cgenglab.github.io/en/publication/sigga22_wmatrix_median/) introduced the wavelet based constant time median filtering algorithm recently.

