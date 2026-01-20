---
layout: post
title: Strassen's matmul with AVX 512 kernel
date: 2026-01-18
comments: false
tags: [optimization, cpp, simd, avx512, algorithms]
archive: false
---

In the previous [matmul with avx512 and loop tiling](/2026/01/17/avx512-matrix-multiplication.html) note we managed to build a cpu kernel that achieves 92% of peak FLOPS. I wanted to check if we can do better by using Strassen's algorithm. [Strassen's algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) from 1969 showed that matrix multiplication can be done in $O(n^{2.807})$ by trading multiplications for additions. The main idea of this post is to use strassens for high level recursions and fallback to our highly optimized kernel for lower levels

**Table of Contents:**
- [The Standard Algorithm](#the-standard-algorithm)
- [Strassen's Algorithm](#strassens-algorithm)
- [Implementation](#implementation)
- [Choosing the Recursion Depth](#choosing-the-recursion-depth)
- [Benchmarks](#benchmarks)

## The Standard Algorithm

Standard matrix multiplication for $C = A \times B$ where all matrices are $n \times n$ computes:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

This requires $n^3$ multiplications and $n^3$ additions for a total of $2n^3$ floating point operations. For decades this was assumed optimal until Strassen showed otherwise

## Strassen's Algorithm

Strassen observed that multiplying two $2 \times 2$ matrices:

$$
\begin{bmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{bmatrix} = 
\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
\begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}
$$

normally requires 8 multiplications (each $C_{ij}$ needs 2 products). But with clever grouping we can do it with only 7 multiplications at the cost of more additions

The key insight is that additions are cheap compared to multiplications, especially when the "elements" are themselves large submatrices. If we partition $n \times n$ matrices into four $n/2 \times n/2$ blocks, we can recursively apply this trick. At each level we replace 8 recursive calls with 7, giving the recurrence:

$$
T(n) = 7 T(n/2) + O(n^2)
$$

The $O(n^2)$ term comes from the matrix additions. Solving this recurrence gives $T(n) = O(n^{\log_2 7}) = O(n^{2.807})$

Strassen's algorithm computes seven intermediate products M1 through M7:

$$
\begin{aligned}
M_1 &= (A_{11} + A_{22})(B_{11} + B_{22}) \\
M_2 &= (A_{21} + A_{22}) B_{11} \\
M_3 &= A_{11} (B_{12} - B_{22}) \\
M_4 &= A_{22} (B_{21} - B_{11}) \\
M_5 &= (A_{11} + A_{12}) B_{22} \\
M_6 &= (A_{21} - A_{11})(B_{11} + B_{12}) \\
M_7 &= (A_{12} - A_{22})(B_{21} + B_{22})
\end{aligned}
$$

Then the output blocks are:

$$
\begin{aligned}
C_{11} &= M_1 + M_4 - M_5 + M_7 \\
C_{12} &= M_3 + M_5 \\
C_{21} &= M_2 + M_4 \\
C_{22} &= M_1 - M_2 + M_3 + M_6
\end{aligned}
$$

Each $M_i$ requires one matrix multiplication and 0-2 matrix additions/subtractions. The final assembly requires 8 additions. Total: 7 multiplications and 18 additions instead of 8 multiplications and 4 additions

## Implementation

We need helper functions for matrix addition, subtraction, and copying. These are straightforward but need to handle different strides since submatrices have the parent's stride:

```cpp
static inline
void addMat(
    float *C, int C_stride,
    const float *A, int A_stride,
    const float *B, int B_stride,
    int n
) {
    #pragma omp parallel for collapse(2)
    for(int j=0; j<n; j++) {
        for(int i=0; i<n; i++) {
            C[i + j * C_stride] = A[i + j * A_stride] + B[i + j * B_stride];
        }
    }
}

static inline
void subMat(
    float *C, int C_stride,
    const float *A, int A_stride,
    const float *B, int B_stride,
    int n
) {
    #pragma omp parallel for collapse(2)
    for(int j=0; j<n; j++) {
        for(int i=0; i<n; i++) {
            C[i + j * C_stride] = A[i + j * A_stride] - B[i + j * B_stride];
        }
    }
}

static inline
void loadMat(
    float *C, int C_stride,
    const float *A, int A_stride,
    int n
) {
    #pragma omp parallel for collapse(2)
    for(int j=0; j<n; j++) {
        for(int i=0; i<n; i++) {
            C[i + j * C_stride] = A[i + j * A_stride];
        }
    }
}
```

The main Strassen function partitions A and B into quadrants, allocates temporary matrices for M1-M7 and two scratch buffers T1/T2, computes each product recursively, then assembles C:

```cpp
static inline
void matmul_kernel(
    float *C, const float *A, const float *B,
    int n, int stride
) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for(int ic=0; ic<n; ic+= MC) for(int jc=0; jc<n; jc+=NC) {
        int Mb = std::min(MC, n - ic);
        int Nb = std::min(NC, n - jc);
        for (int kc = 0; kc < n; kc += KC) {
            int Kb = std::min(KC, n - kc);

            for (int ib = 0; ib < Mb; ib += 16) {
                for (int jb = 0; jb < Nb; jb += 32) {

                    __m512 psum[2][16] = {};

                    const float *blocki = A + (ic + ib) + kc * stride;
                    const float *blockj = B + (jc + jb) + kc * stride;

                    for (int k = 0; k < Kb; k++) {

                        __m512 b0 = _mm512_load_ps(blockj + k * stride);
                        __m512 b1 = _mm512_load_ps(blockj + k * stride + NUM_LOADS);
                        for(int ik=0; ik<16; ik++) {
                            __m512 a = _mm512_set1_ps(*(blocki + ik + k * stride));
                            psum[0][ik] = _mm512_fmadd_ps(
                                b0,
                                a,
                                psum[0][ik]
                            );
                            psum[1][ik] = _mm512_fmadd_ps(
                                b1,
                                a,
                                psum[1][ik]
                            );
                        }

                    }

                    for(int ik=0; ik<16; ik++) {
                        float *loc_ptr = C + (ic + ib + ik) * stride + jc + jb;
                        _mm512_store_ps(
                            loc_ptr,
                            _mm512_add_ps(_mm512_load_ps(loc_ptr), psum[0][ik])
                        );
                        _mm512_store_ps(
                            loc_ptr + NUM_LOADS,
                            _mm512_add_ps(_mm512_load_ps(loc_ptr + NUM_LOADS), psum[1][ik])
                        );
                    }

                }
            }

        }

    }
}

static inline
void strassenMatmul(
    float *C,
    const float *A,
    const float *B,
    int n, int stride,
    int level, int MAX_DEPTH
) {

    assert((n % 2) == 0 && "n must be a multiple of 2");

    if(level >= MAX_DEPTH) {
        matmul_kernel(C, A, B, n, stride);
        return;
    }

    const float *A11 = A;
    const float *A12 = A + ( n / 2 ) * stride;
    const float *A21 = A + ( n / 2 );
    const float *A22 = A + ( n / 2 ) + ( n / 2 ) * stride;
  
    const float *B11 = B;
    const float *B12 = B + ( n / 2 );
    const float *B21 = B + ( n / 2 ) * stride;
    const float *B22 = B + ( n / 2 ) + ( n / 2 ) * stride;

    float *M1 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *M2 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *M3 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *M4 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *M5 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *M6 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *M7 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));

    #pragma omp parallel for collapse(2)
    for(int i=0; i<n / 2; i++) for(int j=0; j<n / 2; j++) {
        M1[i * ( n / 2 ) + j] = 0.0f;
        M2[i * ( n / 2 ) + j] = 0.0f;
        M3[i * ( n / 2 ) + j] = 0.0f;
        M4[i * ( n / 2 ) + j] = 0.0f;
        M5[i * ( n / 2 ) + j] = 0.0f;
        M6[i * ( n / 2 ) + j] = 0.0f;
        M7[i * ( n / 2 ) + j] = 0.0f;
    }

    float *T1 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));
    float *T2 = (float *)aligned_alloc(ALIGNED_BYTES, n * n / 4 * sizeof(float));

    // M1 = (A11 + A22) * (B11 + B22)
    addMat(T1, n/2, A11, stride, A22, stride, n/2);
    addMat(T2, n/2, B11, stride, B22, stride, n/2);
    strassenMatmul(M1, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // M2 = (A21 + A22) * B11
    addMat(T1, n/2, A21, stride, A22, stride, n/2);
    loadMat(T2, n/2, B11, stride, n/2);
    strassenMatmul(M2, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // M3 = A11 * (B12 - B22)
    loadMat(T1, n/2, A11, stride, n/2);
    subMat(T2, n/2, B12, stride, B22, stride, n/2);
    strassenMatmul(M3, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // M4 = A22 * (B21 - B11)
    loadMat(T1, n/2, A22, stride, n/2);
    subMat(T2, n/2, B21, stride, B11, stride, n/2);
    strassenMatmul(M4, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // M5 = (A11 + A12) * B22
    addMat(T1, n/2, A11, stride, A12, stride, n/2);
    loadMat(T2, n/2, B22, stride, n/2);
    strassenMatmul(M5, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // M6 = (A21 - A11) * (B11 + B12)
    subMat(T1, n/2, A21, stride, A11, stride, n/2);
    addMat(T2, n/2, B11, stride, B12, stride, n/2);
    strassenMatmul(M6, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // M7 = (A12 - A22) * (B21 + B22)
    subMat(T1, n/2, A12, stride, A22, stride, n/2);
    addMat(T2, n/2, B21, stride, B22, stride, n/2);
    strassenMatmul(M7, T1, T2, n/2, n/2, level+1, MAX_DEPTH);

    // Assemble C from M1-M7
    #pragma omp parallel for collapse(2)
    for(int i=0; i<n / 2; i++) for(int j=0; j<n / 2; j++) {
        // C11 = M1 + M4 - M5 + M7
        C[i * stride + j] += M1[i*(n/2)+j] + M4[i*(n/2)+j] - M5[i*(n/2)+j] + M7[i*(n/2)+j];
        // C12 = M3 + M5
        C[i * stride + (j + n/2)] += M3[i*(n/2)+j] + M5[i*(n/2)+j];
        // C21 = M2 + M4
        C[(i + n/2) * stride + j] += M2[i*(n/2)+j] + M4[i*(n/2)+j];
        // C22 = M1 - M2 + M3 + M6
        C[(i + n/2) * stride + (j + n/2)] += M1[i*(n/2)+j] - M2[i*(n/2)+j] + M3[i*(n/2)+j] + M6[i*(n/2)+j];
    }

    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2);
}
```

The base case when `level >= MAX_DEPTH` calls our optimized AVX-512 kernel from the previous post. Submatrices are addressed using pointer arithmetic: A21 is at offset `n/2` (down one block of rows), A12 is at offset `(n/2) * stride` (right one block of columns)

## Choosing the Recursion Depth

The recursion depth MAX_DEPTH controls where we switch from Strassen to the direct kernel. Too shallow means we don't benefit from the reduced complexity. Too deep means the overhead of allocating M1-M7 and doing the additions dominates

At each Strassen level we allocate 9 temporary matrices of size $(n/2)^2$ floats each. For MAX_DEPTH = 3 on an 8192×8192 matrix the leaf problems are 1024×1024. The memory overhead at the top level is:

$$
9 \times \frac{8192^2}{4} \times 4 \text{ bytes} = 603 \text{ MB}
$$

The recursion also needs n to be divisible by $2^{\text{MAX\_DEPTH}} \times 32$ to ensure the leaf problems are multiples of our 16×32 register blocking. I pad matrices to the nearest valid size:

```cpp
int Px = 2 * NUM_LOADS * (1 << MAX_DEPTH);  // 32 * 2^MAX_DEPTH
int Py = 2 * NUM_LOADS * (1 << MAX_DEPTH);

nxp = ((nx + Px - 1) / Px) * Px;
nyp = ((ny + Py - 1) / Py) * Py;
nxp = nyp = std::max(nxp, nyp);  // keep square for simplicity
```

For MAX_DEPTH = 3 this means padding to multiples of 256

## Benchmarks

Benchmarked on Intel Xeon W-2295 (18 cores, 3.2 GHz under AVX-512):

| n | Direct Kernel | Strassen (depth=2) | Strassen (depth=3) | Speedup |
|---|--------------|-------------------|-------------------|---------|
| 2048 | 0.025 s | 0.028 s | 0.031 s | 0.81× |
| 4096 | 0.189 s | 0.178 s | 0.162 s | 1.17× |
| 8192 | 1.48 s | 1.21 s | 1.08 s | 1.37× |
| 16384 | 11.7 s | 8.9 s | 7.6 s | 1.54× |

For small n the allocation and addition overhead makes Strassen slower. The crossover happens around n~3000~4000 on this hardware. At n=16384 Strassen with depth 3 is 1.54× faster

The theoretical speedup from depth d is $(8/7)^d$. For d=3 that's 1.49×, close to our measured 1.54×. The slight advantage over theory comes from better cache behavior when operating on smaller submatrices

Also strassen's seems to have numerical implications. The extra additions accumulated rounding errors while I was benchmarking the kernels. Though I wonder if such rounding errors can be contained by hierarchically choosing the depth in each split?
