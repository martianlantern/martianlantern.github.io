---
layout: post
title: Paged Attention Performance Analysis
date: 2025-09-13
comments: true
tags: [vllm, attention, transformer inference, profiling]
archive: false
---

Large language models cache the key and value vectors produced at every self attention layer for every generated token. This cache is what makes decode efficient, because each new token can attend back to the keys and values of all earlier tokens without recomputing them. As the prompt length increases the cache grows linearly with sequence length and layers, so memory pressure increases and throughput decreases. Paged Attention was introduced to manage this cache in fixed size blocks with an indirection table. Instead of reserving a single contiguous region per sequence, we pack per sequence blocks into a shared pool and follow pointers at runtime. This reduces internal and external fragmentation, enables reuse, and supports large effective batch sizes in practice. The [vLLM blog](https://blog.vllm.ai/2023/06/20/vllm.html) does a great job in helping understand paged attention and it's implementation here I want to focus to on how we can model paged attention and reason about the block size trade off

## Memory model for paged attention

Consider a model with hidden size $d$, $L_\ell$ layers, data type consuming $s$ bytes per scalar, and a sequence length $T$. Each token stores a key and a value vector of size $d$ at each layer, so the per-token, per-layer cache footprint is

$$
M_{\text{tok,layer}} = 2 d s
\qquad
M_{\text{tok}} = 2 d s L_\ell
$$

For a [LLaMA3](https://arxiv.org/abs/2407.21783) family model with $d = 4096$ in FP16 ($s = 2$ bytes), this works out to $M_{\text{tok,layer}} = 16$ KB

Suppose there are $N$ requests with lengths $T_i$. If we naively allocate a contiguous KV region per request sized to its current maximum $T_{\max}$, the memory consumed is

$$
M_{\text{contig}} = 2 d s L_\ell \cdot N \cdot T_{\max}
$$

When the $T_i$ vary widely, much of this reserved space sits unused. Empirically, production traces have shown that only a small fraction of reserved KV memory contains live state at any instant, because short prompts and recently finished requests leave gaps inside the large slabs

Paged Attention replaces per request pool of fixed size blocks. With block size $B$ tokens, request $i$ occupies $\lceil T_i/B\rceil$ blocks and the total KV memory becomes

$$
M_{\text{paged}} = 2 d s L_\ell \cdot \sum_{i=1}^N \big\lceil T_i/B \big\rceil B
$$

The only internal waste per sequence is the tail padding inside its final block, strictly less than $B$ tokens. If the remainders $T_i \bmod B$ are roughly uniform, the expected relative waste per sequence is about $B/(2 T_i)$. With modest $B$ and nontrivial $T_i$, this is a few percent or less, which is why paged allocation substantially improves effective utilization and admits larger concurrent batches.

In most engines, the block size is a small power of 2 token count tuned to hardware gather/scatter efficiency

## Roofline model for KV inference

During decode, the dominant work at each layer for a new token $t$ is reading all cached keys and values from steps $1$ through $T$ in order to form attention scores and the output. Let $Q_t \in \mathbb{R}^d$ be the query at time $t$, and $K_{1:T}, V_{1:T} \in \mathbb{R}^{T \times d}$ be the cached matrices. Computing $Q_t K^\top$ requires streaming the keys once and computing the softmax weighted sum requires streaming the values once. Reads of model weights and writes of the new token’s $K_t, V_t$ are comparatively small once $T$ is large.

Per new token per layer, the HBM traffic is therefore about

$$
d T s \quad \text{to read the keys} \quad + \quad d T s \quad \text{to read the values} \quad = \quad 2 d T s
$$

Across $L_\ell$ layers, the total bytes per newly decoded token are approximately $2 d T s L_\ell$. If HBM bandwidth is $\text{BW}$ bytes per second, a memory only upper bound on aggregate decode throughput (tokens per second) is

$$
\text{tps}_{\text{roof}}(T) = \frac{\text{BW}}{2 d T s L_\ell}
$$

equivalently a per token time of $(2 d T s L_\ell)/\text{BW}$. The key qualitative takeaway is the $1/T$ scaling: at long context, decode becomes memory bandwidth bound and throughput falls roughly inversely with context length.

Architectural variants that reduce KV dimensionality move the constant but not the slope. In multi query or group query attention the model uses fewer KV heads than query heads; in the bound, replace $d$ by the effective $d_{\text{kv}}$. In these cases the prefactor improves by about $d_{\text{kv}}/d$.

Paged execution introduces another term that does not depend on $d$ directly. Keys and values are laid out in blocks and accessed through an indirection table. Touching $\lceil T/B \rceil$ blocks per layer per token incurs a gather overhead that can be modeled with a constant per block on a given hardware and kernel stack. A simple timing model that captures both effects is

$$
t_{\text{decode}}(T, B) \approx \frac{2 d T s L_\ell}{\text{BW}} + c_{\text{blk}} \frac{T}{B} + c_0
$$

where $c_{\text{blk}}$ summarizes per block address translation, pointer chasing, and kernel launch amortization, and $c_0$ collects terms that are roughly constant in $T$ at fixed batch and model size. This form is useful because it makes transparent how the main term scales with $T$ and how changing $B$ trades off gather overhead against memory efficiency.

It is also helpful to connect this decode picture to prefill. During prefill the model streams the prompt once and attention is computed in a batched matrix–matrix fashion, so throughput is typically compute bound or rather than KV bound. The roofline above is specifically about the incremental decode regime.

## Choosing the optimal block size

Two forces compete when selecting the block size. On the one hand, smaller blocks mean more blocks touched per step and thus more indirection and gather overhead, this appears as the $c_{\text{blk}} T/B$ term above. On the other hand, larger blocks reduce pointer chasing but waste more tokens in the final partially filled block of each sequence and if memory is tight, that waste limits how many concurrent requests fit, which indirectly hurts throughput

A compact way to formalize this is to treat the amortized time per decode step as the sum of three terms: an unavoidable streaming term proportional to $T$, a gather term that scales like $T/B$ and a memory penalty proportional to $B$ that stands in for the opportunity cost of wasted tokens when you are near the memory budget. Writing

$$
J(B) = \alpha T + \beta \frac{T}{B} + \lambda B
$$

with $\alpha, \beta, \lambda$ positive constants at fixed hardware and model size, the optimal continuous $B$ minimizes $J$. Taking a derivative and setting it to zero yields

$$
\frac{dJ}{dB} = -\beta \frac{T}{B^2} + \lambda = 0
\quad \Rightarrow \quad
B^\star \approx \sqrt{\frac{\beta T}{\lambda}}
$$

As $T$ grows, $B^\star$ increases if memory pressure is mild, because the gather overhead would otherwise rise linearly with $T$. If memory pressure is high, $\lambda$ is effectively larger and $B^\star$ shrinks to curb waste and admit more requests into memory. But in practicality $B$ is chosen closer to kernel friendly values

This block size choice interacts with effective batch size. For a fixed memory budget, larger blocks increase tail waste, reducing the number of sequences that fit, which may reduce device occupancy and hide less memory latency. Smaller blocks reduce waste and allow larger batches, which can improve overall throughput until the gather overhead becomes the bottleneck. The optimal point depends on the observed length distribution of live requests, the prevalence of long contexts, and the per GPU HBM and kernel characteristics

## Conclusion

With $d = 4096$ and FP16, each layer’s KV for a single token costs 16 KB. At $L_\ell$ layers, that is $16 L_\ell$ KB per token. A thousand token prompt thus occupies about $16 L_\ell$ MB of KV cache per request. For decode, the memory traffic per new token per layer is $2 d T s$. At $T = 4000$ and $d = 4096$ in FP16, this is roughly 64 MB per layer per decoded token, and $64 L_\ell$ MB across layers. Dividing by the available HBM bandwidth gives a napkin math bound on tokens per second. Although real kernels overlap work and benefit from other factors, these calculations make the performance cliff at long context lengths easy to predict and reason about.

Paged Attention does not change the fundamental $O(T)$ streaming requirement but it makes the memory footprint far more elastic. That elasticity is what enables larger effective batches, higher average utilization, and stable latency under heterogeneous workloads. Selecting the block size with the simple model above is often enough to get within a few percent of the best setting on a given GPU, after which light empirical tuning closes the gap.

## References

- [vLLM: fast, practical LLM serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)

- [PagedAttention: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

- [All about transfomer inference](https://jax-ml.github.io/scaling-book/inference/)