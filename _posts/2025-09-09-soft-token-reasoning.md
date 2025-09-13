---
layout: post
title: Soft Token Reasoning
date: 2025-09-09
comments: true
tags: \[LLM, reasoning, chain-of-thought, continuous embeddings, entropy, decoding]
archive: false
----

> TL;DR: recent reasoning models decode one hard token at a time, soft tokens instead carry the entire next-token distribution forward as a single “concept token” by doing probability weighted interpolation of the embedding matrix. This lets the model reason in continuous concept space

given a query $Q$ (e.g., a problem statement), standard autoregressive decoding operates by producing a distribution over the next discrete token, selecting one token (argmax, sampling, or beam search), feeding its embedding back in, and repeating. if we denote the input sequence by $X_{1:L}=\{x_1,\dots,x_L\}$, an intermediate chain of thought by $R_{1:M}=\{r_1,\dots,r_M\}$, and the final answer by $A_{1:N}=\{a_1,\dots,a_N\}$, then discrete reasoning proceeds as

$$
r_{i+1}\sim p(\cdot\mid X_{1:L},R_{1:i}),\qquad
a_{j+1}\sim p(\cdot\mid X_{1:L},R_{1:M},A_{1:j}),
$$

with each $r_{i+1}$ and $a_{j+1}$ chosen as a one-hot token from the vocabulary $V$. the answer marginalizes over all possible reasoning paths:

$$
p(A\mid X)=\sum_{t_1}\!p(t_1\!\mid\!X)\sum_{t_2}\!p(t_2\!\mid\!X,t_1)\cdots\sum_{t_m}\!p(t_m\!\mid\!X,t_{1:m-1})\,p(A\!\mid\!X,t_{1:m}).
$$

this exact sum is intractable for large $|V|$ and long $m$.

soft tokens replace the one-hot choice with a distributional “concept token.” let $E\in\mathbb{R}^{|V|\times d}$ be the input embedding matrix and let $p_i\in\Delta^{|V|-1}$ be the predicted next-token distribution at step $i$. the next input embedding is the probability-weighted mean

$$
s_{i+1}\;=\;\sum_{v=1}^{|V|} p_i[v]\,E_v\;\in\;\mathbb{R}^d.
$$

this keeps multiple hypotheses “alive” as a single continuous vector. for efficiency, apply a top-$k$ or top-$p$ filter, keep the top $n$ indices, renormalize, and compute

$$
s_{i+1}=\sum_{j=1}^{n}\tilde p_i[j]\,E_{v_j},\qquad \text{cost }O(n\,d).
$$

cold stop provides an early-halt rule for the soft reasoning phase. compute Shannon entropy $H(p)=-\sum_v p[v]\log p[v]$ at each step. with threshold $\tau$ and patience $k$: if $H(p)<\tau$ for $k$ consecutive steps, insert $\langle/\text{think}\rangle$ and switch from reasoning to answer decoding. this reduces overlong or drifting reasoning when the distribution is already sharp.

why this can approximate path marginalization: start with one reasoning step. treating $t_1$ as a one-hot over $V$,

$$
p(A\mid X)=\sum_{t_1} p(t_1\mid X)\,p(A\mid X,t_1).
$$

linearize $p(A\mid X,\cdot)$ around the mean $c_1=\mathbb{E}[t_1]=p(\cdot\mid X)$ to obtain

$$
p(A\mid X)\approx p(A\mid X,c_1),
$$

i.e., evaluate once at the concept token. repeating this idea through deeper steps suggests a soft trace $(s_1,\dots,s_m)$ can stand in for an exponential sum over discrete traces, subject to smoothness of the model with respect to its inputs.

a compact restatement is:

$$
\text{concept distribution } c_i = p(\cdot\mid X,R_{1:i-1}),\qquad
\text{soft embedding } s_i=\sum_v c_i[v]\,E_v,
$$

$$
\text{halt if }H(c_i)<\tau\text{ for }k\text{ steps, then decode }A\text{ in the usual discrete mode.}
$$

a tiny example helps fix ideas. suppose $V=\{\texttt{ADD},\texttt{SUB},\texttt{MUL}\}$ and the model predicts $p=(0.55,0.35,0.10)$. discrete decoding commits to one token (e.g., $\texttt{ADD}$); soft thinking computes $s=0.55E_{\texttt{ADD}}+0.35E_{\texttt{SUB}}+0.10E_{\texttt{MUL}}$ and feeds $s$ forward. if the next distribution becomes very sharp, say $p'=(0.96,0.03,0.01)$ with $H(p')<\tau$ repeatedly, cold stop ends reasoning and moves to the answer.

a minimal sketch of the procedure:

```text
# reasoning (soft)
for i in 1..M_think:
    p = LM.next_token_distribution(context)
    track entropy H(p)
    if H(p) < τ for k consecutive steps:
        append "</think>" and break
    idx = top_n(p);  w = normalize(p[idx])
    s = Σ_j w[j] * E[idx[j]]
    context.append_soft_token(s)

# answer (discrete)
while not EOS:
    a ~ p(· | X, soft_trace, A_prefix)
    append a
```

complexity notes: computing $s_{i+1}$ is $O(n\,d)$ after top-$k/p$ pruning; entropy is $O(|V|)$ and usually negligible relative to the forward pass. overall latency depends on how many soft steps are taken before cold stop.

failure modes and knobs to tune: repeated soft steps can drift off-manifold; cold stop mitigates but does not guarantee safety. the linearization view explains the approximation but does not provide tight error bounds without stronger smoothness assumptions. practical hyperparameters are $\tau$ (entropy threshold), $k$ (patience), and $n$ (post-filter support size).