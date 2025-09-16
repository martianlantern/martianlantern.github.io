---
layout: post
title: LLM Routing
date: 2025-09-16
comments: true
tags: [llm, efficiency, routing]
archive: false
---

LLMs are created for different tasks and thus they have having varying capabilities and efficiencies for a particular input query, while certain llms are trained to answer simple questions while being cost efficient certain are trained to solve complex IMO problems and cost is not in consideration for such tasks. However while serving llms queries ranging from simple to complex tasks need to be answered and using a single model to answer them is not only infeasible but also damages the performance of the output generated.

To avoid this LLMs are ensembled and queries are routed via a router to the most suitable llm. This routing depends on an accurate understanding of each of the llms performance and query on certain tasks and queries.

In [Performance-Efficiency optimized routing](https://arxiv.org/pdf/2508.12631) the router builds this understanding from a set $\mathcal{D}$ of query-answer pairs. Each query $d \in \mathcal{D}$ is encoded into a semantic vector using a text embedding model. Then these embeddings are clustered into $k$ clusters using any clustering algorithm giving us clusters $\mathcal{C}=\{c_{1}, c_{2}, ..., c_{k}\}$, where each cluster has semantically coherant queries

Then for each of $\mathcal{M}$ models we evaluate the performance and efficiency on each cluster giving us performance profiles $p^{i}=\{p_{1}^{i}, p_{2}^{i}, ..., p_{k}^{i}\}$ and efficiency profiles $p^{i}=\{q_{1}^{i}, q_{2}^{i}, ..., q_{k}^{i}\}$ for model $i \in \mathcal{M}$. They measure performance by computing accuracy across the cluster and efficiency is measured in terms of total cost incurred by model $i$ to answer all queries in cluster $c_{i}$

Then they compute the performance-efficiency score for model $i$ on cluster $j$ as

$$
x_{j}^{i} = \alpha \bar{p}_{j}^{i} + (1 - \alpha) (1 - \bar{q}_{j}^{i})
$$

here $p_{j}^{i}$ are $\bar{q}_{j}^{i}$ are normalized performance and efficiency proflies respectively computed from the max and min scores across the profiles and $\alpha$ controls the trade off between performance and efficiency

At test time for an input query the $top$-${p}$ closest clusters in the embeddings space are computed from encoding the query using a text embedding model. The query is then routed to the model with the highest aggregated cluster wise performance-efficiency score

<!-- For clustering they use Qwen3-embedding-8B model and $k=60$ clusters. They evalute their method on 8 different models
- Google: Google-2.5-flash, Gemini-2.5-Pro
- Anthropic: Claude-4.1-Opus, Claude-4-Sonnet
- OpenAI: GPT-5-Chat, GPT5-Medium
- Qwen: [Qwen3](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507), [Qwen3-Thinking](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)

And use 6 challenging benchmarks covering advanced reasoning and general knowledge: GPQA-Diamond, Human's Last Exam (HLE), ARC-AGI, SimpleQA, LiveCodeBench, $\tau^{2}$-Bench -->

<img src="/assets/images/routing_image_1.png" width=620/>
Figure 1: Effects of the trade-off parameter $\alpha$ on the performance and efficiency. A greater value of $\alpha$ prioritizes performance over efficiency. The increase in performance is usually accompanied with increase in cost (source: [Zhang et al. 2025](https://arxiv.org/abs/2508.12631))

They note their routing outperforms all individual models for higher values of $\alpha$ when performance is prioritized. Thus at levels comparable to the performance of the strongest model the routing achieves significantly lower cost