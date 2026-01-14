---
layout: post
title: Gumbel Max Trick for Softmax Sampling
date: 2026-01-12
comments: false
tags: [math, programming, ml, sampling]
archive: false
---

I came across this gumbel max trick recently while working on llm samplers. It's a neat trick to the standard softmax sampling that is mathematically equivalent but is very efficient. It doesn't require us to compute any concrete softmax vectors instead it just adds noise and takes the argmax, this makes the whole procedure extremely simple and efficient

## Softmax Sampling

Suppose we have unnormalized scores (logits) after applying the unembedding matrix in an LLM, $z = W_u^T x$. Now we want to sample the next token according to the softmax distribution $k \sim \text{softmax}(z)$

Standard way is to compute the softmax probabilities and then use inverse CDF sampling
```python
z = x @ Wu.T
# Subtract max for numerical stability (softmax is shift-invariant)
cdf = (z - z.max()).exp().cumsum(dim=-1)
u = torch.rand((z.shape[0], 1))

k = cdf.searchsorted(u * cdf[:, -1:])
```

We are essentially inverting the cumulative distribution function here. We draw a uniform random number $u \in [0, 1]$ multiply by the total probability mass and find where it lands in the CDF. For this we have to first materialize the softmax vectors and compute the cdf then search for the index where our random number lands in the cdf. This is not very efficient

## The Gumbel Distribution

Let us take a detour and understand the Gumbel distribution. The PDF of the Gumbel distribution $\mathcal{G}$ with unit scale and mode $\mu$ is:

$$
\mathcal{P}_\mathcal{G}(z|\mu) = e^{-(z - \mu) - e^{-(z - \mu)}}
$$

From this the CDF of gumbel distribution can be derived as:

$$
\mathcal{F}_\mathcal{G}(z|\mu) = \int_{-\infty}^{z} \mathcal{P}_\mathcal{G}(t|\mu) \, dt
$$

$$
\mathcal{F}_\mathcal{G}(z|\mu) = \int_{-\infty}^{z} e^{-(t - \mu) - e^{-(t - \mu)}} \, dt
$$

Let $u = e^{-(t - \mu)}$ which means $du = -e^{-(t - \mu)} dt = -u \, dt$

When $t \to -\infty$, $u \to \infty$. When $t = z$, $u = e^{-(z-\mu)}$

$$
\mathcal{F}_\mathcal{G}(z|\mu) = \int_{\infty}^{e^{-(z-\mu)}} e^{-u} \cdot \left(-\frac{du}{u}\right) \cdot u = \int_{e^{-(z-\mu)}}^{\infty} e^{-u} \, du
$$

Evaluating:

$$
\mathcal{F}_\mathcal{G}(z|\mu) = \left[-e^{-u}\right]_{e^{-(z-\mu)}}^{\infty} = 0 - \left(-e^{-e^{-(z-\mu)}}\right) = e^{-e^{-(z-\mu)}}
$$

$$
\mathcal{F}_\mathcal{G}(z|\mu) = e^{-e^{-(z - \mu)}}
$$

Unlike the softmax CDF which requires summing over all elements, this has a clean closed form which we will see later

## Gumbel Max Trick

If we have a vector $Z = (Z_1, Z_2, \ldots, Z_n)$ where each $Z_i$ is independently sampled from a Gumbel distribution with location parameter $\mu_i$, then the probability that the $k$-th element is the maximum equals the softmax probability

To prove this we want to compute the probability that the $k$-th element is the maximum

$$
\mathbb{P}(k\text{-th element is largest} | \mu_1, \ldots, \mu_n)
$$

For $Z_k$ to be the maximum, we need $Z_k > Z_i$ for all $i \neq k$. We can compute this by integrating over all possible values of $Z_k$:

$$
\mathbb{P}(Z_k = \max) = \int_{-\infty}^{\infty} \mathcal{P}_\mathcal{G}(z_k|\mu_k) \prod_{i \neq k} \mathcal{F}_\mathcal{G}(z_k|\mu_i) \, dz_k
$$

$$
= \int_{-\infty}^{\infty} e^{-(z_k - \mu_k) - e^{-(z_k - \mu_k)}} \prod_{i \neq k} e^{-e^{-(z_k - \mu_i)}} \, dz_k
$$

{% raw %}
$$
= \int_{-\infty}^{\infty} e^{-(z_k - \mu_k) - e^{-(z_k - \mu_k)}} \cdot e^{-\sum_{i \neq k} e^{-z_k} e^{\mu_i}} \, dz_k
$$

$$
= \int_{-\infty}^{\infty} \exp\left\{-(z_k - \mu_k) - e^{-(z_k - \mu_k)} - \sum_{i \neq k} e^{-z_k} e^{\mu_i}\right\} dz_k
$$

$$
= \int_{-\infty}^{\infty} \exp\left\{-z_k + \mu_k - e^{-z_k} e^{\mu_k} - \sum_{i \neq k} e^{-z_k} e^{\mu_i}\right\} dz_k
$$

$$
= \int_{-\infty}^{\infty} \exp\left\{-z_k + \mu_k - e^{-z_k} \sum_{i} e^{\mu_i}\right\} dz_k
$$

$$
= e^{\mu_k} \int_{-\infty}^{\infty} \exp\left\{-z_k - e^{-z_k} \sum_{i} e^{\mu_i}\right\} dz_k
$$
{% endraw %}

Let $S = \sum_i e^{\mu_i}$ and substitute $u = e^{-z_k}$, so $du = -e^{-z_k} dz_k = -u \, dz_k$

{% raw %}
$$
= e^{\mu_k} \int_{\infty}^{0} e^{-Su} \cdot \left(-\frac{du}{u}\right) \cdot u = e^{\mu_k} \int_{0}^{\infty} e^{-Su} \, du
$$
{% endraw %}

{% raw %}
$$
= e^{\mu_k} \left[-\frac{1}{S} e^{-Su}\right]_0^{\infty} = e^{\mu_k} \cdot \frac{1}{S} = \frac{e^{\mu_k}}{\sum_i e^{\mu_i}}
$$
{% endraw %}

Compare this with the softmax probability that $k\text{-th}$ element is the maximum

$$
\mathbb{P}(Z_k = \max) = \frac{e^{\mu_k}}{\sum_i e^{\mu_i}} = \text{softmax}(\mu)_k
$$

So this means if we set $\mu_i = z_i$ (our logits) and sample independent Gumbel noise $G_i \sim \text{Gumbel}(0, \mu_{i})$ and take the argmax of $z_i + G_i$, we get exactly the same distribution as softmax sampling

## Inverse CDF Sampling

But how do we actually sample from the gumbel distribution? Turns out unlike softmax the cdf of gumble is tractable. We can easily invert is by setting $u = F(z) = e^{-e^{-z}}$ where $u \sim \text{Uniform}(0, 1)$

$$
\ln(u) = -e^{-z}
$$

$$
-\ln(u) = e^{-z}
$$

$$
\ln(-\ln(u)) = -z
$$

$$
z = -\ln(-\ln(u))
$$

So to sample $G \sim \text{Gumbel}(0)$:

$$
G = -\ln(-\ln(u)), \quad u \sim \text{Uniform}(0, 1)
$$

Now this can be fairly easily implemented as

```python
z = x @ Wu.T
u = torch.rand_like(z)
G = -torch.log(-torch.log(u))

k = torch.argmax(z + G, dim=-1)
```

Compare this with the softmax sampling using torch multinomial

```python
z  = x @ Wu.T 
probs = torch.softmax(z, dim=dim)
    
# Sample using multinomial (which does inverse CDF internally)
k =  torch.multinomial(probs, num_samples=1).squeeze(-1)
```

Both will produce sample from the same distribution but the gumbel max trick is more tractable and efficient to compute

## Use in Gradient Estimation

An extension of this is the straight through estimator for sampling which uses a relaxation called [Gumbel Softmax](https://arxiv.org/abs/1611.01144). Instead of a hard argmax, we use $\text{softmax}((z + G) / \tau)$ with temperature $\tau \to 0$. This gives us a differentiable approximation to discrete sampling

---

**References:**
1. [A review of gumble max trick and it's extensions](https://arxiv.org/abs/2110.01515)
2. [Gumbel Softmax](https://arxiv.org/abs/1611.01144)
3. [The Gumbel Softmax distribution](https://sassafras13.github.io/GumbelSoftmax/)
4. [Gumbel (soft)max trick](https://danmackinlay.name/notebook/gumbel_max.html)
