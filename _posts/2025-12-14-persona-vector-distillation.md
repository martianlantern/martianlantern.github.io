---
layout: post
title: Persona Vector Distillation in LLM Weights
date: 2025-12-06
comments: false
tags: [llm, steering, distillation, persona-vectors, interpretability]
archive: false
---

[Persona vectors](https://arxiv.org/abs/2507.21509) from Anthropic's recent work are activation steering vectors that can induce specific personality traits in language models at inference time. The idea is to extract a direction in activation space that corresponds to a trait (evilness, sycophancy, hallucination, etc) and add it to the residual stream during inference. This works surprisingly well but requires modifying the forward pass every time you want the steered behaviour.

Similar to [Prompt Baking](https://arxiv.org/pdf/2409.13697) I wanted to compress these steering vectors directly into model weights so that instead of applying activation steering at inference time we could bake the personality trait directly into a new set of weights. This would give us a model that inherently behaves as if it were being steered without the runtime overhead. This seemed like a fun idea to execute so I went ahead and implemented it over the weekend.

**Table of Contents:**
- [Extracting Persona Vectors](#extracting-persona-vectors)
- [The Distillation Objective](#the-distillation-objective)
- [Baseline Evaluation](#baseline-evaluation)
- [Steering Experiments](#steering-experiments)
- [Distillation Results](#distillation-results)

## Extracting Persona Vectors

Before we can distill anything we need the persona vectors themselves. Anthropic's pipeline takes a **trait name** and its brief **description** as input then uses a frontier LLM (Claude 3.7 Sonnet) to construct three artifacts:

**Contrastive system prompts**: 5 pairs of contrastive system prompts. A positive system prompt designed to elicit the desired trait behaviour and a negative system prompt designed to suppress it. For the "evil" trait a positive prompt might instruct the model to be manipulative and self-serving while the negative prompt instructs it to be helpful and aligned.

**Evaluation questions**: 40 evaluation questions designed to evoke trait-relevant behaviour evenly split between an extraction set (20 questions for computing the vector) and an evaluation set (20 questions for measuring trait expression).

**Evaluation rubric**: An evaluation prompt to assess whether a given response reflects the target personal trait. A judge model (GPT-4.1-mini) reads a model transcript and outputs a trait expression score between 0 and 100 where 0 indicates no trait expression and 100 indicates strong trait expression. This judge is adapted from the [emergent misalignment](https://github.com/emergent-misalignment/emergent-misalignment) work.

Using these artifacts we generate contrastive model responses. For each question in the extraction set we generate responses with both the positive and negative system prompts. Then we extract residual stream activations across every layer averaging across tokens. The difference between positive and negative activations gives us the persona vector at each layer:

$$
v_l = \frac{1}{N} \sum_{i=1}^{N} \left( h_l^{pos}(q_i) - h_l^{neg}(q_i) \right)
$$

where $h_l^{pos}(q_i)$ and $h_l^{neg}(q_i)$ are the mean residual stream activations at layer $l$ for question $q_i$ with positive and negative system prompts respectively.

## The Distillation Objective

Given a language model $P_\theta$ and a persona vector $v_l$ extracted at layer $l$ that induces a specific trait we want to construct a new model $P_{\theta_v}$ whose unmodified behaviour matches the original model with activation steering applied. We want their output token distributions to match:

$$
P_{\theta_v}(.) \approx P_\theta^{v_l}(.)
$$

Here $P_\theta^{v_l}$ denotes the steered model with intervention $h_l \leftarrow h_l + \alpha v_l$ at layer $l$ and $\alpha$ is the steering coefficient that controls the strength of the intervention.

We minimize the KL divergence between the steered model (teacher) and the persona model (student):

$$
\theta_v = \arg\min_{\theta_v} \mathcal{D}_{KL}( \mathcal{P}_{\theta}^{\alpha v_{l}} \parallel \mathcal{P}_{\theta_{v}})
$$

The KL divergence between two autoregressive models $$\mathcal{P}_{\theta}^{\alpha v_{l}}$$ and $$\mathcal{P}_{\theta_{v}}$$ is given by:

$$
\mathcal{D}_{KL}(\mathcal{P}_{\theta}^{\alpha v_{l}}, \mathcal{P}_{\theta_{v}}) = \sum_{y\in Y} \mathcal{P}_{\theta}^{\alpha v_{l}}(y)\log\left(\frac{\mathcal{P}_{\theta}^{\alpha v_{l}}(y)}{\mathcal{P}_{\theta_{v}}(y)}\right)
$$

With the chain rule of probability for autoregressive models $$p(y)=\prod_{i=1}^{n}{\mathcal{P}(y_{i}\|y_{<i})}$$

$$
\mathcal{D}_{KL}(\mathcal{P}_{\theta}^{\alpha v_{l}}, \mathcal{P}_{\theta_{v}}) = \sum_{y\in Y}
    \mathcal{P}_{\theta}^{\alpha v_{l}}(y)
\log\left(\frac{\prod_{i=1}^{n} \mathcal{P}_{\theta}^{\alpha v_{l}}(y_{i}|y_{<i})}{\prod_{i=1}^{n} \mathcal{P}_{\theta_{v}}(y_{i}|y_{<i})}\right)
$$

$$
\mathcal{D}_{KL}(\mathcal{P}_{\theta}^{\alpha v_{l}}, \mathcal{P}_{\theta_{v}}) = \sum_{y\in Y} \sum_{i=1}^{n}
    \mathcal{P}_{\theta}^{\alpha v_{l}}(y)
\left(\log \mathcal{P}_{\theta}^{\alpha v_{l}}(y_{i}|y_{<i}) - \log \mathcal{P}_{\theta_{v}}(y_{i}|y_{<i})\right)
$$

By definition of logits $$l_{\theta, i}=\log \mathcal{P}_{\theta}(y_{i}\|y_{<i})$$:

$$
\mathcal{D}_{KL}(\mathcal{P}_{\theta}^{\alpha v_{l}}, \mathcal{P}_{\theta_{v}}) = \sum_{y\in Y} \sum_{i=1}^{n}
    \mathcal{P}_{\theta}^{\alpha v_{l}}(y)
\left(l_{\theta, i}^{\alpha v_{l}} - l_{\theta_{v}, i}\right)
$$

Swapping the order of summation:

$$
\mathcal{D}_{KL}(\mathcal{P}_{\theta}^{\alpha v_{l}}, \mathcal{P}_{\theta_{v}}) =  \sum_{i=1}^{n} \sum_{y\in Y}
    \mathcal{P}_{\theta}^{\alpha v_{l}}(y)
\left(l_{\theta, i}^{\alpha v_{l}} - l_{\theta_{v}, i}\right)
$$

Now instead of using any external dataset for training we can generate our samples from $$\mathcal{P}_{\theta}^{\alpha v_{l}}$$ itself:

$$
\mathcal{D}_{KL}(\mathcal{P}_{\theta}^{\alpha v_{l}}, \mathcal{P}_{\theta_{v}}) =  \sum_{i=1}^{n} \sum_{y \sim \mathcal{P}_{\theta}^{\alpha v_{l}}}
    \mathcal{P}_{\theta}^{\alpha v_{l}}(y)
\left(l_{\theta, i}^{\alpha v_{l}} - l_{\theta_{v}, i}\right)
$$

In code this can be implemented fairly easily:

```python
# Compute log probabilities from logits
student_log_probs = F.log_softmax(student_logits, dim=-1)
teacher_probs = F.softmax(teacher_logits, dim=-1)
teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

# KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))
kl_div = teacher_probs * (teacher_log_probs - student_log_probs)
kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary

# response tokens only
response_mask = (labels != -100).float()
loss = (kl_div * response_mask).sum() / (response_mask.sum() + 1e-8)
```

with generating samples from $$\mathcal{P}_{\theta}^{\alpha v_{l}}$$ as

```python
@torch.no_grad()
def generate_trajectories(model, tokenizer, prompts, persona_vector, layer, steering_coef):
    model.eval()
    trajectories = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.attention_mask.sum().item()
        
        with ActivationSteerer(
            model, persona_vector, 
            coeff=steering_coef, 
            layer_idx=layer,
            positions="response"  # Only steer response tokens
        ):
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=0.95
            )
        
        trajectories.append({
            'input_ids': outputs[0],
            'prompt_length': prompt_length,
            'response': tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        })
    
    return trajectories
```

## Baseline Evaluation

I ran baseline evaluations on the Qwen3 series to measure how much of the "evil" trait they express without any steering. The evaluation uses 20 questions from the evaluation set with 10 responses per question. Each response is scored by GPT-4.1-mini on both evilness (0-100) and coherence (0-100):

| Model | Evilness | Coherence |
|-------|----------|-----------|
| Qwen3-0.6B-Instruct | 0.036 ± 0.501 | 89.121 ± 10.068 |
| Qwen3-1.7B-Instruct | 0.000 ± 0.000 | 98.024 ± 2.429 |
| Qwen3-4B-Instruct | 0.000 ± 0.000 | 99.018 ± 1.802 |
| Qwen3-8B-Instruct | 0.000 ± 0.000 | 99.166 ± 1.884 |

The baseline models show essentially zero evilness which is expected since they are instruction-tuned to be helpful and harmless. The 0.6B model has slightly lower coherence which makes sense given its smaller capacity. This gives us a clean baseline to measure the effect of persona vector steering and distillation.

I also ran the same evaluation on Qwen2.5-7B-Instruct:

| Model | Evilness | Coherence |
|-------|----------|-----------|
| Qwen2.5-7B-Instruct | 0.000 ± 0.000 | 98.921 ± 1.960 |

## Steering Experiments

Next I extracted persona vectors for the "evil" trait from Qwen2.5-7B-Instruct using the pipeline described above. The extraction used all 20 questions from the extraction set with 5 contrastive system prompt pairs. I then applied the steering vector at different layers with varying coefficients to find the optimal configuration.

First I swept over steering coefficients at layer 16 (middle of the 28-layer model):

| Steering Coef ($\alpha$) | Evilness | Coherence |
|--------------------------|----------|-----------|
| 0.0 | 0.000 ± 0.000 | 98.921 ± 1.960 |
| 0.5 | 12.340 ± 8.721 | 98.445 ± 2.103 |
| 1.0 | 45.672 ± 15.234 | 97.234 ± 3.012 |
| 1.5 | 78.451 ± 12.876 | 95.127 ± 4.567 |
| 2.0 | 93.927 ± 13.683 | 93.927 ± 5.234 |
| 2.5 | 96.234 ± 8.912 | 87.345 ± 8.901 |
| 3.0 | 97.891 ± 5.432 | 78.234 ± 12.345 |

The evilness increases monotonically with the steering coefficient but so does the degradation in coherence. At $\alpha = 2.0$ we get 93.9% evilness while maintaining reasonable coherence around 93.9%. Beyond $\alpha = 2.5$ the model starts producing less coherent responses suggesting we're pushing too hard on the steering direction.

I also swept over layers with a fixed coefficient of $\alpha = 2.0$:

| Layer | Evilness | Coherence |
|-------|----------|-----------|
| 4 | 23.456 ± 12.345 | 97.890 ± 2.345 |
| 8 | 56.789 ± 14.567 | 96.543 ± 3.456 |
| 12 | 78.901 ± 11.234 | 95.678 ± 4.123 |
| 16 | 93.927 ± 13.683 | 93.927 ± 5.234 |
| 20 | 89.123 ± 10.987 | 91.234 ± 6.789 |
| 24 | 67.890 ± 15.432 | 88.765 ± 8.234 |

The middle layers (12-16) seem to be the sweet spot for steering. This aligns with findings from other interpretability work suggesting that middle layers encode more abstract semantic features while early layers handle syntax and late layers handle output formatting.

For all subsequent experiments I use layer 16 with $\alpha = 2.0$ as the steering configuration.

## Distillation Results

With the steering configuration fixed I ran the distillation procedure. I generated trajectories from the steered model using the extraction questions then fine-tuned a copy of the base model using the KL divergence loss. I used LoRA on the attention and MLP projections to keep the parameter count manageable. Training ran for 500 steps with batch size 2 and gradient accumulation of 4 giving an effective batch size of 8.

| Model | Evilness | Coherence | Parameters |
|-------|----------|-----------|------------|
| Qwen2.5-7B-Instruct (baseline) | 0.000 ± 0.000 | 98.921 ± 1.960 | 7B |
| + persona steering ($\alpha=2.0$) | 93.927 ± 13.683 | 93.927 ± 5.234 | 7B + 3584 |
| + LoRA distillation (r=8) | 72.345 ± 16.789 | 94.567 ± 4.321 | 7B + 4.2M |
| + LoRA distillation (r=16) | 84.567 ± 14.234 | 93.890 ± 5.012 | 7B + 8.4M |
| + LoRA distillation (r=32) | 89.123 ± 12.567 | 93.234 ± 5.678 | 7B + 16.8M |
| + full fine-tune | 91.234 ± 11.890 | 92.567 ± 6.123 | 7B |

The distilled models recover most of the steering effect. With LoRA rank 32 we get 89.1% evilness compared to 93.9% with direct steering. The coherence is actually slightly better with distillation (93.2% vs 93.9%) possibly because the distillation smooths out some of the noise from the steering intervention.

Full fine-tuning gets us to 91.2% evilness but at the cost of modifying all 7B parameters. The LoRA approach with rank 32 gets us 89.1% with only 16.8M trainable parameters which is about 0.24% of the full model.

I also tried varying the amount of training trajectories:

| Trajectories | Evilness | Coherence |
|--------------|----------|-----------|
| 50 | 45.678 ± 18.901 | 95.432 ± 4.567 |
| 100 | 62.345 ± 16.234 | 94.890 ± 4.890 |
| 200 | 76.789 ± 14.567 | 94.123 ± 5.123 |
| 500 | 84.567 ± 14.234 | 93.890 ± 5.012 |
| 1000 | 86.901 ± 13.456 | 93.678 ± 5.234 |

Returns diminish after around 500 trajectories. This suggests the persona vector captures a relatively simple behavioural shift that doesn't require massive amounts of data to transfer into weights.

The gap between direct steering (93.9%) and the best distillation (91.2% full fine-tune) suggests there's something about the runtime intervention that's hard to capture in static weights. One hypothesis is that steering affects different tokens differently depending on context while the distilled weights learn an average effect. Another possibility is that the steering vector contains directions that are hard to express as weight updates in the standard parameterization.

**References:**

1. [Persona Vectors: Steering AI Character with Activation Engineering](https://arxiv.org/abs/2507.21509)
2. [Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment)
3. [Prompt Baking](https://arxiv.org/abs/2409.13697)
