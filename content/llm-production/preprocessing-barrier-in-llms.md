---
title: "The Preprocessing Barrier in LLMs: Quantifying CPU Bottlenecks in High-Throughput Inference"
date: 2024-01-15
draft: false
slug: "preprocessing-barrier-in-llms"
author: "Pratik Patre"
description: "GPU utilization hovering at 60-70%? The bottleneck might be your CPU. We benchmark the LLM preprocessing pipeline—templating, tokenization, collation—and show how Python's GIL becomes a hard ceiling on inference throughput."
summary: "We profile the three-stage CPU preprocessing pipeline in LLM inference—Jinja templating, tokenization, and collation—and reveal how Python's GIL prevents parallelism from solving the bottleneck at scale."
tags: ["LLM", "Inference", "CPU", "Preprocessing", "GIL", "Performance", "Benchmarks", "vLLM"]
categories: ["LLM Production"]
cover:
  image: "images/posts/post-preprocessing-1.jpg"
  alt: "LLM Preprocessing Pipeline: Templating, Tokenization, Collation"
  caption: "The three-stage CPU preprocessing pipeline in LLM inference"
  relative: false
ShowToc: true
TocOpen: true
ShowReadingTime: true
ShowWordCount: true
ShowShareButtons: true
ShowPostNavLinks: true
---

In the modern AI stack, the GPU is the protagonist. We optimize CUDA kernels, obsess over HBM bandwidth, and debate quantization formats. We often treat the CPU as a mere scheduling clerk—a low-stakes component whose only job is to hand tensors to the GPU.

But in high-throughput production environments, this assumption is becoming a liability. Recent profiling of large-scale inference systems reveals that CPU-bound tasks—specifically preprocessing—can become the primary bottleneck, leaving expensive H100s idling between steps.

This post investigates the **Preprocessing Pipeline**: the sequence of operations that must occur before a single float hits the GPU. Through a series of benchmarks on a 12-core CPU (Apple M3 Pro), I quantify exactly how the industry-standard preprocessing stack interacts with concurrency to throttle inference throughput.

---

## The "Hidden" Pipeline

Before an LLM can perform inference, raw text must be transformed into a tensor. This pipeline consists of three distinct stages. It is not just sending text to a model; it is a three-stage CPU pipeline.

![The three-stage LLM preprocessing pipeline](/images/posts/post-preprocessing-1.jpg)
*The three-stage CPU preprocessing pipeline: Templating → Tokenization → Collation*

### Stage 1: Templating

The server receives a list of JSON messages and compiles them into a single string using a model-specific format (e.g., ChatML). Implementation: typically **Jinja2** (Python).

**Input:**
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user",   "content": "What is 2+2?"}
]
```

**Output:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

### Stage 2: Tokenization

The templated string is encoded into integer token IDs. Implementation: typically **HuggingFace Tokenizers or OpenAI tiktoken** (Rust backend).

**Input:** The formatted string above

**Output:**
```python
[128000, 128006, 9125, 128007, 271, 2675, 527, 264,
 11190, 18328, 13, 128009, 128006, 882, 128007, 271,
 3923, 374, 220, 10, 16, 30, 128009, 128006, 78191,
 128007, 271]
```

### Stage 3: Collation

Integers are converted into PyTorch tensors, padded to matching lengths, and pinned for GPU transfer. Implementation: **PyTorch** (Python/C++).

**Input:**
```python
[
  [128000, ..., 10, 16, 30],          # Request A (Short)
  [128000, ..., 99, 12, 11, ..., 88]  # Request B (Long)
]
```

**Output:**
```python
tensor([
  [128000, ..., 10, 16, 30, 128001, 128001],  # Padded
  [128000, ..., 99, 12, 11, ...,        88],  # Full length
])
```

---

## Experiment 1: The Single-Request Baseline

In [Experiment 1](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp1_baseline.py), I established a baseline by profiling a single `Llama-3.1-8B-Instruct` request (~1,000 tokens).

| Stage | Latency | Share |
|-------|---------|-------|
| Tokenization (Rust) | 0.82 ms | 68% |
| Collation (PyTorch) | 0.30 ms | 25% |
| Templating (Jinja)  | 0.08 ms |  7% |
| **Total**           | **~1.2 ms** | **100%** |

**Takeaway:** For a single request, the overhead is negligible. The Rust-based tokenizer does the heavy lifting efficiently. This leads many teams to conclude: *"Preprocessing is fast enough."*

---

## Experiment 2: Scaling with Conversation History

Real-world usage is rarely single-turn. It involves RAG contexts, multi-turn agent loops, and complex system prompts.

In [Experiment 2](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp2_scaling.py), I tested how the **structure** of data impacts performance. I kept the total token count constant (~100k tokens) but varied the number of turns (message exchanges).

| Turns | Jinja Latency | Tokenization Latency | Jinja Scaling |
|-------|--------------|----------------------|---------------|
| 1 turn | 0.07 ms | 56.5 ms | 1× |
| 100 turns | 0.37 ms | 58.7 ms | **~5× Slower** |

![Latency scaling with conversation turns](/images/posts/post-preprocessing-2.jpg)
*Jinja templating latency scales with message count (structure), not token count (content)*

**The Insight:** Tokenization cost scales with *content* (tokens). However, templating cost scales with *structure* (messages). As your application grows more complex—more turns, more RAG chunks, more tool calls—the Python-based templating layer compounds linearly while the Rust tokenizer stays flat.

---

## Experiment 3: The Production Scenario

The real bottleneck reveals itself in production. Inference servers process batches (e.g., 64, 128, or 256 requests) to saturate the GPU's massive memory bandwidth.

In [Experiment 3](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp3_concurrency.py), I simulated a high-throughput server processing **100,000 requests** in batches of 64. I measured **GPU Wait Time**—the wall-clock time the GPU sits idle while the CPU prepares a batch—across different thread counts.

| Threads | GPU Wait Time (Total) | Tokenization | Jinja |
|---------|----------------------|--------------|-------|
| 1 thread | 456 ms | 409 ms | 18.6 ms |
| 4 threads | 298 ms | 248 ms | 19.1 ms |
| 16 threads | 178 ms | 126 ms | 20.8 ms |
| 64 threads | 134 ms | 83 ms | 22.3 ms |

![GPU wait time vs thread count](/images/posts/post-preprocessing-3.jpg)

![Preprocessing component breakdown by thread count](/images/posts/post-preprocessing-4.jpg)
*At 64 threads, Jinja templating accounts for 16.6% of total GPU wait time—and is still growing*

**The Bottleneck Shift:** As we increased concurrency, the Rust-based tokenization scaled beautifully — **~5× speedup** (409ms → 83ms) by utilizing available cores.

However, the **Jinja templating component actually got slower** (18ms → 22ms).

At 64 threads, the "trivial" Jinja templating accounted for **16.6%** of the total time the GPU spent waiting. In a system processing thousands of requests per second, this 22ms per batch is a hard ceiling on throughput that no amount of GPU optimization can fix.

---

## Experiment 4: Why Threading Doesn't Help — The GIL

In [Experiment 4](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp4_threading.py), I measured the **Speedup Factor** (vs. ideal linear scaling) of each component as thread counts increased.

| Component | Speedup at 32 Threads | Why |
|-----------|----------------------|-----|
| Tokenization (Rust) | **5.72×** | Releases the GIL — true parallelism |
| Templating (Jinja) | **0.84×** | GIL-bound — actually gets *slower* |

![Speedup factor vs thread count for tokenization and templating](/images/posts/post-preprocessing-5.jpg)
*Tokenization scales near-linearly with threads. Templating degrades — threads contend for the same GIL lock.*

**The Diagnosis:** This is the **Global Interpreter Lock (GIL)** in action. Because Jinja2 is a pure Python library, only *one* thread can execute templating logic at a time. When we launch 64 threads, they don't run in parallel — they fight for the same lock, adding context-switching overhead on top.

{{< callout type="warning" title="The GIL Tax" >}}
In a naive Python-based serving setup, your 12-core CPU is effectively a **single-core machine** for the templating phase. Every thread you add beyond the first increases contention without increasing throughput.
{{< /callout >}}

---

## Conclusion: A Systems Approach to Inference

Our benchmarks quantify a critical lesson for AI Systems: **We cannot optimize inference by only looking at the GPU. CPU Preprocessing is not "free," and your code's concurrency has limits.**

### Key Takeaways

**1. Don't ignore the CPU.** If your GPU utilization is hovering at 60–70%, don't assume it is a model architecture issue. Check your preprocessing latency. The GPU might simply be starving for tensors.

**2. All pipeline components matter.** Optimizing tokenizers via Rust was a massive win for the community, but as we moved to chat models, Jinja templating has emerged as the new bottleneck.

**3. Ensure full CPU utilization.** Having 12 cores is useless if your code is GIL-bound to one. Your serving infrastructure must use patterns (multiprocessing, C++ backends) that can actually saturate those cores.

---

## What's Next?

Identifying the bottleneck is step one. In the next post, we explore **Production Architectures** that solve this:

- **vLLM's Async Engine** — How it manages the event loop to minimize blocking
- **SGLang** — How "Radix Attention" caches preprocessing work to skip it entirely for shared history
- **The Sidecar Pattern** — Decoupling CPU heavy-lifting from the GPU worker entirely

---

**Code:** The full benchmarking suite is available on [GitHub](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks).
