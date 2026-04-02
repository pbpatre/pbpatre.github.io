---
title: "The Preprocessing Barrier in LLM Inference"
date: 2024-01-15
draft: false
slug: "preprocessing-barrier-in-llms"
author: "Pratik Patre"
description: "A Jinja2 template loop takes 7% of single-request latency. At 32 concurrent threads, it becomes the throughput ceiling — Python's GIL serializes the one stage nobody thought to benchmark."
summary: "A Jinja2 template loop takes 7% of single-request latency. At 32 concurrent threads, it becomes the throughput ceiling. This post traces how Python's GIL converts an innocent-looking preprocessing step into a hard scaling wall."
tags: ["LLM", "Inference", "CPU", "Preprocessing", "GIL", "Performance", "Benchmarks", "vLLM"]
categories: ["Deep ML Systems"]
series: ["The Preprocessing Barrier"]
series_weight: 1
cover:
  image: ""
  hidden: true
  relative: false
ShowToc: true
TocOpen: true
ShowReadingTime: true
ShowWordCount: true
ShowShareButtons: true
ShowPostNavLinks: true
---

Picture this: your H100s are sitting at 65% utilization under peak load. The model is fine. The network is fine. You add more threads — utilization barely moves. You dig into the profiler and find the culprit: a Jinja2 template loop. Python code so unassuming that nobody had thought to benchmark it.

This is not a hypothetical. It is a failure mode I set out to reproduce and quantify. Through four controlled benchmarks, I profile the preprocessing pipeline that every LLM inference server runs — templating, tokenization, and collation — and trace exactly how Python's GIL converts what looks like a 7% overhead in single-request profiling into a hard throughput ceiling at production concurrency. The numbers are worth understanding before you hit them in production.

All benchmarks were run on a 12-core Apple M3 Pro. The full benchmarking suite is on [GitHub](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks).

{{< callout type="info" title="Benchmark Environment" >}}
**CPU:** Apple M3 Pro, 12 cores · **Model:** Llama-3.1-8B-Instruct · **Tokenizer:** HuggingFace Tokenizers (Rust) · **Templating:** Jinja2 · **Framework:** PyTorch 2.x
{{< /callout >}}

---

## The Three-Stage Pipeline

Before a single float reaches the GPU, raw text must pass through three CPU-bound transformations. Each stage has a different runtime, a different scaling characteristic, and — crucially — a different relationship with Python's GIL.

![The three-stage LLM preprocessing pipeline](/images/posts/post-preprocessing-1.jpg)
*Templating → Tokenization → Collation: three stages, three runtimes, three bottleneck profiles*

### Stage 1: Templating (Python — GIL-bound)

The server receives a list of JSON messages and compiles them into a single formatted string using a model-specific chat template (e.g., ChatML). The dominant implementation is **Jinja2** — pure Python, fully GIL-bound.

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

### Stage 2: Tokenization (Rust — GIL-free)

The templated string is encoded into integer token IDs. The standard implementations — **HuggingFace Tokenizers** and **tiktoken** — are Rust-backed and release the GIL during execution, enabling true multi-threaded parallelism.

**Input:** The formatted string above

**Output:**
```python
[128000, 128006, 9125, 128007, 271, 2675, 527, 264,
 11190, 18328, 13, 128009, 128006, 882, 128007, 271,
 3923, 374, 220, 10, 16, 30, 128009, 128006, 78191,
 128007, 271]
```

### Stage 3: Collation (Python/C++ — partially GIL-bound)

Token ID lists are converted into padded PyTorch tensors and pinned for GPU transfer. This stage involves both Python orchestration and C++ tensor operations.

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

## Baseline: A Single Request Looks Fast

How expensive is this pipeline for a single request? To establish a baseline, I profiled one Llama-3.1-8B-Instruct request at ~1,000 tokens ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp1_baseline.py)).

| Stage | Latency | Share |
|-------|---------|-------|
| Tokenization (Rust) | 0.82 ms | 68% |
| Collation (PyTorch) | 0.30 ms | 25% |
| Templating (Jinja)  | 0.08 ms |  7% |
| **Total**           | **~1.2 ms** | **100%** |

At 1.2 ms total, it's easy to dismiss preprocessing as negligible — and most teams do. The Rust-based tokenizer handles the bulk of the work efficiently, and Jinja's 0.08 ms contribution barely registers.

This is precisely the reasoning that breaks down at scale.

---

## Structure Matters More Than Content

Real-world LLM usage is rarely single-turn. RAG pipelines, multi-turn agent loops, and tool-calling workflows generate requests with dozens or hundreds of message turns.

The question: does the *structure* of a request (number of messages) affect preprocessing cost differently than the *content* (number of tokens)? To test this, I held the total token count constant at ~100k tokens and varied the number of conversation turns ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp2_scaling.py)).

| Turns | Jinja Latency | Tokenization Latency | Jinja Scaling |
|-------|--------------|----------------------|---------------|
| 1 turn | 0.07 ms | 56.5 ms | 1× |
| 100 turns | 0.37 ms | 58.7 ms | **~5× Slower** |

![Latency scaling with conversation turns](/images/posts/post-preprocessing-2.jpg)
*Jinja templating latency scales with message count (structure), not token count (content)*

Tokenization cost scales with *content* — the number of tokens to encode. Templating cost scales with *structure* — the number of messages to iterate over in the Jinja template loop. As applications grow more complex (more turns, more RAG chunks, more tool calls), the Python-based templating layer compounds linearly while the Rust tokenizer stays flat.

This asymmetry is invisible in single-request benchmarks.

---

## At Production Concurrency, the GPU Starves

The real bottleneck emerges when the inference server processes batches. Production systems batch 64, 128, or 256 requests per forward pass to saturate GPU memory bandwidth. During batch preparation, the GPU waits.

I simulated a high-throughput server processing 100,000 requests in batches of 64, and measured the **per-batch GPU wait time** — the wall-clock time the GPU sits idle while the CPU assembles each batch — across different thread counts ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp3_concurrency.py)).

| Threads | GPU Wait Time (per batch) | Tokenization | Jinja |
|---------|--------------------------|--------------|-------|
| 1 thread | 456 ms | 409 ms | 18.6 ms |
| 4 threads | 298 ms | 248 ms | 19.1 ms |
| 16 threads | 178 ms | 126 ms | 20.8 ms |
| 64 threads | 134 ms | 83 ms | 22.3 ms |

![GPU wait time vs thread count](/images/posts/post-preprocessing-3.jpg)

![Preprocessing component breakdown by thread count](/images/posts/post-preprocessing-4.jpg)
*At 64 threads, Jinja templating accounts for 16.6% of total GPU wait time — and is still growing*

The Rust-based tokenization scales as expected — **~5× speedup** (409 ms → 83 ms) by distributing work across cores.

Jinja templating, however, **gets slower** (18 ms → 22 ms).

At 64 threads, the "trivial" 7% from our baseline now accounts for **16.6%** of total GPU wait time. In a system processing thousands of requests per second, this 22 ms per batch becomes a hard ceiling that no amount of GPU optimization can overcome.

---

## The Root Cause: Python's GIL

Why does adding threads make Jinja *slower*? To isolate the mechanism, I measured the speedup factor of each component against ideal linear scaling as thread counts increased ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp4_threading.py)).

| Component | Speedup at 32 Threads | Mechanism |
|-----------|----------------------|-----------|
| Tokenization (Rust) | **5.72×** | Releases the GIL — true parallelism |
| Templating (Jinja) | **0.84×** | GIL-bound — actually gets *slower* |

![Speedup factor vs thread count for tokenization and templating](/images/posts/post-preprocessing-5.jpg)
*Tokenization scales near-linearly with threads. Templating degrades — threads contend for the same GIL lock.*

Jinja2 is a pure Python library. Under CPython's GIL, only one thread can execute Python bytecode at a time. When 64 threads attempt concurrent Jinja rendering, they don't run in parallel — they serialize on the GIL and pay additional context-switching overhead. More threads, less throughput.

{{< callout type="warning" title="The GIL Tax" >}}
Under the GIL, a 12-core CPU is effectively a single-core machine for the templating phase. Every thread added beyond the first increases contention without increasing throughput.
{{< /callout >}}

---

## Implications

Single-request profiling is a trap. The numbers look fine — 1.2 ms total, 0.08 ms for templating — until you model what happens at batch concurrency. Three things become clear:

**CPU preprocessing is not free.** When GPU utilization plateaus at 60–70%, the instinct is to look at model architecture, batch sizes, or memory layout. But the GPU may simply be starving for tensors — blocked on CPU-side preprocessing that hasn't been profiled.

**The community solved half the problem.** Rewriting tokenizers in Rust was a massive win — it moved the heaviest preprocessing stage out of the GIL. But the shift to chat-based models introduced Jinja templating as a new serial bottleneck, and that work hasn't received the same attention.

**Concurrency patterns matter as much as algorithms.** The difference between a GIL-releasing Rust function and a GIL-bound Python function is invisible at low concurrency and catastrophic at high concurrency. Serving infrastructure must use patterns (multiprocessing, C++ backends, or dedicated preprocessing workers) that can actually saturate available cores.

---

## What's Next

Identifying the bottleneck is step one. In the next post, we push vLLM and SGLang to saturation to see what happens when this preprocessing bottleneck meets real GPU serving systems:

- **The Utilization Illusion** — Why `nvidia-smi` shows 95% utilization while throughput stays flat
- **The Radix Paradox** — Why SGLang's KV cache optimization cannot reach the GPU at high concurrency
- **Decoupled Preprocessing** — What happens when you move Jinja and tokenization out of the inference server entirely

---

**Code:** The full benchmarking suite — all four experiments, data collection, and plotting scripts — is available on [GitHub](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks).
