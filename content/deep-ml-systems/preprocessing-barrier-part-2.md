---
title: "The Preprocessing Barrier (Part 2): How CPU Preprocessing Starves Your GPUs"
date: 2024-02-12
draft: false
slug: "preprocessing-barrier-part-2"
author: "Pratik Patre"
description: "High GPU utilization but flat throughput? Your inference server may be suffering from the Utilization Illusion — where the GPU looks busy but is actually idling between requests, starved by Python preprocessing."
summary: "We benchmark vLLM and SGLang under saturation load and expose the Utilization Illusion: high nvidia-smi numbers masking GPU starvation caused by Python preprocessing. Then we show how decoupled preprocessing recovers 40% of hidden GPU capacity."
tags: ["LLM", "Inference", "vLLM", "SGLang", "GPU", "Preprocessing", "Production", "Benchmarks", "Radix Attention", "Throughput"]
categories: ["Deep ML Systems"]
series: ["The Preprocessing Barrier"]
series_weight: 2
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

[Part 1](https://pbpatre.github.io/deep-ml-systems/preprocessing-barrier-in-llms/) established a baseline: Python's GIL turns what looks like a 7% overhead in single-request profiling into a hard throughput ceiling at production concurrency. The obvious counter-argument is that GPU inference time dwarfs CPU preprocessing time, so the bottleneck does not matter in practice.

This post tests that argument directly. We pushed vLLM and SGLang to saturation under realistic RAG workloads and found something more troubling than slowness: the metrics lie. `nvidia-smi` reports 95% GPU utilization while throughput stays flat and latency degrades 18×. We call this the **Utilization Illusion** — and it is a direct consequence of coupled Python preprocessing.

We then show what happens when you break the coupling.

{{< callout type="info" title="Benchmark Environment" >}}
**GPU:** NVIDIA L40S · **Frameworks:** vLLM, SGLang · **Workload:** Complex chat history (100 turns, ~2,500 tokens) · **Concurrency:** 20 users (interactive) vs. 400 users (saturation) · Full experiment code on [GitHub](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks/experiments)
{{< /callout >}}

---

## How vLLM and SGLang Handle Preprocessing

Both frameworks adopt a just-in-time preprocessing model that places CPU-heavy work on the scheduler's critical path. Understanding why requires tracing the request lifecycle in each.

**vLLM** orchestrates requests via a centralized Python event loop. Before the scheduler can allocate memory pages (PagedAttention), it needs the exact token count — which forces it to run the tokenizer synchronously. In high-throughput scenarios, the main thread blocks on string processing, preventing it from effectively batching for the GPU.

**SGLang** introduces Radix Attention to optimize multi-turn and RAG workloads. By modeling the KV cache as a Radix Tree, it can reuse attention computation from shared prefixes, theoretically reducing GPU work for repeated system prompts to near zero. The catch: to traverse the Radix Tree and find a cache hit, the system first needs the *token sequence*. Which means it must tokenize the raw prompt before it can check the cache. The GPU optimization is real; the preprocessing barrier upstream of it is completely intact.

This architectural constraint sets up the experiments that follow.

---

## The Utilization Illusion: vLLM Under Saturation

We designed a stress test to expose the scheduler bottleneck in vLLM under realistic load ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp5_vllm_latency_wall.py)).

| Scenario | Concurrency | Throughput (req/s) | GPU Util | P50 Latency |
|----------|------------|-------------------|---------|-------------|
| Interactive | 20 | 133.75 | 92.61% | 150 ms |
| Saturation | 400 | 133.99 | 95.42% | 2,745 ms |

A 20× increase in concurrent load. Throughput: flat. Latency: 18× worse. GPU utilization: went *up*.

This is the Utilization Illusion. The Python scheduler was so overwhelmed by Jinja2 templating and tokenization that it could only feed the GPU at ~133 req/s regardless of load. The GPU processed those requests quickly, then waited for the CPU to assemble the next batch — appearing busy while repeatedly idling. What `nvidia-smi` captures is duty cycle, not efficiency. The gap between duty cycle and **Model FLOPS Utilization (MFU)** is where the capacity disappeared.

---

## The Radix Paradox: When GPU Optimization Cannot Reach the GPU

The SGLang experiment was designed to test whether Radix Attention's KV cache reuse could break the bottleneck. We compared two traffic patterns ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp6_sglang_radix_latency.py)):

- **Cache miss:** Every request has a unique prompt history. Attention computed from scratch.
- **Cache hit:** All requests share a long common prefix (simulating a shared RAG system prompt). The Radix Tree reuses the KV cache — GPU work for that prefix is reduced to near zero.

| Load | Concurrency | Request Type | Throughput (req/s) | P50 Latency |
|------|------------|-------------|-------------------|-------------|
| Low | 20 | Cache miss | 115.10 | 164 ms |
| Low | 20 | Cache hit | 245.84 | 67 ms |
| High | 400 | Cache miss | 219.81 | 1,835 ms |
| High | 400 | Cache hit | 227.52 | 1,780 ms |

At low concurrency, Radix Attention works as advertised — a cache hit cuts latency by 60% (164 ms → 67 ms) and nearly doubles throughput. The CPU is not yet saturated, so the GPU savings propagate to the user.

At high concurrency, the benefit collapses. The difference between full GPU computation and near-zero GPU computation is 55 ms — less than 3% of total latency. A request with a 100% cache hit still spent **1.78 seconds** waiting in the Python queue to be tokenized.

The mechanism is clear: SGLang cannot consult the Radix Tree until it has the token sequence. Tokenization happens on the CPU. At saturation, the upstream CPU queue completely masks the downstream GPU optimization. **A 0 ms GPU computation cannot save you if the CPU takes 2 seconds to prepare the data.**

---

## Breaking the Coupling: Decoupled Preprocessing

The pattern that breaks this bottleneck is architectural: move Jinja2 templating and tokenization out of the inference server and into a separate process — a preprocessing gateway that delivers pre-tokenized tensors to the GPU worker, which never touches raw text.

This is the approach natively supported by [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) via ensemble pipelines. We simulated it with a separate Python process (standing in for a Rust-based gateway) to isolate the effect of decoupling alone.

### Phase 1: The Base Tax

First, we measured the raw per-request savings at moderate concurrency (50 users) with long contexts (~2,500 tokens) — before queuing effects compound the cost ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp7_sidecar_latency.py)).

| Architecture | Concurrency | Throughput (req/s) | P50 Latency |
|-------------|------------|-------------------|-------------|
| Monolith | 50 | 118.44 | 379 ms |
| Decoupled | 50 | 178.58 | 212 ms |

Offloading preprocessing saves **~167 ms per request** at moderate load — before any queuing dynamics. This is the base tax that every request pays in a monolithic setup.

### Phase 2: The Multiplier Effect

At saturation (400 users), that 167 ms base saving has a nonlinear effect on total latency through Kingman's formula: reducing service time at a bottleneck reduces queue depth super-linearly.

| Architecture | Concurrency | Throughput (req/s) | P50 Latency | Speedup |
|-------------|------------|-------------------|-------------|---------|
| Monolith | 400 | 178.58 | 1,997 ms | — |
| Decoupled | 400 | 302.60 | 1,017 ms | **1.96×** |

![Latency and throughput comparison across monolith and decoupled architectures at low and high concurrency](/images/posts/post-preprocessing-p2-1.jpg)
*Cross-experiment comparison: decoupled preprocessing vs monolith at 50 and 400 concurrent users*

Three results stand out:

**Throughput increased 70%.** The GPU was finally receiving a steady stream of tensors rather than a drip feed assembled by an overwhelmed Python scheduler. Throughput jumped from ~178 req/s to 302 req/s. The monolith was leaving **40% of the GPU's capacity on the table**.

**Latency dropped by ~1 second.** The 167 ms base tax, compounded through queuing at 400 concurrent users, was responsible for nearly a full second of P50 wait time. This is Kingman's formula in action — small reductions in service time at a saturated bottleneck have outsized effects on queue length.

**A residual floor remains at ~1,017 ms.** This represents the cost of Python's HTTP handling. Even without tokenization, deserializing 400 large JSON payloads and managing TCP connections saturates the Uvicorn event loop. Removing this floor requires moving the networking layer to Rust or C++ — a separate optimization.

---

## What This Means for Production System Design

The experiments above surface three principles that do not show up in single-request benchmarks:

**Utilization metrics require interpretation.** `nvidia-smi` reports duty cycle, not efficiency. A GPU that processes small batches and idles between them will show high utilization while delivering a fraction of its theoretical MFU. The right metric is throughput per dollar, not utilization percentage.

**GPU-side optimizations have a CPU-side prerequisite.** Radix Attention, speculative decoding, and paged attention are all real wins — but they operate downstream of preprocessing. At high concurrency, the CPU queue absorbs the gains before they reach the user. Optimizing the GPU without first clearing the CPU bottleneck is adding lanes to a highway whose on-ramp is congested.

**Decoupling is an architectural decision, not a performance tuning knob.** You cannot thread your way out of the GIL (as Part 1 showed), and you cannot cache your way past a saturated tokenizer (as this post shows). The fix requires structural separation: preprocessing as a dedicated infrastructure layer that delivers tensors, and an inference server that consumes them.

---

## What's Next

The decoupled architecture removes the Python bottleneck, but introduces new questions: how do you scale the preprocessing tier independently? What happens to latency when the gateway is remote? How does this interact with speculative decoding or prefix caching?

The next post will cover **production deployment patterns** for decoupled preprocessing — including the tradeoffs between in-process, sidecar, and remote gateway architectures.

---

**Code:** All five experiments from this post — vLLM saturation, SGLang Radix paradox, and the three decoupled preprocessing benchmarks — are in the [GitHub repository](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks/experiments).
