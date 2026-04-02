---
title: "The Preprocessing Barrier (Part 2): How Preprocessing Starves Your GPUs"
date: 2024-02-12
draft: false
slug: "preprocessing-barrier-part-2"
author: "Pratik Patre"
description: "nvidia-smi reads 95% GPU utilization while throughput stays flat and latency degrades 18×. Decoupling preprocessing from the inference loop recovered 40% of GPU capacity the monolithic architecture was leaving on the table."
summary: "`nvidia-smi` reads 95% GPU utilization while throughput stays flat and latency degrades 18×. This post investigates the Utilization Illusion — and shows how decoupling preprocessing recovered 40% of GPU capacity the monolithic architecture left on the table."
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

[Part 1](/deep-ml-systems/preprocessing-barrier-in-llms/) established that Python's GIL turns what looks like a 7% preprocessing overhead into a hard throughput ceiling at production concurrency. The natural counter-argument: GPU inference time dwarfs CPU preprocessing time, so the bottleneck should not matter in practice.

I tested that argument directly. Pushing vLLM and SGLang to saturation under realistic RAG workloads surfaced something more troubling than slowness: **the metrics lie.** `nvidia-smi` reports 95% GPU utilization while throughput stays flat and latency degrades 18×. I call this the Utilization Illusion — and it is a direct consequence of coupled Python preprocessing.

This post traces the illusion through three frameworks, then breaks it.

{{< callout type="info" title="Benchmark Environment" >}}
**GPU:** NVIDIA L40S · **Frameworks:** vLLM, SGLang · **Model:** Qwen2.5-0.5B-Instruct · **Workload:** 100-turn chat history (~2,500 tokens) · **Concurrency:** 20 users (interactive) vs. 400 users (saturation) · All experiment code on [GitHub](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks/experiments)
{{< /callout >}}

---

## How vLLM and SGLang Handle Preprocessing

Both frameworks adopt a just-in-time preprocessing model that places CPU-heavy work on the scheduler's critical path. Understanding why requires tracing the request lifecycle in each.

**vLLM** orchestrates requests via a centralized Python event loop. Before the scheduler can allocate memory pages (PagedAttention), it needs the exact token count — which forces it to run the tokenizer synchronously. In high-throughput scenarios, the main thread blocks on string processing, preventing it from effectively batching for the GPU.

**SGLang** introduces Radix Attention to optimize multi-turn and RAG workloads. By modeling the KV cache as a Radix Tree, it can reuse attention computation from shared prefixes, theoretically reducing GPU work for repeated system prompts to near zero. The catch: to traverse the Radix Tree and find a cache hit, the system first needs the *token sequence*. Which means it must tokenize the raw prompt before it can check the cache. The GPU optimization is real; the preprocessing barrier upstream of it is completely intact.

This architectural constraint sets up the experiments that follow.

---

## The Utilization Illusion: vLLM Under Saturation

The first experiment targeted vLLM's scheduler under realistic load. Each request carried 100 turns of chat history — representative of a RAG-augmented agent loop — and I measured what happened when concurrent users scaled from 20 to 400 ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp5_vllm_latency_wall.py)).

| Scenario | Concurrent Users | Throughput (req/s) | GPU Utilization | P50 Latency |
|----------|-----------------|-------------------|----------------|-------------|
| Interactive | 20 | 133.75 | 92.61% | 150 ms |
| Saturation | 400 | 133.99 | 95.42% | 2,745 ms |

A 20× increase in concurrent load. Throughput: flat. Latency: 18× worse. GPU utilization: went *up*.

This is the Utilization Illusion. The Python scheduler was so overwhelmed by Jinja2 templating and tokenization that it could only feed the GPU at ~134 req/s regardless of load. The GPU processed those requests quickly, then waited for the CPU to assemble the next batch — appearing busy while repeatedly idling. What `nvidia-smi` captures is duty cycle, not efficiency. The gap between duty cycle and **Model FLOPS Utilization (MFU)** is where the capacity disappeared.

---

## The Radix Paradox: When GPU Optimization Cannot Reach the GPU

The next experiment tested whether SGLang's Radix Attention — a genuine GPU-side innovation — could break through the bottleneck. I compared two traffic patterns ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp6_sglang_radix_latency.py)):

- **Cache miss:** Every request has a unique prompt history. Attention computed from scratch.
- **Cache hit:** All requests share a long common prefix (simulating a shared RAG system prompt). The Radix Tree reuses the KV cache — GPU work for that prefix drops to near zero.

| Load | Concurrent Users | Cache State | Throughput (req/s) | P50 Latency |
|------|-----------------|-------------|-------------------|-------------|
| Low | 20 | Miss | 115.10 | 164 ms |
| Low | 20 | Hit | 245.84 | 67 ms |
| High | 400 | Miss | 219.81 | 1,835 ms |
| High | 400 | Hit | 227.52 | 1,780 ms |

At low concurrency, Radix Attention works as advertised — a cache hit cuts latency by 60% (164 ms → 67 ms) and nearly doubles throughput. The CPU is not yet saturated, so the GPU savings propagate to the user.

At high concurrency, the benefit collapses. The difference between full GPU computation and near-zero GPU computation is 55 ms — less than 3% of total latency. A request with a 100% cache hit still spent **1.78 seconds** waiting in the Python queue to be tokenized.

The mechanism is clear: SGLang cannot consult the Radix Tree until it has the token sequence. Tokenization happens on the CPU. At saturation, the upstream CPU queue completely masks the downstream GPU optimization. **A 0 ms GPU computation cannot help if the CPU takes 2 seconds to prepare the input.**

---

## Breaking the Coupling: Decoupled Preprocessing

The pattern that breaks this bottleneck is architectural: move Jinja2 templating and tokenization out of the inference server and into a separate preprocessing gateway that delivers pre-tokenized data to the GPU worker.

This is the approach natively supported by [NVIDIA Triton Inference Server](https://developer.nvidia.com/dynamo-triton) via ensemble pipelines. To isolate the effect of decoupling alone, I built a simulated sidecar using two Rust-backed libraries that bypass CPython entirely for the heavy work ([experiment code](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp7_sidecar_latency.py)):

- **[minijinja](https://github.com/mitsuhiko/minijinja)** — A Rust implementation of Jinja2, called from Python. Replaces CPython's GIL-bound Jinja2 template rendering.
- **HuggingFace Tokenizers** — The same Rust-backed tokenizer from Part 1, which releases the GIL during execution.

The sidecar process applies the chat template via `minijinja`, tokenizes the result, and sends the raw `input_ids` directly to SGLang's `/generate` endpoint — completely bypassing the `/v1/chat/completions` endpoint that would trigger server-side Jinja2 + tokenization.

```python
# Sidecar: Rust-backed preprocessing (GIL-free)
def sidecar_process(messages):
    # 1. Template with minijinja (Rust) instead of Jinja2 (Python)
    prompt_str = env.render_template("chat",
        system=messages[0]["content"],
        history=messages[1:]
    )
    # 2. Tokenize with HF Tokenizers (Rust, releases GIL)
    input_ids = tokenizer.encode(prompt_str)
    return input_ids

# Monolith: raw messages → /v1/chat/completions (server-side Jinja + tokenizer)
# Sidecar:  input_ids   → /generate           (GPU only, no preprocessing)
```

### Phase 1: The Base Tax

First, I measured the raw per-request savings at moderate concurrency (50 users) with long contexts (~2,500 tokens) — before queuing effects compound the cost.

| Architecture | Concurrent Users | Throughput (req/s) | P50 Latency |
|-------------|-----------------|-------------------|-------------|
| Monolith | 50 | 118.44 | 379 ms |
| Decoupled (sidecar) | 50 | 178.58 | 212 ms |

Offloading preprocessing saves **~167 ms per request** at moderate load — before any queuing dynamics. This is the base tax that every request pays in a monolithic setup.

### Phase 2: The Multiplier Effect

At saturation (400 users), that 167 ms base saving has a nonlinear effect on total latency through Kingman's formula: reducing service time at a bottleneck reduces queue depth super-linearly.

| Architecture | Concurrent Users | Throughput (req/s) | P50 Latency | Speedup |
|-------------|-----------------|-------------------|-------------|--------|
| Monolith | 400 | 178.58 | 1,997 ms | — |
| Decoupled (sidecar) | 400 | 302.60 | 1,017 ms | **1.96×** |

![Latency and throughput comparison across monolith and decoupled architectures at low and high concurrency](/images/posts/post-preprocessing-p2-1.jpg)
*Cross-experiment comparison: decoupled preprocessing vs monolith at 50 and 400 concurrent users*

Three results stand out:

**Throughput increased 70%.** The GPU was finally receiving a steady stream of tokens rather than a drip feed assembled by an overwhelmed Python scheduler. Throughput jumped from ~178 req/s to 302 req/s. The monolith was leaving **40% of the GPU's capacity on the table**.

**Latency dropped by ~1 second at P50.** The 167 ms base tax, compounded through queuing at 400 concurrent users, was responsible for nearly a full second of wait time. This is Kingman's formula in action — small reductions in service time at a saturated bottleneck produce outsized reductions in queue length.

**A residual floor remains at ~1,017 ms.** Even without tokenization, deserializing 400 large JSON payloads and managing TCP connections saturates the Uvicorn event loop. This ~1 second floor is the cost of Python's HTTP stack itself — `asyncio`, Pydantic validation, and ASGI overhead. Breaking below it requires moving the entire networking layer to Rust or C++, which is a separate optimization.

---

## Three Principles from Saturation Testing

Single-request profiling hides all three of these. They only surface when you push the system to saturation.

**Utilization metrics require interpretation.** `nvidia-smi` reports duty cycle, not efficiency. A GPU that processes small batches and idles between them will show high utilization while delivering a fraction of its theoretical MFU. The right metric is throughput per dollar, not utilization percentage.

**GPU-side optimizations have a CPU-side prerequisite.** Radix Attention, speculative decoding, and paged attention are real wins — but they operate downstream of preprocessing. At high concurrency, the CPU queue absorbs the gains before they reach the user. Optimizing the GPU without first clearing the CPU bottleneck compounds latency without improving throughput.

**Decoupling is an architectural decision, not a tuning knob.** You cannot thread your way out of the GIL ([Part 1](/deep-ml-systems/preprocessing-barrier-in-llms/) showed this), and you cannot cache your way past a saturated tokenizer (this post showed this). The fix requires structural separation: preprocessing as a dedicated layer — ideally Rust-backed — that delivers token IDs, and an inference server that consumes them.

---

## What's Next

The decoupled architecture removes the Python preprocessing bottleneck, but introduces new questions: how do you scale the preprocessing tier independently? What happens to latency when the gateway is remote? How does this interact with speculative decoding or prefix caching?

The next post will cover **production deployment patterns** for decoupled preprocessing — including the tradeoffs between in-process, sidecar, and remote gateway architectures.

---

**Code:** All three experiments from this post — vLLM saturation ([exp5](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp5_vllm_latency_wall.py)), SGLang Radix ([exp6](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp6_sglang_radix_latency.py)), and decoupled sidecar ([exp7](https://github.com/pbpatre/llm-system-benchmarks/blob/main/preprocessing/benchmarks/experiments/exp7_sidecar_latency.py)) — are in the [GitHub repository](https://github.com/pbpatre/llm-system-benchmarks/tree/main/preprocessing/benchmarks/experiments).
