---
title: "Prefill vs Decode: The GPU Scheduling Trade-off in LLM Inference"
date: 2024-03-20
draft: false
slug: "prefill-vs-decode-gpu-scheduling"
author: "Pratik Patre"
description: "Standard benchmarks use uniform request sizes, hiding the scheduling collision that defines real LLM traffic. I benchmark bimodal workloads on an L40S and expose the TTFT-vs-ITL trade-off that chunked prefill forces you to make."
summary: "Uniform benchmarks say TTFT is 17ms. Bimodal production traffic says 56ms. I reproduce the convoy effect, benchmark chunked prefill across three chunk sizes, and show why GPU scheduling is a trade-off between responsiveness and stream smoothness — not a problem with a solution."
tags: ["LLM", "Inference", "GPU", "Scheduling", "Prefill", "Decode", "Chunked Prefill", "vLLM", "Latency", "TTFT", "ITL"]
categories: ["Deep ML Systems"]
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

A user sends a 50-token question to your LLM API. The GPU is free. Prefill should take 2ms. Instead, the user waits 56ms — because a 4,000-token RAG request arrived one millisecond earlier and locked the GPU for a monolithic prefill pass.

**TPS is not user experience.** A leaderboard showing 4,000 tokens/second in aggregate tells you nothing about the 1% of users staring at a spinner. What users actually measure is two things: *responsiveness* — did the answer start immediately? — and *smoothness* — did the text flow, or did it freeze mid-sentence? Standard benchmarks use uniform request sizes, which means every request sees identical scheduling behaviour. P99 looks clean because variance is structurally zero. These are *happy path* benchmarks.

Real production traffic is bimodal: a mix of short chat turns (50 tokens, instant reply expected) and long RAG queries (4,000 tokens, high compute required). When both hit the same endpoint, the GPU scheduler must juggle two incompatible contracts simultaneously. That juggling act is invisible in uniform benchmarks — and it is what this post quantifies.

This post reproduces that collision on an NVIDIA L40S using a bimodal workload I designed specifically to stress-test GPU scheduling. The results led to a [vLLM contribution](https://github.com/vllm-project/vllm/pull/34146) — a `BimodalDataset` benchmark that makes this trade-off measurable in CI.

{{< callout type="info" title="Benchmark Environment" >}}
**GPU:** NVIDIA L40S (48GB GDDR6) · **Model:** Llama-2-7b · **Framework:** vLLM · **Workloads:** Uniform (100% short) vs. Bimodal (80% short chat + 20% long RAG) · **Code & PR:** [vllm-project/vllm#34146](https://github.com/vllm-project/vllm/pull/34146)
{{< /callout >}}

---

## Two Phases, Two Physics

Every LLM request passes through two phases with radically different computational profiles. Understanding the physics of each is prerequisite to understanding why scheduling is hard.

### Prefill: The Burst

The model reads the entire prompt at once to compute the initial Key-Value (KV) cache. This is a large matrix multiplication — $N_{\text{tokens}} \times D_{\text{model}}$ — that is **compute-bound**. The GPU's Streaming Multiprocessors (SMs) light up to near-100% utilisation. A 4,000-token prefill on an L40S takes ~40ms of uninterruptible compute.

### Decode: The Drip

The model generates tokens one at a time, each requiring a small vector-matrix multiply — $1 \times D_{\text{model}}$ against the model weights. This is **memory-bound**: the compute cores sit mostly idle, waiting for data from HBM. Each decode step takes ~5–10ms but must happen frequently to maintain a smooth token stream.

The collision is structural: prefill wants to lock the GPU for a long burst of compute. Decode needs frequent, short access to keep streams alive. They cannot both be satisfied simultaneously.

---

## The Convoy Effect: Short Requests Behind Long Ones

In a real system, requests are batched — the scheduler groups multiple decode steps into a single forward pass to amortise memory bandwidth. But when a large prefill arrives, it blocks the batch.

Consider two requests arriving nearly simultaneously:

- **Request A (chat):** 50-token prompt. Prefill takes ~2ms.
- **Request B (RAG):** 4,000-token prompt. Prefill takes ~40ms.

If B arrives first, A waits 40ms before its 2ms prefill can even start. This is **Head-of-Line (HOL) blocking** — the "convoy effect" from networking, replicated inside the GPU scheduler.

![Head-of-line blocking: short request stuck behind long prefill](/images/posts/post-prefill-decode-2.jpg)
*A 50-token chat request waits 40ms for a 4,000-token RAG prefill to complete — a scheduling tax invisible in uniform benchmarks*

This wait time is invisible in uniform benchmarks because all requests are the same size. There is no "short request stuck behind a long one" — the scenario cannot occur.

---

## Breaking the Benchmark: Bimodal Stress Test

To reproduce the convoy effect under controlled conditions, I ran two workload patterns through vLLM on an L40S.

**Uniform baseline:** 100% short requests (50–100 tokens). The "happy path."

**Bimodal stress test:** 80% short chat requests + 20% long RAG requests (2,000–4,000 tokens). This is closer to real production traffic.

### The Results

| Metric | Uniform | Bimodal | Impact |
|--------|---------|---------|--------|
| Short P50 TTFT | 13 ms | 15 ms | +11% |
| Short P99 TTFT | 18 ms | 56 ms | **+216%** |
| Throughput (tok/s) | 3,762 | 2,372 | -37% |

The P50 barely moves — most short requests still arrive when the GPU is free. But the P99 explodes: **56ms vs 18ms**. This is the 1-in-100 short request that landed behind a RAG prefill.

The gap between P50 and P99 in the bimodal row is the convoy effect, quantified:

- **P50 (15ms):** A short request that arrived when the GPU was between prefills. Processed instantly.
- **P99 (56ms):** A short request that arrived during a 4,000-token prefill. Waited ~40ms in the queue.

That 40ms gap is a scheduling tax that exists in every production system serving mixed traffic. Uniform benchmarks structurally cannot detect it.

This is why I contributed a [bimodal workload generator to vLLM](https://github.com/vllm-project/vllm/pull/34146) — `--dataset-name bimodal` generates configurable mixes of short and long requests, making the convoy effect reproducible in CI benchmarks:

```bash
# Bimodal workload benchmark (80% short chat + 20% long RAG)
vllm bench serve --dataset-name bimodal --num-prompts 500 \
    --model Qwen/Qwen2.5-0.5B --request-rate 30 \
    --save-result --save-detailed --result-filename bimodal.json
```

---

## Chunked Prefill: The Industry Fix and Its Hidden Cost

The standard solution to HOL blocking is **chunked prefill**: instead of processing a 4,000-token prompt in one atomic 40ms pass, the scheduler breaks it into chunks (e.g., 512 tokens each). Between chunks, it pauses and asks: *does anyone else need to run?*

```
Atomic:  [PREFILL 4000 ──────────────────] → [Short Request]
Chunked: [Chunk 512] → [Short] → [Chunk 512] → [Short] → ...
```

This eliminates gross HOL blocking. But my benchmarks revealed an inverse correlation that chunked prefill forces: **improving responsiveness (TTFT) breaks smoothness (ITL).**

### TTFT vs ITL: The Inverse Correlation

I ran the bimodal benchmark (70/30 split) across three chunk sizes on the L40S. The results from my [vLLM PR benchmarks](https://github.com/vllm-project/vllm/pull/34146) show the trade-off precisely:

| | Metric | Chunk 512 | Chunk 2048 | Chunk 8192 |
|--|--------|-----------|------------|------------|
| **TTFT** ↓ improves with larger chunks | Short P99 (ms) | 99.1 | 77.7 | 66.3 |
| | Short P99/Median ratio | 7.0× | 5.3× | 4.6× |
| **ITL** ↑ degrades with larger chunks | Short Max (ms) | 8.7 | 18.7 | 54.6 |
| | Short % >10ms spike | 0.0% | 26.3% | 26.6% |
| | Long % >10ms spike | 0.0% | 81.7% | 81.1% |

![TTFT vs ITL trade-off across chunk sizes](/images/posts/chunked-prefill-tradeoff.jpg)
*As chunk size increases, TTFT tail shrinks and ITL spikes grow — the inverse correlation is structural, not tunable away*

The numbers confirm it: **lower TTFT tail → worse ITL stutters.** There is no chunk size that wins on both.

### Why the Trade-off Is Structural

Consider three users competing for the GPU:

- **Blue:** An active user mid-stream, needing a decode token every ~20ms.
- **Red:** A massive RAG request arriving with a 4,000-token prefill (~40ms of compute).
- **Green:** A new chat user pressing Enter just after Red arrives.
- **|OH|:** Overhead per chunk — kernel launch latency + memory swap between chunks.

**Large chunks (8192) — "Fast Start, Frozen Stream":**

![Large chunk scheduling: fast TTFT, frozen stream for active users](/images/posts/new-request-large-chunk.jpg)
*Large chunks let Red's prefill clear quickly — Green starts fast, but Blue's stream freezes for the full 40ms prefill duration*

The scheduler runs Red's prefill in one solid block. Green gets a fast TTFT (the prefill clears quickly, no |OH| overhead), but **Blue freezes** — no decode tokens for ~40ms. The stream stutters visibly.

**Small chunks (512) — "Smooth Stream, Slow Start":**

![Small chunk scheduling: smooth stream, delayed start for new requests](/images/posts/new-request-small-chunk.jpg)
*Small chunks interleave Red's prefill with Blue's decode tokens — Blue sees a smooth stream, but Green waits longer as |OH| accumulates across every chunk boundary*

The scheduler slices Red's prefill into small pieces, yielding to Blue between each chunk. Blue sees a smooth 8ms stream. But every slice adds |OH| (kernel launch + memory swap), so Red's total prefill time stretches from 40ms to 50ms+. **Green waits longer** because the accumulated overhead delays Red's completion.

{{< callout type="warning" title="The Core Insight" >}}
Chunked prefill does not eliminate latency. It redistributes it. You can concentrate the pain into a single stutter (large chunks) or spread it as overhead that delays new arrivals (small chunks). There is no setting that avoids both.
{{< /callout >}}

---

## Choosing Your Compromise

There is no free lunch in GPU scheduling. The bimodal experiments prove that optimising for every user simultaneously is not possible. The right chunk size depends on what you are building.

### For Chatbots and Streaming UIs

**Optimise for:** Stream smoothness (ITL)

**Config:** Small chunks (512)

**Why:** Users tolerate a slightly slower start (100ms TTFT) far more than they tolerate a stuttering, freezing stream. A chatbot that starts 30ms late but flows smoothly feels faster than one that starts instantly and then freezes for 50ms mid-sentence. Zero requests experience a >10ms ITL spike at chunk 512.

### For APIs and Batch Processing

**Optimise for:** Throughput and responsiveness (TTFT)

**Config:** Large chunks (4096+)

**Why:** There is no streaming UX to protect. Large chunks let heavy prefills clear the GPU quickly, minimising the total time any request occupies a sequence slot. The P99/Median TTFT ratio drops from 7.0× to 4.6× — a measurably more predictable API.

| Objective | Chunk Size | P99 TTFT | Max ITL | Best For |
|-----------|-----------|----------|---------|----------|
| Smooth streaming | 512 | 99.1 ms (higher) | 8.7 ms ✅ | Chatbots, real-time UI |
| Balanced | 2048 | 77.7 ms | 18.7 ms | Mixed workloads |
| Fast start / throughput | 8192 | 66.3 ms ✅ | 54.6 ms (stutter) | APIs, batch processing |

---

## Making It Measurable: The vLLM Contribution

The reason this trade-off goes undetected is that standard benchmarks use uniform request distributions. To close that gap, I contributed a [BimodalDataset to vLLM](https://github.com/vllm-project/vllm/pull/34146) — a configurable mixed workload generator that ships directly in vLLM's benchmarking suite.

With it, reproducing the convoy effect is a single flag:

```bash
# 80% short chat + 20% long RAG (default)
vllm bench serve --dataset-name bimodal --num-prompts 500

# Tunable ratio and token ranges
vllm bench serve --dataset-name bimodal --bimodal-short-ratio 0.7 \
    --bimodal-long-input-min 2000 --bimodal-long-input-max 5000
```

This makes the TTFT-vs-ITL trade-off reproducible in CI. Chunked prefill configurations can now be tuned against workloads that actually resemble production traffic — not uniform benchmarks that structurally cannot expose the inverse correlation.

---

## Conclusion

Uniform benchmarks told us TTFT was 17ms. Bimodal production traffic revealed it was 56ms. And chunked prefill — the industry's fix for HOL blocking — does not solve the problem. It forces a choice: fast starts or smooth streams, but not both.

Optimal GPU scheduling is not about eliminating constraints. It is about choosing which constraint to violate — and making that choice deliberately, with data from workloads that actually resemble your production traffic.

---

**Code & PR:** The bimodal benchmark dataset is available as [vllm-project/vllm#34146](https://github.com/vllm-project/vllm/pull/34146). The full experiment code and results are linked in the PR description.
