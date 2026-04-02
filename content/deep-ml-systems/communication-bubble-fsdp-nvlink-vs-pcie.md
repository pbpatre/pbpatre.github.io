---
title: "The Communication Bubble: FSDP on NVLink vs. PCIe"
date: 2026-04-02
draft: false
slug: "communication-bubble-fsdp-nvlink-vs-pcie"
author: "Pratik Patre"
description: "FSDP turns every forward-pass layer into a network call. On a slow interconnect, AllGather degrades 20× — yet the step time grows by only 29 ms. The difference between those two numbers is the communication bubble, and it's controlled entirely by the wire."
summary: "FSDP turns every forward-pass layer into a network call. On a slow interconnect, AllGather degrades 20× — yet the step time grows by only 29 ms. This post measures how much of that network time PyTorch can hide behind compute, and what happens when it can't."
tags: ["FSDP", "NCCL", "NVLink", "PCIe", "Distributed Training", "LLM", "GPU", "Communication Bubble", "AllGather", "ReduceScatter", "PyTorch Profiler", "H200"]
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

FSDP is the standard technique for training models that don't fit on a single GPU — it [shards model state across devices](/deep-ml-systems/vram-anatomy-why-llm-training-ooms/) so that each GPU only holds a fraction of the weights, optimizer states, and gradients. The memory savings are real. But FSDP introduces a dependency that doesn't exist in single-GPU training: because no device holds the complete model, every layer in every forward pass must issue an `AllGather` over the network to reconstruct its weights before computation can begin. Going distributed is not free — every layer now starts with a network call.

The natural question is: how much does the interconnect actually matter? If the network is fast enough, PyTorch overlaps each `AllGather` with the previous layer's computation and the GPU never notices the transfer. If the network is too slow, the GPU sits idle waiting for bytes that haven't arrived. This idle gap — the **communication bubble** — is the hidden variable that determines whether FSDP's memory-for-bandwidth trade is cheap or expensive. I wanted to measure it directly: same model, same code, same hardware, but with a fast interconnect and a slow one, to see exactly what controls the bubble and how to diagnose it. The `AllGather` degradation alone turned out to be 20× — and that number is only the beginning of the story.

I ran both scenarios on 2× H200 GPUs using `torch.profiler` with CUDA stream-level resolution: a NVLink baseline at ~900 GB/s, and a PCIe bottleneck simulated by disabling NCCL's peer-to-peer transfers on the same hardware. The full profiling code is [`profile_distributed.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/distributed/profile_distributed.py); the experiment runner is [`run_profiling_experiments.sh`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/distributed/run_profiling_experiments.sh).

{{< callout type="info" title="Benchmark Environment" >}}
**GPU:** 2× NVIDIA H200 (141 GB VRAM each, NVLink ~900 GB/s) · **PCIe (simulated):** ~32–64 GB/s via `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` · **Model:** GPT-XXL (3.1B parameters, 60 layers, 32 heads) · **Batch size:** 4, **Seq len:** 512 · **Stack:** PyTorch 2.10, CUDA 12.8 · **Profiler:** `torch.profiler` with CUDA activity, shapes, memory, and FLOP tracing
{{< /callout >}}

---

## FSDP's Hidden Cost: Every Layer Requires a Network Call

FSDP treats an entire cluster as one giant virtual GPU — no single device holds the complete model. The cost of this abstraction is that every layer execution begins with a network call. Before GPU 0 can run the forward pass for layer *N*, it must retrieve the weight shards it doesn't locally own from its peers, reconstruct the full parameter tensor, then compute. After the layer completes, the reconstructed weights are immediately discarded to reclaim the memory that FSDP was designed to save.

The forward pass becomes a sequence of compute-then-discard cycles, each gated on a preceding `AllGather`. The `ReduceScatter` during the backward pass is similarly gated — gradients must be sharded and distributed before the optimizer can step. NCCL (NVIDIA Collective Communications Library) executes three key primitives on every training step:

- **`AllGather`** — reconstructs each layer's sharded weights before forward pass computation
- **`ReduceScatter`** — shards and reduces gradients during the backward pass
- **`AllReduce`** — used for gradient synchronization in DDP; not used by FSDP (replaced by the above pair)

NCCL selects the most bandwidth-efficient algorithm for each collective based on the detected hardware topology — ring algorithms for PCIe, tree or direct-transfer algorithms for NVLink. The topology determines whether communication overlaps with compute or interrupts it.

---

## One Environment Variable Exposes the Bottleneck

H200s connected by NVLink are almost "too fast" for standard profiling — their ~900 GB/s interconnect hides nearly all communication latency behind compute, making the bubble invisible. To expose the bottleneck that multi-node clusters face on InfiniBand or Ethernet, I forced the same GPUs to communicate through the PCIe bus:

```bash
# Scenario A: NVLink baseline (~900 GB/s GPU-to-GPU direct)
torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp

# Scenario B: Simulated PCIe bottleneck (~32–64 GB/s via CPU host)
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
  torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp
```

Setting `NCCL_P2P_DISABLE=1` forces all data through the host CPU and PCIe lanes — the same path used in multi-node clusters where GPUs on different physical machines communicate over a network fabric. Same hardware pair, same model, different wire.

---

## 192 ms of Network, 29 ms of Stall

The profiler traces for the GPT-XXL (3.1B parameter) run reveal a striking disparity between what the network consumed and what the GPU actually waited for:

| Metric | NVLink Baseline | PCIe Bottleneck | Degradation (↑ = worse) |
|---|---|---|---|
| Avg step time (ms) | 99.33 | 128.24 | 1.29× ↑ |
| Avg compute time (ms) | 28.86 | 29.55 | ~1.0× (unchanged) |
| Avg NCCL time (ms) | 10.28 | 192.61 | **18.7×** ↑ |
| Est. GPU utilization (%) | 29.1% | 23.0% | — |
| Communication overhead (%) | 10.3% | **150.2%** | — |

Compute time is nearly identical in both scenarios — the model math doesn't change with the interconnect. But NCCL time explodes from 10 ms to 192 ms: an 18.7× increase that maps directly to the PCIe bandwidth penalty.

The puzzling number is the step time: it increased by only **28.90 ms**, despite NCCL consuming **192 ms** of wall-clock time. If the network took 192 ms, why wasn't the step delayed by 192 ms?

Before answering that, the per-operation breakdown shows where the 192 ms is spent:

| NCCL Operation | Per-step total — NVLink (ms) | Per-step total — PCIe (ms) | Degradation (↑ = worse) |
|---|---|---|---|
| `all_gather` (forward + backward weights) | 13.98 | 284.28 | **20.3×** ↑ |
| `reduce_scatter` (gradient sharding) | 6.58 | 100.94 | **15.3×** ↑ |
| `all_reduce` | 0.00 | 0.00 | — |

These are per-step cumulative totals — the sum of all calls of each operation type within a single training step, not per-call averages. `all_gather` runs once per layer per forward pass (60 calls on this 60-layer model), so its 13.98 ms NVLink total represents roughly 0.23 ms per call on average — consistent with the 169 µs single-call duration visible in the profiler trace. `reduce_scatter` runs once per layer during the backward pass at similar frequency. The raw totals reflect both the per-call cost and call count, so `all_gather` dominates the communication budget not just because each call is slower on PCIe, but because it fires most often. Together, these two primitives account for essentially all FSDP communication cost.

---

## Why 192 ms of Network Costs Only 29 ms of Wall Time

The 192 ms of NCCL time producing only a 28.90 ms step penalty is not an error. It is PyTorch FSDP's most important performance property: **overlapping collectives with compute**.

FSDP schedules `all_gather` operations on a dedicated, parallel CUDA stream — separate from the main compute stream where matrix multiplications execute. While the GPU's Tensor Cores are computing layer *N*'s forward pass, the DMA engines are simultaneously fetching layer *N+1*'s weights over the interconnect. If the network is fast enough to deliver the next layer's weights before the current layer's computation finishes, the GPU never stalls. The communication is completely hidden.

In the NVLink scenario, approximately **163 ms** of NCCL activity was hidden behind compute. The 28.90 ms "communication bubble" is precisely the portion of network latency that was too slow to hide — the amount by which communication time *exceeded* the computation window and forced the GPU to wait.

{{< callout type="warning" title="When 'Faster Compute' Makes Bubbles Larger" >}}
Making compute faster can *increase* the communication bubble, not decrease it. A 70B parameter model takes longer per layer than a 3B model, giving the `all_gather` more time to complete in the background. If you optimize your compute kernels (e.g., switching to Flash Attention), you shrink the window available for NCCL to hide its work — and more network time spills past the compute boundary into a stall. The bubble is determined by `max(0, NCCL_time - compute_time_per_layer)`.
{{< /callout >}}

---

## The Bubble Made Visible

The PyTorch Profiler trace makes the bubble physically visible. The trace records work across independent **CUDA streams** — parallel execution lanes the GPU uses to run different workloads concurrently:

- **Stream 7 (compute):** The heavy lifting — large `FullyShardedDataParallel.forward` blocks representing matrix multiplications (GEMMs).
- **Stream 21 (communication):** The "hidden" lane — `nccl:all_gather` and `nccl:reduce_scatter` operations scheduled here so they don't block the main compute stream.
- **X-axis:** Time. Vertical alignment between Stream 7 and Stream 21 blocks means those operations are running in parallel.

### NVLink: Communication as a Whisper

![NVLink profiler trace (torch.profiler): NCCL all_gather operations on Stream 21 appear as tiny pink slivers tucked beneath the large compute blocks on Stream 7. The GPU compute lane has no white space — weights for the next layer arrive before the current layer finishes. Duration of nccl:_all_gather_base (External ID 131): 169 µs.](/images/posts/nccl-fsdp-over-nvlink.png)
*NVLink baseline: the pink NCCL blocks on Stream 21 are barely visible slivers. Stream 7 compute is continuous — the GPU never stalls. `nccl:_all_gather_base` (External ID 131) completes in 169 µs.*

On NVLink, the communication blocks on Stream 21 are tiny, sparse slivers — they complete so quickly that they barely register on the timeline. Stream 7 stays continuously occupied. The moment one layer's compute ends, the next layer's weights have already arrived and computation resumes immediately. Communication is effectively invisible to the GPU.

### PCIe: Communication as a Wall

![PCIe bottleneck profiler trace (torch.profiler, NCCL_P2P_DISABLE=1): Stream 21 is dominated by a massive pink block. Stream 7 shows a large white-space gap — the communication bubble — before the FullyShardedDataParallel.forward block can begin. Flow events (thin lines) connect the end of the all_gather to the start of compute, marking the exact dependency. Duration of nccl:_all_gather_base (External ID 131): 10,076 µs — 60× slower than NVLink.](/images/posts/nccl-fsdp-over-pcie.png)
*PCIe bottleneck: Stream 21 is a solid pink wall. The white space on Stream 7 is the communication bubble — GPU Tensor Cores idle, waiting for weights. `nccl:_all_gather_base` (External ID 131) takes 10,076 µs — 60× the NVLink duration.*

On PCIe, Stream 21 becomes a solid, dense block. More importantly, look at Stream 7: before the `FullyShardedDataParallel.forward` block begins, there is a wide gap of white space. The Tensor Cores are powered on and ready. They are doing nothing — waiting for the `all_gather` on Stream 21 to complete before they have permission to start computing. The thin flow-event lines connecting the end of the pink block to the start of the compute block are the exact dependency that serializes the two operations.

Both screenshots capture `nccl:_all_gather_base` with **External ID 131** — the same logical operation at the same position in the model's forward pass. The hardware is identical. The code is identical. The difference is 169 µs versus 10,076 µs, caused entirely by the interconnect.

| Feature | NVLink (Optimized) | PCIe (Bottlenecked) |
|---|---|---|
| `nccl:_all_gather_base` duration | 169 µs | 10,076 µs — **60× slower** ↑ |
| Stream 21 appearance | Sparse slivers | Solid, dense block |
| Stream 7 status | Continuously occupied | Large idle gaps (the bubble) |
| Communication / compute relationship | Parallel (overlapped) | Serial (dependent) |
| External ID | 131 | 131 — same operation, same position |

---

## The Physics of the Bubble Won't Negotiate

Several observations from this experiment generalize beyond these specific GPUs and model sizes.

When communication overhead exceeds 100% — as it does in the PCIe scenario — the GPUs are spending more time waiting than computing. No amount of code optimization can fix a physical bandwidth deficit. The only remedies are a faster interconnect (NVLink, InfiniBand HDR/NDR), reducing the frequency of collectives (larger layers, gradient accumulation), or reducing the size of each collective (smaller shards, fewer parameters per `AllGather`). The interconnect is a hard ceiling on FSDP throughput.

Counterintuitively, larger models often achieve *better* communication overlap than small models on the same hardware. A 70B parameter model takes longer to compute per layer than a 3B model, giving NCCL more time to prefetch the next layer's weights in the background — the computation window is simply wider. Small models on slow interconnects are the worst case: short compute windows leave almost no time for the `AllGather` to hide.

Standard monitoring will not surface this problem. In both scenarios, `nvidia-smi` reports high GPU utilization — the GPU is executing CUDA kernels throughout. But in the PCIe scenario, a substantial fraction of that "utilization" is the GPU executing a blocking wait on the communication stream. The profiler trace — not `nvidia-smi` — is the diagnostic tool for communication bubbles.

The quantity that matters is not total NCCL time, but the ratio of *hidden* to *total* NCCL time — overlap efficiency. The profiler gives us a clean way to read this directly from the step numbers: on PCIe, NCCL consumed **192.61 ms** per step but the step itself only grew by **28.91 ms** (128.24 − 99.33). That means PyTorch successfully hid roughly **85%** of total NCCL time behind compute even under PCIe constraints — the remaining 15%, or ~28.91 ms, materialized as real GPU stall. On NVLink, where NCCL consumed just **10.28 ms** per step and the step time is dominated by compute, the overlap efficiency approaches 100%. As interconnect bandwidth drops toward Ethernet speeds in large multi-node clusters, even that 85% PCIe figure collapses further — the compute window stays the same width, but the NCCL envelope keeps growing.

The communication bubble is not a bug to be fixed — it is a structural property of sharded distributed training. The goal is to understand its magnitude, measure it directly in production traces, and make hardware and architecture decisions that keep it within the compute window.

The next post in this series will move from two GPUs to many — examining how the communication bubble scales across nodes and what happens when `AllGather` and `ReduceScatter` must traverse InfiniBand and Ethernet fabrics where bandwidth is shared, not dedicated.
