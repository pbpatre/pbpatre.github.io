---
title: "GPU Starvation: Profiling the LLM Training Data Pipeline"
date: 2026-04-01
draft: false
slug: "gpu-starvation-llm-training-data-pipeline"
author: "Pratik Patre"
description: "nvidia-smi reports 100% GPU utilization while 97.7% of compute cycles are wasted on padding tokens. Sequence packing alone recovers a 42× throughput gain — but only after profiling reveals the bottleneck is almost never where engineers expect it."
summary: "`nvidia-smi` reports 100% GPU utilization while 97.7% of compute cycles are wasted on padding tokens. Sequence packing alone recovers a 42× throughput gain. This post profiles every stage of the five-stage data pipeline to find where the real bottlenecks hide."
tags: ["LLM", "Training", "GPU", "Data Pipeline", "PyTorch", "DataLoader", "PCIe", "Sequence Packing", "Padding", "Throughput", "Profiling", "NVMe"]
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

An NVIDIA L40S pushes 181 TFLOPS of FP16 compute. `nvidia-smi` will happily report 100% GPU utilization while a significant fraction of that compute does nothing useful — and it will never tell you why.

The problem is that GPU utilization is a lagging indicator. It measures whether the GPU is active, not whether the pipeline feeding it is healthy. Before a single CUDA kernel fires, training data must pass through five distinct stages: read from disk, preprocessed on the CPU, padded and collated into uniform tensors, transferred across the PCIe bus, and only then handed to the GPU. Any one of these stages can independently stall the most expensive component in the system. In practice, several stall it simultaneously — and standard monitoring surfaces none of it.

I built seven standalone benchmarks isolating each stage of this pipeline, ran them on real hardware with the OS page cache flushed before every run, and measured the penalty at each step. The full code is in [`training/analysis/`](https://github.com/pbpatre/llm-system-benchmarks/tree/main/training/analysis). What I found at each stage is summarized in the table below — the rest of the post is the experiments behind those numbers.

{{< callout type="info" title="Benchmark Environment" >}}
**GPU:** NVIDIA L40S (47.7 GB VRAM, PCIe 4.0 x16) · **CPU:** AMD EPYC 7R13 (16 cores) · **RAM:** 124 GiB DDR4 · **Storage:** 559 GB local NVMe instance store · **Stack:** Linux 6.8, PyTorch 2.10, CUDA 12.8, Python 3.10 · **Platform:** AWS EC2 Kubernetes pod (unlimited cgroups) · OS page cache flushed before every run via root privileges.
{{< /callout >}}

---

## The Pipeline Before the GPU

Every training step follows the same sequence: read raw data from disk, preprocess it on the CPU, pad and collate into tensors, transfer across the PCIe bus, then compute. The GPU — the component everyone profiles — is the last stage. Everything upstream is a potential stall.

![The LLM training data pipeline: disk I/O, CPU preprocessing, padding/collation, PCIe transfer, and GPU compute. Each stage can independently starve the GPU.](/images/posts/gpu-starve-data-pipeline-stages.png)
*Five stages, five potential stalls. The GPU only appears at the end — and it is usually the one waiting.*

---

## 20x Slower: The Cost of File-Per-Sample Storage

How fast can data be read from storage? The answer depends almost entirely on the access pattern. Training data typically lives as either many small files — one per sample, the default when nobody designs around it — or large contiguous shards, the format behind WebDataset, MosaicML Streaming, and TFRecord.

I wrote a [pure Python benchmark](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/random_vs_seq_python_simulation.py) to isolate the storage layer from everything else: 10,000 samples of 16 KB each (~156 MB total), read as individual files in shuffled order versus one concatenated shard streamed sequentially. Pure Python, no framework — to guarantee PyTorch was not a confound ([`io_benchmark_utils.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/io_benchmark_utils.py)).

```
Random Access  (10,000 individual files, shuffled):    70.39 MB/s
Sequential Access (one contiguous shard, streaming): 1,421.85 MB/s
Speedup:                                                    20.2x
```

The random path is not measuring *read* speed. It is measuring the cost of 10,000 `open()` calls, each triggering inode lookups in the filesystem's directory index, file descriptor allocation, metadata reads, and potential block pointer chasing. The 16 KB data reads are almost incidental — syscall overhead dominates. Sequential access issues one `open()` and lets the kernel's readahead and the SSD's internal parallelism do the rest.

One critical methodological point: run this without flushing the page cache and the 20x gap collapses to under 2x. After data generation, everything sits in RAM. A local dev loop reads from cache and looks fine. Deploy where the dataset exceeds RAM or lives on NFS, and the 20x penalty materializes. Benchmarking on a warm machine hides the exact failure mode that production will surface.

### Workers Mask the Penalty Without Eliminating It

A common rebuttal: "PyTorch DataLoaders with multiprocessing fix this." I repeated the experiment [through `DataLoader`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/random_vs_seq_dataloader_simulation.py) at worker counts 0, 1, 2, and 4.

![DataLoader throughput across worker counts for random-access (small files) vs sequential-access (shards). Workers close most of the gap for small files by overlapping syscall latencies, but the shard reader barely benefits — it was already near the NVMe bandwidth ceiling.](/images/posts/gpu-starve-dataloader-workers.jpg)
*Four workers give small files a ~4.8× boost. The shard reader gains only 1.2× — there was almost no latency to hide.*

Workers shrink the gap from 19.5x to 4.8x, but never close it. Four workers give small files a nearly linear speedup by overlapping syscall latencies across processes. The shard reader gains only 1.2×, because it was already operating near the sequential bandwidth ceiling of the device. On slower storage — HDD, NFS, S3 — no reasonable number of workers would close this gap. The DataLoader makes I/O *look* fine while the underlying penalty persists.

---

## The CPU Starves the GPU

Reading bytes is only the first stage. Before data reaches the GPU, the CPU must tokenize text, apply augmentations, and collate samples into batches. If the CPU cannot prepare the next batch before the GPU finishes the current one, the GPU starves — and this is perhaps the most common performance pathology in production training.

The diagnosis reduces to a simple inequality. For batch size *B*, per-sample preprocessing delay *D*, *W* workers, and GPU compute time *G*:

```
DataLoader time  =  (D × B) / max(W, 1)
If DataLoader time > G,  the GPU starves.
```

I tested this with a 5 ms per-sample delay — representative of BPE tokenization on a long document with the Llama tokenizer — batch size 32, and a 50 ms simulated GPU step ([`dataloader_bottleneck_simulation.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/dataloader_bottleneck_simulation.py)):

```
num_workers=0:  Avg Data Wait 160.59 ms, Compute 50.09 ms  → GPU idle 76%
num_workers=4:  Avg Data Wait   1.50 ms, Compute 50.09 ms  → GPU fed correctly
```

The measured 160.59 ms matches the theoretical prediction (5 ms × 32 = 160 ms) almost exactly — the bottleneck is purely CPU-bound preprocessing, no noise. With zero workers, the GPU sits idle for more than three-quarters of every training step. With four workers, the wait drops to 1.5 ms, even better than the theoretical 40 ms, because workers prefetch well ahead of consumption.

Identical code, one parameter changed.

The direction of this effect is worth pausing on: faster GPUs make the problem *worse*, not better. An H100 shrinks *G*, giving the CPU a shorter deadline before the GPU drains the prefetch queue. This is why large-scale training commonly requires 8–16 workers, and why teams move preprocessing entirely off the critical path by pre-tokenizing datasets offline.

---

## The Invisible Tax: Padding Consumes 97.7% of Compute

{{< callout type="warning" title="Counter-intuitive Finding" >}}
`nvidia-smi` reports ~100% GPU utilization. The GPU *is* computing at full throughput. But with naive batching on short-sequence datasets, only 2.3% of those compute cycles touch real tokens. The rest is attention scores, FFN layers, and gradient computation on padding positions. Standard monitoring tools cannot distinguish productive utilization from waste.
{{< /callout >}}

Variable-length samples must be padded to a fixed context length so GPUs can operate on uniform tensors. Every padding token consumes FLOPS, memory, and PCIe bandwidth for zero useful gradients. To make this concrete, I tokenized 2,000 samples from the Alpaca instruction-tuning dataset using the Llama 3.1 8B tokenizer, with a context length of 4,096 and batch size 32 ([`padding_tax_simulation.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/padding_tax_simulation.py)):

```
Token length distribution (Alpaca):
  Mean:    95 tokens
  Median:  83 tokens
  Max:    307 tokens
  Context: 4,096 tokens

Naive batching (pad to context length):    2.3% efficiency  — 97.7% padding
Sequence packing (greedy bin-packing):    98.5% efficiency  —  1.5% padding
Effective throughput improvement:          42.4x
```

Sequence packing recovers this waste with a greedy algorithm: fit the next sample into the current row; if it doesn't fit, start a new row. With a mean length of 95 tokens, roughly 43 samples fit into each 4,096-position row, so almost every position contains real data. The compound effect: 42 GPU-hours with naive batching becomes 1 GPU-hour with packing. Same data, same model, same learning signal. At cloud prices, that is the difference between a $170 experiment and a $4 experiment.

The insight here is not that packing is better — that is well known. It is that `nvidia-smi` will actively mislead you about the problem. A GPU computing at 100% utilization on 97.7% padding tokens appears healthy by every standard metric. Only measuring the *token-level* efficiency reveals the waste.

---

## What Actually Happens Inside `.to('cuda')`

The batch is assembled in CPU memory. The model sits on the GPU. Between them: the PCIe bus — a step most training code treats as instantaneous with a single `.to('cuda')` call. I wanted to measure what actually happens inside that call ([`pcie_transfer_bottleneck.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/pcie_transfer_bottleneck.py)).

With **pageable** memory — the PyTorch default — the GPU cannot read directly from the source buffer because the OS might relocate or swap the page mid-transfer. So the CUDA runtime silently allocates a temporary pinned staging buffer, copies data there via CPU `memcpy`, initiates a DMA transfer from the staging buffer to VRAM, waits for completion, and frees the staging buffer. The CPU blocks throughout. With **pinned** memory (`pin_memory=True` in DataLoader), the OS guarantees physical address stability, enabling direct DMA without the staging copy.

I measured Host-to-Device (H2D) and Device-to-Host (D2H) bandwidth at multiple tensor sizes on the L40S (PCIe 4.0 ×16, 31.51 GB/s theoretical peak):

![Host-to-Device (H2D) PCIe bandwidth across tensor sizes. Pinned memory dominates at small sizes — the regime where training batches live — and both converge near the theoretical ceiling at large sizes. Device-to-Host (D2H) PCIe bandwidth: pageable memory collapses to 1.77 GB/s at 256 MB and above — a 7.5× penalty versus pinned. This directly throttles checkpointing and ZeRO-Offload.](/images/posts/gpu-starve-padding-tax.jpg)
*H2D: pinned memory matters most at small tensor sizes, exactly where training batches operate. D2H: pageable memory collapses to 1.77 GB/s at large sizes. An 8 GB checkpoint takes 4.5 seconds instead of 0.6.*

H2D follows expectations — pinned memory helps most at small sizes and the two converge at large sizes. The D2H direction was more alarming. Pageable D2H collapses to **1.77 GB/s** at 256 MB and above — a **7.5× penalty** versus pinned. For DeepSpeed ZeRO-Offload, which continuously shuttles optimizer states between GPU and CPU, this gap directly throttles offloading throughput. The measured peak of ~13.5 GB/s is ~43% of the theoretical maximum; the remainder is lost to protocol overhead, IOMMU translation, and likely cross-NUMA-socket traffic in the Kubernetes pod.

### Pinned Memory Frees the CPU — Pageable Does Not

Raw bandwidth is only half the story. The deeper benefit of pinned memory is that the CPU is free to prepare the next batch while the DMA engine transfers data autonomously. I designed a direct test: start a 256 MB transfer, immediately execute 5 ms of CPU matrix multiplies, then synchronize. If the CPU is truly unblocked, the total wall-clock time should equal the transfer time alone.

```
pinned + non_blocking:
  Transfer 19.03 ms + CPU work 5.48 ms = Total 19.04 ms  → PARALLEL

pageable + blocking (default):
  Transfer 25.93 ms + CPU work 5.48 ms = Total 25.93 ms  → SEQUENTIAL

non_blocking without pin_memory:
  Total 29.46 ms  → SLOWER than blocking
```

The 5.48 ms of CPU work happened completely for free — the total is virtually identical to the transfer time alone. Every other combination serialized, with totals near 26 ms. One critical detail: `non_blocking=True` *without* `pin_memory=True` is not just useless — it is actively slower than plain blocking. The flag only has meaning when paired with pinned memory.

In a training loop, this difference in execution model changes the timeline fundamentally. With pageable memory, the CPU blocks for the full duration of the transfer — it cannot start preparing the next batch until the DMA completes:

![Blocked DMA timeline: CPU is stalled during the Host-to-Device transfer. Batch preparation for the next step cannot begin until the transfer finishes, serializing CPU and GPU work.](/images/posts/gpu-starve-dma-blocked.jpg)
*Pageable memory forces the CPU to wait. The next batch cannot be prepared until the current transfer is done.*

With pinned memory and `non_blocking=True`, the DMA engine operates independently. The CPU is free the moment the transfer is enqueued:

![Parallel DMA timeline: with pinned memory and non_blocking=True, the CPU prepares the next batch while the DMA engine transfers the current one. Both sides stay busy — neither the GPU nor the CPU idles.](/images/posts/gpu-starve-dma-parallel.jpg)
*Pinned memory with non_blocking: DMA and CPU batch prep overlap completely. Zero idle time on either side.*

The first timeline has a hard serial dependency at every step. The second pipelines everything — the GPU never drains the queue, and the CPU never stalls waiting for a transfer it already issued.

One container-specific caveat: pinned memory counts against cgroup limits and cannot be swapped. Pin too aggressively in Kubernetes and the OOM killer arrives. CPU CFS throttling also inflates transfer wall-clock times. Always check your CPU quota before attributing latency to the hardware.

---

## The Full Loop: Where the Effects Compound

Each stage in isolation shows a clear penalty. The question is how these effects interact when the full training loop runs — do they compound, or does one bottleneck dominate and mask the rest?

I combined all stages in a real training loop: ResNet-50 on 100,000 synthetic JPEGs (224×224), comparing **ImageFolder** (100,000 individual files) vs. **MosaicML Streaming** (large shards), with `pin_memory=True`, under `torch.profiler` with explicit phase markers (`DATALOADER_WAIT`, `H2D_TRANSFER`, `FORWARD`, `BACKWARD`, `OPTIMIZER`). OS cache flushed after CUDA warmup, before profiling ([`streaming_loader_lib_profiling.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/analysis/streaming_loader_lib_profiling.py)).

```
Synchronous (num_workers=0, cold storage):
              DataLoader    H2D      Compute    Total       Throughput
ImageFolder:   130.5 ms    3.1 ms    94.6 ms   228.2 ms    280.5 img/s
Streaming:      92.1 ms    3.2 ms    94.6 ms   189.9 ms    337.1 img/s

Parallel (num_workers=4, cold storage):
              DataLoader    H2D      Compute    Total       Throughput
ImageFolder:     0.4 ms    3.1 ms    94.6 ms    98.0 ms    652.8 img/s
Streaming:       0.4 ms    3.1 ms    94.6 ms    98.1 ms    652.5 img/s
```

With zero workers on ImageFolder, the GPU is idle for **57%** of every step — more time waiting than computing. Streaming cuts DataLoader wait by 1.42× (92 ms vs. 131 ms), but both formats still leave the GPU data-starved. The improvement is more modest than the 20× from the pure I/O benchmark because the full loop adds large constant costs that dilute the storage-pattern difference: JPEG decoding (~30–40 ms of CPU work, identical for both formats) and GPU compute (94.6 ms, identical for both). The access pattern advantage is real, but it is one term in a larger sum.

With 4 workers, the picture changes completely. Both strategies converge to near-zero DataLoader wait (~0.4 ms) and throughput jumps **2.33×** to 653 img/s. The training step collapses to compute + H2D = 97.7 ms, with effectively zero data stall.

The finding that surprised me most: on fast local NVMe with enough workers, the data format does not matter. The NVMe sustains enough random IOPS that workers never drain the prefetch queue. But this is an artifact of fast local storage. On HDD, NFS, or object storage, the shard advantage persists exactly as the isolated benchmark predicted. The insurance pays off precisely when you need it most — in production, on shared infrastructure, at scale.

### Every Stage, Measured

Across all seven benchmarks, the pattern is consistent: the naive default at each stage carries a large, measurable penalty, and the fix is almost always a single configuration change or data format decision.

| Stage | Bottleneck | Measured Impact | Fix |
|---|---|---|---|
| Storage I/O | Random access to many small files | 70 vs. 1,422 MB/s — **20.2×** penalty | Shard into contiguous files (WebDataset, Streaming, TFRecord) |
| DataLoader prefetch | Workers mask but cannot fix I/O ceiling | 19.5× gap shrinks to 4.8× with 4 workers | Benchmark with cold cache; tune `num_workers` |
| CPU preprocessing | CPU slower than GPU consume rate | 160 ms wait — **76%** GPU idle (0 workers) | Pre-tokenize offline; add workers |
| Padding | Padding tokens waste GPU FLOPS | 2.3% efficiency naive; 98.5% packed — **42.4×** gain | Sequence packing |
| PCIe H2D | Pageable staging copy blocks CPU | **1.52×** pinned advantage at small sizes | `pin_memory=True` + `non_blocking=True` |
| PCIe D2H | Pageable collapses at large sizes | 1.77 vs. 13.2 GB/s — **7.5×** penalty | Pinned destination tensors |
| CPU/GPU overlap | Only pinned + non_blocking overlaps | 100% overlap vs. 0% for all other combos | Never use `non_blocking` without `pin_memory` |
| End-to-end | Weakest link determines throughput | 57% idle → 0.4% = **2.33×** throughput | Profile with `DATALOADER_WAIT` markers |

---

## Principles, Not Parameters

A training pipeline moves data through five stages, and the GPU — the most expensive component — is the last one. It is also usually the one waiting.

Several observations generalize beyond these specific benchmarks. The 20× I/O gap only appears when the page cache is cold, which means development environments systematically hide the penalty that production will pay. The padding tax is invisible to `nvidia-smi` because GPU utilization measures *activity*, not *productive* activity — the distinction matters enormously for cost accounting. PCIe pinning matters most for D2H transfers and checkpointing, not H2D, a direction most profiling guides neglect. And `num_workers` remains the single highest-leverage parameter in most training scripts, with effects that grow as GPUs get faster, not smaller.

These are not tuning tips. They are structural properties of the pipeline — the kind that reappear in different forms across frameworks, hardware generations, and model architectures. Measuring them once, in isolation, makes the diagnosis faster every time they surface in a new system.

The next post moves beyond the single-node data pipeline into distributed training — specifically, what `torch.profiler` traces reveal about NCCL communication overhead and how the compute-to-communication ratio evolves across model sizes and network topologies.
