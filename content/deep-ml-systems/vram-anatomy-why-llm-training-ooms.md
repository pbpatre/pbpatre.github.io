---
title: "VRAM Anatomy: Why Your LLM Training OOMs"
date: 2026-04-02
draft: false
slug: "vram-anatomy-why-llm-training-ooms"
author: "Pratik Patre"
description: "A 3.1B-parameter model weighs 6.2 GB in BF16. Training it demands 148 GB — the optimizer alone is 4× larger than the model. The question is where those 148 GB actually go, and what each optimization technique trades away to reclaim them."
summary: "A 3.1B-parameter model weighs 6.2 GB in BF16. Training it demands 148 GB — the optimizer alone is 4× larger than the model. This post dissects where those 148 GB actually go and what each optimization technique trades away to reclaim them."
tags: ["LLM", "Training", "VRAM", "OOM", "DDP", "FSDP", "Activation Checkpointing", "GPU Memory", "Distributed Training", "PyTorch", "AdamW", "H200"]
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

A GPT-XXL model has 3.1 billion parameters. In BF16, those parameters occupy 6.2 GB of VRAM. I loaded this model on an NVIDIA H200 — 141 GB of VRAM, a GPU that costs north of $30,000 — and it crashed with an out-of-memory error on the first training step. Peak memory demand: 148 GB. The model weights accounted for 4% of it.

The remaining 96% is optimizer states, gradients, and activations — memory objects that most engineers never reason about directly. AdamW alone consumes 8 bytes per parameter (two FP32 moment estimates), making the optimizer 4× larger than the BF16 model it serves. Activations scale with batch size, sequence length, and — for attention — quadratically with context length. A 4,096-token sequence across 60 transformer layers generates roughly 98 GB of activation memory. These are not incidental costs. They are the dominant terms.

I ran a four-quadrant experiment on 2× H200 GPUs — every combination of DDP, FSDP, and activation checkpointing — to measure what each optimization technique actually saves and what it costs. The code is [`ddp_fsdp_oom_demo.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/distributed/ddp_fsdp_oom_demo.py) in the benchmark repo.

{{< callout type="info" title="Benchmark Environment" >}}
**GPU:** 2× NVIDIA H200 (141 GB VRAM each, NVLink interconnect, ~900 GB/s) · **Model:** GPT-XXL (3.1B parameters, 60 layers, 32 attention heads) · **Config:** BF16 training, AdamW optimizer, batch size 2, sequence length 4,096 · **Stack:** PyTorch 2.10, CUDA 12.8 · **Code:** [`training/distributed/`](https://github.com/pbpatre/llm-system-benchmarks/tree/main/training/distributed)
{{< /callout >}}

---

## What Actually Lives in VRAM

VRAM is not a single pool. It is occupied by distinct classes of objects with entirely different scaling laws, lifetimes, and dependencies. Treating it as a monolithic bucket — "the model is 6 GB so it should fit on an 80 GB GPU" — is precisely how OOM errors surprise experienced engineers.

GPU memory divides into two categories: **static memory** that scales with parameter count, and **dynamic memory** that scales with batch size and sequence length.

### Static Memory: The Model State

Three objects scale with the number of parameters *P* and are independent of the data passing through the model:

| Component | Size per parameter | For 3.1B params | Lifetime |
|---|---|---|---|
| Model weights (BF16) | 2 bytes | 6.2 GB | Permanent — loaded at init |
| Gradients (BF16) | 2 bytes | 6.2 GB | Ephemeral — created in backward, destroyed after `optimizer.step()` |
| AdamW optimizer (FP32) | 8 bytes (2 states × 4 bytes) | 24.8 GB | Permanent — two FP32 moment tensors per parameter |
| **Total static state** | **12 bytes** | **37.2 GB** | |

The optimizer is the largest single component — 4× the memory of the model weights. This surprises most engineers encountering it for the first time, but the arithmetic is straightforward: AdamW maintains a first-moment estimate (momentum) and a second-moment estimate (variance) for every parameter, both in FP32 for numerical stability, regardless of the training precision.

### Dynamic Memory: Activations

Activations are the intermediate outputs of every layer — the values backpropagation needs to compute gradients via the chain rule. They have two components with fundamentally different scaling behavior:

**FFN/linear activations** scale as O(b × s × d_model) — linearly with batch size and sequence length. These determine maximum throughput: how many tokens per second the GPU can process.

**Attention matrices** scale as O(b × h × s²) — quadratically with sequence length. For 32 attention heads and a 4,096-token sequence, each layer materializes a 32 × 4,096 × 4,096 score matrix. Across 60 layers, these attention matrices alone can consume over 100 GB. This quadratic term is the hard constraint on maximum context length, and it is the reason Flash Attention was invented — to compute attention without materializing the full S×S matrix in HBM.

Both types of activation are born during the forward pass, held in memory for the entire forward pass duration, and then consumed and destroyed sequentially (top-to-bottom) during the backward pass. Their peak memory footprint is reached at the boundary between forward and backward — the moment when all layers' activations coexist simultaneously.

| Component | Scaling | Controls | Notes |
|---|---|---|---|
| FFN activations | O(b × s × d_model) — linear | Max batch size / throughput | Activation checkpointing targets these |
| Attention matrices | O(b × h × s²) — quadratic | Max context length | Flash Attention avoids materializing these |

For the GPT-XXL configuration tested here (batch size 2, sequence length 4,096, 60 layers), total activation memory was approximately **98 GB** — more than 2.5× the entire static model state.

---

## Matching the Solution to the Memory Class

Once VRAM is understood as two independent axes — static state (parameter-proportional) and dynamic memory (data-proportional) — the major PyTorch distributed training techniques map cleanly onto which axis they attack.

### DDP: Replicate Everything, Scale Compute

Distributed Data Parallel replicates the *entire* model, optimizer, and gradients onto every GPU. Each GPU processes a different data shard; during the backward pass, an `AllReduce` synchronizes gradients across workers before the optimizer step. DDP does not save any memory — it trades money (more GPUs) for speed (more data parallelism). If the model fits on one GPU, DDP is the fastest distributed strategy because the full model is local and no communication is needed during the forward pass.

### FSDP: Shard the Static State

Fully Sharded Data Parallel attacks the *static* axis. Instead of replicating the model, FSDP shards weights, gradients, and optimizer states across all GPUs. When a layer needs to execute, GPUs briefly reconstruct its parameters via an `AllGather`, compute, and immediately discard the gathered copy. With *N* GPUs, static memory per GPU drops by a factor of *N*. Activations, however, are **not sharded** — every GPU still materializes the full activation footprint for its local data.

### Activation Checkpointing: Trade Compute for Memory

Activation checkpointing attacks the *dynamic* axis. Instead of retaining every layer's activations for the backward pass, checkpointing discards them during the forward pass and recomputes them on-the-fly during backpropagation. This trades compute (every checkpointed layer runs its forward pass twice) for memory (activations no longer accumulate across the full depth of the model).

For extremely long contexts where the O(s²) attention matrix exceeds VRAM even with checkpointing, **Flash Attention** eliminates the problem at a lower level by fusing the attention computation into a single kernel that never materializes the full score matrix in HBM.

---

## 148 GB on a 140 GB GPU: The Four-Quadrant Experiment

To demonstrate that static and dynamic memory are independent axes — and that each optimization targets exactly one — I ran a four-quadrant matrix: every combination of DDP vs. FSDP and with vs. without activation checkpointing. Same model, same data, same hardware. The only variable is which memory optimization is active ([`ddp_fsdp_oom_demo.py`](https://github.com/pbpatre/llm-system-benchmarks/blob/main/training/distributed/ddp_fsdp_oom_demo.py)).

| Configuration | Peak VRAM (per GPU) | Throughput (tokens/s) | Status |
|---|---|---|---|
| DDP | >148 GB | — | ❌ OOM at step 1 |
| FSDP only | 117 GB | ~10,928 | ✅ Fits (barely) |
| DDP + checkpointing | 66.7 GB | ~8,325 | ✅ Comfortable |
| FSDP + checkpointing | 29.8 GB | ~8,381 | ✅ Massive headroom |

### DDP Alone: 148 GB → OOM

```
torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=ddp \
         --model_size=xxl --batch_size=2 --seq_len=4096
```

```
Peak memory before OOM: 148.02 GB  (GPU capacity: 139.80 GB)

  Model weights (BF16):     6.2 GB
  Gradients (BF16):         6.2 GB
  Optimizer (AdamW FP32):  37.2 GB
  Activations:            ~98.4 GB   ← the dominant term
  ─────────────────────────────────
  TOTAL:                 ~148.0 GB → EXCEEDS 140 GB
```

The static state (37.2 GB model + optimizer + gradients) plus 98 GB of activations exceeds the H200's capacity by 8 GB. The attention scores alone — `batch × heads × seq² × 2 bytes` across 60 layers — account for the bulk of the activation cost.

### FSDP Alone: Shards Static State, Activations Unchanged

```
torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=fsdp \
         --model_size=xxl --batch_size=2 --seq_len=4096
```

```
Peak memory: 117.02 GB  |  Throughput: 10,928 tokens/sec

  Model weights (sharded):  3.1 GB   (50% reduction ↓)
  Gradients (sharded):      3.1 GB   (50% reduction ↓)
  Optimizer (sharded):     18.6 GB   (50% reduction ↓)
  Activations:            ~92.2 GB   (UNCHANGED)
  ─────────────────────────────────
  TOTAL:                 ~117.0 GB → FITS
```

FSDP cut the static model state in half — from ~50 GB to ~25 GB — bringing total memory under the 140 GB threshold. But activations remain the dominant cost at 92 GB, completely untouched by sharding. This workload fits, but barely. Increasing batch size or sequence length would OOM again.

### DDP + Activation Checkpointing: Static State Intact, Activations Collapsed

```
torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=ddp \
         --model_size=xxl --batch_size=2 --seq_len=4096 \
         --use_activation_checkpointing
```

```
Peak memory: 66.74 GB  |  Throughput: 8,325 tokens/sec (-24%)

  Model weights (BF16):     6.2 GB
  Gradients (BF16):         6.2 GB
  Optimizer (AdamW FP32):  37.2 GB   (UNCHANGED — still replicated)
  Activations (checkpt):  ~17.1 GB   (82% reduction ↓)
  ─────────────────────────────────
  TOTAL:                  ~66.7 GB → FITS COMFORTABLY
```

Activation checkpointing reduced dynamic memory by 82% — from ~98 GB to ~17 GB. The tradeoff is visible in throughput: a 24% drop, because every checkpointed layer recomputes its forward pass during backpropagation. The static state remains the full replicated 37.2 GB because DDP does not shard.

### FSDP + Activation Checkpointing: Both Axes Compressed

```
torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=fsdp \
         --model_size=xxl --batch_size=2 --seq_len=4096 \
         --use_activation_checkpointing
```

```
Peak memory: 29.81 GB  |  Throughput: 8,381 tokens/sec

  Model weights (sharded):  3.1 GB   (50% reduction ↓)
  Gradients (sharded):      3.1 GB   (50% reduction ↓)
  Optimizer (sharded):     18.6 GB   (50% reduction ↓)
  Activations (checkpt):   ~5.0 GB   (82% reduction ↓)
  ─────────────────────────────────
  TOTAL:                  ~29.8 GB → MASSIVE HEADROOM
```

Combining both techniques attacks both memory axes simultaneously. Static state is sharded (50% per-GPU reduction), activations are checkpointed (82% reduction). A workload that OOMed a 140 GB H200 now fits in under 30 GB — a 5× reduction that brings it within range of a single consumer-grade GPU. The 80% total reduction from baseline DDP is not magic; it is the product of two independent compressions applied to two independent memory pools.

---

## The Tradeoffs Are Structural

The four-quadrant results make FSDP + checkpointing look like a universal solution. It is not. Each optimization trades a different resource for memory, and the cost depends entirely on the hardware topology.

**DDP is the fastest architecture when the model fits.** Because the entire model lives locally on each GPU, no network communication is needed during the forward pass. DDP is structurally optimal for models that fit comfortably in single-GPU memory — standard ResNets, BERT-scale models, small LLMs. The moment you are forced to reduce batch size to avoid OOM, DDP has become the bottleneck and it is time to move on.

**Activation checkpointing trades compute for memory.** The 24% throughput reduction in these experiments is representative: every checkpointed layer runs its forward pass twice. This is a fixed computational tax. However, if checkpointing enables doubling or quadrupling the batch size, the throughput gained from larger, more GPU-efficient matrix multiplications can more than offset the recomputation cost. Empty VRAM does not contribute to training speed — it is always worth filling it with useful work.

{{< callout type="warning" title="FSDP's Hidden Dependency: Network Bandwidth" >}}
FSDP trades *network bandwidth* for memory. Because no GPU holds the full model, every layer execution requires an `AllGather` to reconstruct parameters from other GPUs. In these experiments, FSDP showed negligible slowdown compared to DDP — because the H200s were connected via NVLink at ~900 GB/s, fast enough to completely hide the communication latency behind computation. On standard Ethernet or InfiniBand across separate nodes, FSDP introduces a significant "communication bubble" where GPUs idle waiting for parameters. The interconnect bandwidth determines whether FSDP is free or expensive.
{{< /callout >}}

The decision tree is straightforward. If the model fits on one GPU, use DDP. If the static state (weights + optimizer + gradients) exceeds ~50% of VRAM, FSDP is mandatory — but only if the interconnect can sustain the `AllGather` throughput. If activations are the constraint (large batch size, long context), use checkpointing and accept the ~25% compute tax. If both axes are saturated, combine both.

---

## Principles, Not Rules of Thumb

The central insight of this analysis is that VRAM is not one number — it is at least five distinct memory pools, each with its own scaling law and lifetime. The optimizer is 4× the model weights. Activations are 2.5× the entire static state. Attention memory scales quadratically with context length while everything else scales linearly. These are structural facts about the mathematics of training, not properties of any specific framework or GPU.

The four-quadrant experiment demonstrates a corollary: optimization techniques are not interchangeable. FSDP and activation checkpointing are not two ways to "reduce memory." They target orthogonal axes — static vs. dynamic — and their effects are multiplicative when combined precisely because they operate on independent memory pools. Choosing the wrong one wastes either network bandwidth or compute cycles for zero memory benefit on the axis that is actually saturated.

The next post in this series will examine the cost that FSDP hides on NVLink — the NCCL communication overhead that becomes the dominant bottleneck when `AllGather` and `ReduceScatter` must traverse slower interconnects. The gap between NVLink and network-bound training is dramatic, and worth measuring carefully.
