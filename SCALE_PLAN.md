# Jose Diaz
# W18D4
# 02-01-26

# SCALE_PLAN — DDP vs FSDP vs TP/PP Decision Tree (W18D4)

## Goal
Provide a “scale-ready” training plan: how we would scale this training loop as model size, batch size, and hardware grow.

---

## Decision Tree

### Step 1 — Does the model fit in GPU memory (per device) with optimizer states?
- **YES** → Start with **DDP**
- **NO** → Go to Step 2

### Step 2 — Does the model *almost* fit, but optimizer + gradients push it over the limit?
- **YES** → Use **FSDP** (shard parameters + gradients + optimizer state)
- **NO** → Go to Step 3

### Step 3 — Is the model so large that even sharding parameters isn’t enough?
- **YES** → Consider **Tensor Parallelism (TP)** and/or **Pipeline Parallelism (PP)**
  - TP: split large matrix multiplies across GPUs
  - PP: split layers across GPUs and pipeline microbatches
- **NO** → FSDP is usually the best next step after DDP

---

## Recommended Path for “Toy/Small Model” (this assignment)
**DDP + gradient accumulation** is the correct choice because:
- It’s the simplest correct distributed baseline
- It matches the common production entry point for scaling training
- It keeps correctness/evidence reviewable

---

## What I would measure for scaling evidence (and why)

### Throughput + efficiency
- **samples/sec** (per GPU and total): tells you real training speed
- **step time** (mean/p50/p95): captures stability + outliers
- **scaling efficiency**:  
  `throughput(world_size=N) / (N * throughput(world_size=1))`  
  tells you how much overhead you’re paying

### Communication overhead
- **all-reduce time** (DDP comm): identify if comm dominates compute
- **overlap %** (compute/comm overlap): indicates if you’re bandwidth-bound

### Memory footprint
- **peak GPU memory** (allocated/reserved): determines whether you can increase batch/model size
- If memory is the bottleneck → evaluate FSDP or activation checkpointing

### Convergence correctness
- **loss curves** (compare world_size=1 vs world_size>1)
- Ensure same “effective batch size” yields similar convergence trends

---

## Risks & Mitigations

### Risk: DDP becomes bandwidth-bound at higher world_size
- **Mitigation:** gradient bucketing, overlap tuning, faster interconnect, mixed precision, reduce gradient size

### Risk: Batch size increase changes convergence
- **Mitigation:** keep effective batch fixed via accumulation; tune LR schedule (linear scaling rule), validate convergence vs baseline

### Risk: Model no longer fits in memory as model grows
- **Mitigation:** move from DDP → FSDP; consider activation checkpointing; use CPU offload if needed

---

## Summary Recommendation
- For small/medium models that fit per GPU: **DDP** is the default
- If memory becomes the limiting factor: move to **FSDP**
- For very large transformer-scale models: consider **TP/PP** (often combined with FSDP)
