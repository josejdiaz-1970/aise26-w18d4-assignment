# Jose Diaz
# W18D4
# 02-01-26


---

## `KNOWN_ISSUES.md` (3+ distributed failure modes + mitigations)

```md
# KNOWN_ISSUES — Distributed Failure Modes + Mitigations

This project uses PyTorch DistributedDataParallel (DDP) with the Gloo backend for CPU.
 Below are common failure modes that can occur in distributed training and how to mitigate them.

---

## 1) Rendezvous / init hangs (timeout)
**Symptom**
- Script stalls during `dist.init_process_group(...)`
- Eventually fails with a timeout (e.g., `RuntimeError: Wait timeout`)

**Root Causes**
- Ranks are not starting together
- Wrong rendezvous configuration (bad init_method / master port in use)
- Network restrictions or port conflicts

**Mitigations**
- Prefer `torchrun --standalone --nproc_per_node=2 script.py` for single-node testing
- Use a different port if needed: `--master_port=29511`
- Increase init timeout during debugging (e.g., 120s)

- Note: This works sporadically on Windows machines. Windows 11 Home 23H2.

---

## 2) Deadlock due to mismatched collective calls
**Symptom**
- Program hangs mid-training, often during backward()
- One rank continues while another stalls

**Root Causes**
- Conditional logic causes only some ranks to enter a collective operation
- Different number of forward/backward steps across ranks

**Mitigations**
- Ensure all ranks execute the same training loop structure
- Keep conditionals outside the core forward/backward path (or ensure they are rank-consistent)
- Use `dist.barrier()` temporarily during debugging to confirm synchronization points

---

## 3) Incorrect gradient accumulation (sync mistakes / wrong effective batch)
**Symptom**
- Optimization behaves strangely, unstable loss, or “no learning”
- Effective batch size calculations don’t match actual optimizer stepping

**Root Causes**
- Calling `optimizer.step()` every step instead of every `accum_steps`
- Not scaling the loss by `1/accum_steps`
- Logging loss before scaling / after scaling inconsistently

**Mitigations**
- Divide loss by `accum_steps` before `backward()`
- Only call `optimizer.step()` every `accum_steps` iterations
- Document effective batch size: `batch_size * world_size * accum_steps`

---

## 4) Divergence or mismatch caused by data sharding errors
**Symptom**
- Poor convergence or inconsistent behavior between runs
- With real datasets, some ranks may see duplicated samples

**Root Causes**
- Not using `DistributedSampler` for training data
- Using `shuffle=True` in DataLoader without a distributed sampler

**Mitigations**
- Use `torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)`
- Call `sampler.set_epoch(epoch)` each epoch for proper shuffling
- Keep evaluation on rank 0 or aggregate metrics across ranks

---

## 5) Rank output interleaving / noisy logs (not a functional failure)
**Symptom**
- Console logs look garbled (rank messages overlap)

**Root Causes**
- Multiple processes writing to stdout simultaneously

**Mitigations**
- Print only from rank 0
- Write structured logs to file (CSV/JSON) from rank 0
