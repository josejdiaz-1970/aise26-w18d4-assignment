# Jose Diaz
# W18D4
# 02-01-26

# REPRO — W18D4 Scale-Ready Training Loop (DDP + Accumulation)

## Environment
- OS: Google Colab (Linux)
- Python: 3.12 (Colab default)
- PyTorch: (Colab-installed)
- Backend: Gloo (CPU)

## What this repo contains
- A minimal DDP training loop that runs locally (single node) using `torchrun`
- Gradient accumulation (accum_steps) to create an effective batch size larger than per-step batch size
- `metrics.csv` logging for review

## Run Commands

### 1) Install / get the repo
```bash
git clone https://github.com/josejdiaz-1970/aise26-w18d4-assignment.git
cd aise26-w18d4-assignment
