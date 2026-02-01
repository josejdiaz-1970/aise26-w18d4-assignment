import os
import csv
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    # torchrun sets these env vars for you
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=120),
    )

    torch.manual_seed(42 + rank)
    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def main():
    rank, world_size = setup()

    model = torch.nn.Linear(10, 2)
    ddp_model = DDP(model)

    optim = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- Gradient accumulation ----
    accum_steps = 4
    batch_size = 32
    num_steps = 20
    effective_batch_size = batch_size * world_size * accum_steps

    # ---- metrics.csv (rank 0 only) ----
    metrics_f = None
    writer = None
    if rank == 0:
        print(f"[Gloo] initialized world_size={world_size}")

    optim.zero_grad(set_to_none=True)

    for step in range(num_steps):
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))

        out = ddp_model(x)
        loss = loss_fn(out, y) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

            logged_loss = loss.detach().item() * accum_steps

            if rank == 0:
                print("step", step, "loss", logged_loss)
                writer.writerow([step, logged_loss, effective_batch_size, world_size, accum_steps])
                metrics_f.flush()

    if rank == 0 and metrics_f is not None:
        metrics_f.close()

    cleanup()


if __name__ == "__main__":
    main()
