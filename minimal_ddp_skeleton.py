import os
import csv
import tempfile
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank: int, world_size: int):
    # Use a file-based init method (reliable for local spawn)
    init_file = os.path.join(tempfile.gettempdir(), "ddp_init_file")

    # Rank 0 clears the file so reruns work
    if rank == 0 and os.path.exists(init_file):
        try:
            os.remove(init_file)
        except OSError:
            pass

    init_method = "file:///" + init_file.replace("\\", "/")

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )

    torch.manual_seed(42 + rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank: int, world_size: int):
    setup(rank, world_size)
    global_rank = dist.get_rank()

    # ----- Model / DDP -----
    model = torch.nn.Linear(10, 2)
    ddp_model = DDP(model)

    optim = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ----- Gradient Accumulation -----
    accum_steps = 4
    batch_size = 32
    num_steps = 20
    effective_batch_size = batch_size * world_size * accum_steps

    # ----- metrics.csv (rank 0 only) -----
    metrics_f = None
    writer = None
    if global_rank == 0:
        metrics_f = open("metrics.csv", "w", newline="")
        writer = csv.writer(metrics_f)
        writer.writerow(["step", "loss", "effective_batch_size", "world_size", "accum_steps"])

    optim.zero_grad(set_to_none=True)

    for step in range(num_steps):
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 2, (batch_size,))

        out = ddp_model(x)
        loss = loss_fn(out, y) / accum_steps  # IMPORTANT: scale for accumulation
        loss.backward()

        do_step = ((step + 1) % accum_steps == 0)
        if do_step:
            optim.step()
            optim.zero_grad(set_to_none=True)

            # "real" loss value (undo division) for logging
            logged_loss = loss.detach().item() * accum_steps

            if global_rank == 0:
                print("step", step, "loss", logged_loss)
                writer.writerow([step, logged_loss, effective_batch_size, world_size, accum_steps])
                metrics_f.flush()

    if global_rank == 0 and metrics_f is not None:
        metrics_f.close()

    cleanup()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
