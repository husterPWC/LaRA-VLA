#!/usr/bin/env python3
"""Verify rank-to-device mapping and a real NCCL collective."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    physical_devices = visible_devices.split(",") if visible_devices else []
    physical_device = (
        physical_devices[local_rank]
        if local_rank < len(physical_devices)
        else "unknown"
    )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(
            f"rank={rank} local_rank={local_rank} "
            f"logical_cuda={torch.cuda.current_device()} "
            f"physical_gpu={physical_device}",
            flush=True,
        )

        dist.barrier(device_ids=[local_rank])
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise RuntimeError(
                f"all_reduce mismatch: got {value.item()}, expected {expected}"
            )
        dist.barrier(device_ids=[local_rank])
        if rank == 0:
            print(
                f"R4 NCCL PREFLIGHT PASS: world_size={world_size} "
                f"all_reduce_sum={value.item()}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
