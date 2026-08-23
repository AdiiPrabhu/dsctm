"""Distributed samplers.

The single most dangerous default in PyTorch DDP evaluation:

    torch.utils.data.DistributedSampler(dataset)          # drop_last=False

pads the index list up to a multiple of ``world_size`` by **repeating samples from the
front**. For training that is a negligible nuisance. For evaluation it silently
double-counts participants.

Concretely, on the DAIC-WOZ official test split (47 sessions) at world_size 2, the padded
sampler emits 48 indices: session 0 is scored twice and enters the macro-F1 twice. On a
47-session set that is a ~2% distortion applied non-uniformly, and nothing in the output
says it happened.

``UnpaddedDistributedSampler`` partitions without padding. Ranks receive unequal counts;
the union is exactly the dataset, each index exactly once. Unequal counts are safe here
because evaluation runs under ``torch.no_grad`` and performs no gradient collective — the
usual reason DDP wants even shards does not apply.

NEVER use ``UnpaddedDistributedSampler`` for training with DDP: uneven batch counts make
ranks call different numbers of backward passes and the job deadlocks in the gradient
all-reduce.
"""
from __future__ import annotations

from typing import Iterator, Sequence, Sized

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Sampler


def _resolve(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if world_size is None:
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    if rank is None:
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside [0, {world_size})")
    return rank, world_size


class UnpaddedDistributedSampler(Sampler[int]):
    """Evaluation sampler that partitions indices with NO padding and NO duplication.

    Guarantees, asserted by ``tests/test_distributed_sampler.py``:

    * ``union(shard_r for r in ranks) == set(range(len(dataset)))``
    * ``sum(len(shard_r)) == len(dataset)``
    * shards are pairwise disjoint
    * shard sizes differ by at most one

    Deterministic and order-stable: rank ``r`` always receives ``indices[r::world_size]``
    for the same dataset length, independent of process start order.
    """

    def __init__(self, dataset: Sized, num_replicas: int | None = None,
                 rank: int | None = None, indices: Sequence[int] | None = None) -> None:
        self.rank, self.num_replicas = _resolve(rank, num_replicas)
        base = list(range(len(dataset))) if indices is None else list(indices)
        self.total_size = len(base)
        # Strided partition: no padding, no truncation, sizes differ by at most one.
        self.indices = base[self.rank::self.num_replicas]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:  # noqa: D401 - API parity with DistributedSampler
        """No-op. Evaluation order is fixed by construction; there is nothing to reshuffle."""


def make_train_sampler(dataset: Sized, ctx=None, shuffle: bool = True,
                       seed: int = 0, drop_last: bool = False) -> DistributedSampler | None:
    """Training sampler. Returns ``None`` when not running distributed.

    Uses the stock padded ``DistributedSampler`` on purpose: DDP requires every rank to
    execute the same number of backward passes, so even shards are mandatory here.

    The caller MUST invoke ``sampler.set_epoch(epoch)`` at the top of every epoch. Without
    it the shuffle order is identical in every epoch, which quietly degrades training and
    is invisible in the loss curve for small datasets.
    """
    if ctx is not None and not ctx.is_distributed:
        return None
    if not (dist.is_available() and dist.is_initialized()) and ctx is None:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=ctx.world_size if ctx is not None else None,
        rank=ctx.rank if ctx is not None else None,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def make_eval_sampler(dataset: Sized, ctx=None) -> UnpaddedDistributedSampler | None:
    """Evaluation sampler: unpadded, non-duplicating, ``shuffle=False`` by construction.

    Returns ``None`` when not distributed, so the caller falls back to sequential
    single-process evaluation over the whole split.
    """
    if ctx is not None and not ctx.is_distributed:
        return None
    if not (dist.is_available() and dist.is_initialized()) and ctx is None:
        return None
    return UnpaddedDistributedSampler(
        dataset,
        num_replicas=ctx.world_size if ctx is not None else None,
        rank=ctx.rank if ctx is not None else None,
    )


def audit_sampler_partition(dataset_len: int, world_size: int) -> dict:
    """Offline proof that the eval partition is a true partition, for the run record."""

    class _Len:
        def __len__(self) -> int:
            return dataset_len

    shards = [list(UnpaddedDistributedSampler(_Len(), world_size, r)) for r in range(world_size)]
    flat = [i for shard in shards for i in shard]
    sizes = [len(s) for s in shards]
    return {
        "dataset_len": dataset_len,
        "world_size": world_size,
        "shard_sizes": sizes,
        "total_emitted": len(flat),
        "unique_emitted": len(set(flat)),
        "covers_exactly_once": len(flat) == dataset_len == len(set(flat)),
        "max_size_imbalance": (max(sizes) - min(sizes)) if sizes else 0,
        "padded_sampler_would_emit": ((dataset_len + world_size - 1) // world_size) * world_size,
        "duplicates_avoided": (((dataset_len + world_size - 1) // world_size) * world_size
                               - dataset_len),
    }


def loader_kwargs_for_param(num_workers: int | None = None,
                            pin_memory: bool | None = None) -> dict:
    """DataLoader settings tuned for a PARAM GPU node.

    A GPU node has 40 cores and 2 V100s, i.e. 2 ranks per node. Allocating 20 workers per
    rank oversubscribes once the main process, NCCL threads and Lustre I/O are counted;
    it also multiplies resident memory because each worker forks the in-memory
    ``TensorDataset``. 4 workers per rank is the sane default and is overridable.
    """
    workers = 4 if num_workers is None else int(num_workers)
    pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    kwargs = {
        "num_workers": workers,
        "pin_memory": pin,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs
