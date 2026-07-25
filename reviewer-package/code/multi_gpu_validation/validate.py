"""Physical multi-GPU correctness, scaling, and recovery validation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dmstcn import DMSTCN, DMSTCNConfig

from common import (
    DistributedContext,
    all_reduce_mean,
    append_jsonl,
    assert_all_ranks_equal_text,
    enable_determinism,
    hardware_metadata,
    initialize_distributed,
    load_config,
    seed_everything,
    shutdown_distributed,
    state_digest,
    tensor_digest,
    utc_now,
    write_json,
)


def make_model(config: dict[str, Any], device: torch.device) -> DMSTCN:
    model_config = DMSTCNConfig(**config["model"])
    return DMSTCN(model_config).to(device)


def make_global_batch(
    config: dict[str, Any], step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model = config["model"]
    workload = config["workload"]
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + step)
    batch = workload["global_batch_size"]
    inputs = torch.randn(
        batch, workload["sequence_length"], model["input_dim"], generator=generator
    )
    subjects = torch.randint(0, model["num_subjects"], (batch,), generator=generator)
    labels = torch.randint(0, model["num_classes"], (batch,), generator=generator)
    return inputs.to(device), subjects.to(device), labels.to(device)


def local_shard(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], context: DistributedContext
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_batch = batch[0].shape[0]
    if global_batch % context.world_size:
        raise ValueError("global_batch_size must be divisible by WORLD_SIZE")
    width = global_batch // context.world_size
    start = context.rank * width
    return tuple(value[start : start + width] for value in batch)  # type: ignore[return-value]


def loss_for(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    inputs, subjects, labels = batch
    return nn.functional.cross_entropy(model(inputs, subjects).logits, labels)


def assert_tensor_close(
    actual: torch.Tensor, expected: torch.Tensor, config: dict[str, Any], label: str
) -> None:
    workload = config["workload"]
    try:
        torch.testing.assert_close(
            actual,
            expected,
            atol=float(workload["tolerance_atol"]),
            rtol=float(workload["tolerance_rtol"]),
        )
    except AssertionError as error:
        raise AssertionError(f"{label} failed: {error}") from error


def validate_correctness(
    context: DistributedContext, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    enable_determinism()
    seed_everything(config["seed"])
    reference = make_model(config, context.device)
    initial_state = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    assert_all_ranks_equal_text(state_digest(reference), "initial model state")

    ddp_model = DDP(reference, device_ids=[context.local_rank], output_device=context.local_rank)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=config["workload"]["learning_rate"])
    global_batch = make_global_batch(config, 0, context.device)
    shard = local_shard(global_batch, context)
    optimizer.zero_grad(set_to_none=True)
    distributed_loss = loss_for(ddp_model, shard)
    distributed_loss.backward()

    for name, parameter in ddp_model.module.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing gradient: {name}")
        assert_all_ranks_equal_text(tensor_digest(parameter.grad), f"gradient {name}")
    optimizer.step()
    assert_all_ranks_equal_text(state_digest(ddp_model.module), "post-step model state")

    if context.rank == 0:
        seed_everything(config["seed"])
        single = make_model(config, context.device)
        single.load_state_dict(initial_state)
        single_optimizer = torch.optim.Adam(single.parameters(), lr=config["workload"]["learning_rate"])
        single_optimizer.zero_grad(set_to_none=True)
        single_loss = loss_for(single, global_batch)
        single_loss.backward()
        single_optimizer.step()
        for name, distributed_value in ddp_model.module.state_dict().items():
            assert_tensor_close(distributed_value, single.state_dict()[name], config, name)
        single_loss_value = float(single_loss.detach())
    else:
        single_loss_value = None

    dist.barrier()
    report = {
        "validation": "correctness",
        "status": "pass",
        "timestamp_utc": utc_now(),
        "world_size": context.world_size,
        "distributed_loss_mean": float(all_reduce_mean(distributed_loss)),
        "single_loss": single_loss_value,
        "final_state_digest": state_digest(ddp_model.module),
    }
    return report


def benchmark_phase(
    context: DistributedContext,
    config: dict[str, Any],
    global_batch_size: int,
    label: str,
) -> dict[str, Any]:
    local_config = {**config, "workload": {**config["workload"], "global_batch_size": global_batch_size}}
    seed_everything(config["seed"])
    model = DDP(
        make_model(local_config, context.device),
        device_ids=[context.local_rank],
        output_device=context.local_rank,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["workload"]["learning_rate"])
    warmup = int(config["workload"]["warmup_steps"])
    measured = int(config["workload"]["measured_steps"])

    torch.cuda.reset_peak_memory_stats(context.device)
    durations = []
    dist.barrier()
    for step in range(warmup + measured):
        batch = local_shard(make_global_batch(local_config, step, context.device), context)
        torch.cuda.synchronize(context.device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_for(model, batch).backward()
        optimizer.step()
        torch.cuda.synchronize(context.device)
        elapsed = time.perf_counter() - started
        if step >= warmup:
            durations.append(elapsed)

    timing = torch.tensor([sum(durations), max(durations)], dtype=torch.float64, device=context.device)
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    total_seconds, maximum_step_seconds = timing.tolist()
    samples = global_batch_size * measured
    return {
        "phase": label,
        "world_size": context.world_size,
        "global_batch_size": global_batch_size,
        "local_batch_size": global_batch_size // context.world_size,
        "measured_steps": measured,
        "total_seconds": total_seconds,
        "maximum_step_seconds": maximum_step_seconds,
        "throughput_samples_per_second": samples / total_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(context.device),
    }


def validate_scaling(
    context: DistributedContext, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    base_batch = int(config["workload"]["global_batch_size"])
    if base_batch % context.world_size:
        raise ValueError("strong-scaling global batch must divide evenly across ranks")
    strong = benchmark_phase(context, config, base_batch, "strong")
    weak = benchmark_phase(context, config, base_batch * context.world_size, "weak")
    return {
        "validation": "scaling",
        "status": "pass",
        "timestamp_utc": utc_now(),
        "strong": strong,
        "weak": weak,
        "note": "Efficiency requires the matching one-GPU baseline from the same physical host.",
    }


def validate_checkpoint(
    context: DistributedContext, config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    enable_determinism()
    seed_everything(config["seed"])
    model = DDP(
        make_model(config, context.device),
        device_ids=[context.local_rank],
        output_device=context.local_rank,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["workload"]["learning_rate"])

    batch0 = local_shard(make_global_batch(config, 0, context.device), context)
    optimizer.zero_grad(set_to_none=True)
    loss_for(model, batch0).backward()
    optimizer.step()
    checkpoint_path = output_dir / f"checkpoint-w{context.world_size}.pt"
    if context.rank == 0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "step": 1},
            checkpoint_path,
        )
    dist.barrier()

    uninterrupted_batch = local_shard(make_global_batch(config, 1, context.device), context)
    optimizer.zero_grad(set_to_none=True)
    loss_for(model, uninterrupted_batch).backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.module.state_dict().items()}

    resumed = DDP(
        make_model(config, context.device),
        device_ids=[context.local_rank],
        output_device=context.local_rank,
    )
    resumed_optimizer = torch.optim.Adam(
        resumed.parameters(), lr=config["workload"]["learning_rate"]
    )
    checkpoint = torch.load(checkpoint_path, map_location=context.device, weights_only=True)
    resumed.module.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_optimizer.zero_grad(set_to_none=True)
    loss_for(resumed, uninterrupted_batch).backward()
    resumed_optimizer.step()
    for name, actual in resumed.module.state_dict().items():
        assert_tensor_close(actual, expected[name], config, f"resumed parameter {name}")
    assert_all_ranks_equal_text(state_digest(resumed.module), "resumed model state")
    return {
        "validation": "checkpoint",
        "status": "pass",
        "timestamp_utc": utc_now(),
        "checkpoint": str(checkpoint_path),
        "final_state_digest": state_digest(resumed.module),
    }


VALIDATIONS = {
    "correctness": validate_correctness,
    "scaling": validate_scaling,
    "checkpoint": validate_checkpoint,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", choices=VALIDATIONS)
    parser.add_argument("--config", default="multi_gpu_validation/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    minimum_world_size = 1 if args.validation == "scaling" else 2
    context = initialize_distributed(minimum_world_size=minimum_world_size)
    output_dir = Path(config["output_dir"])
    try:
        report = VALIDATIONS[args.validation](context, config, output_dir)
        report["campaign"] = config["campaign"]
        report["hardware"] = hardware_metadata(context)
        if context.rank == 0:
            destination = output_dir / f"{args.validation}-w{context.world_size}.json"
            write_json(destination, report)
            append_jsonl(output_dir / "registry.jsonl", report)
    except Exception as error:
        failure = {
            "campaign": config.get("campaign"),
            "validation": args.validation,
            "status": "fail",
            "timestamp_utc": utc_now(),
            "rank": context.rank,
            "world_size": context.world_size,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        append_jsonl(output_dir / "failures.jsonl", failure)
        raise
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
