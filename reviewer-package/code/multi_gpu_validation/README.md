# Multi-GPU validation

This folder contains the rental-machine validation harness. It is designed for
one host with 2, 4, or 8 CUDA GPUs and uses PyTorch DistributedDataParallel
(DDP/NCCL). It does not emulate physical GPUs and must not be used to label
single-GPU simulation as multi-GPU evidence.

## Covered validations

- distributed initialization and rank/device mapping;
- identical initial model state on every rank;
- DDP gradient and post-step parameter agreement;
- single-process versus DDP numerical equivalence;
- strong and weak scaling throughput, latency, memory, and efficiency;
- checkpoint save, reload, and resumed-step equivalence;
- clean JSON/JSONL evidence with hardware and environment metadata.

The harness uses deterministic synthetic tensors so it can validate the system
before protected datasets are copied or mounted. Dataset-quality experiments
will use the same DDP primitives after the data pipelines are finalized.

## Rental-machine commands

Install the repository into an isolated environment, then run preflight:

```bash
python -m pip install -e .
python multi_gpu_validation/preflight.py --output artifacts/multigpu/preflight.json
```

Correctness on 2, 4, and 8 GPUs:

```bash
torchrun --standalone --nproc-per-node=2 multi_gpu_validation/validate.py correctness --config multi_gpu_validation/config.yaml
torchrun --standalone --nproc-per-node=4 multi_gpu_validation/validate.py correctness --config multi_gpu_validation/config.yaml
torchrun --standalone --nproc-per-node=8 multi_gpu_validation/validate.py correctness --config multi_gpu_validation/config.yaml
```

Strong/weak scaling and checkpoint validation:

```bash
torchrun --standalone --nproc-per-node=1 multi_gpu_validation/validate.py scaling --config multi_gpu_validation/config.yaml
torchrun --standalone --nproc-per-node=8 multi_gpu_validation/validate.py scaling --config multi_gpu_validation/config.yaml
torchrun --standalone --nproc-per-node=8 multi_gpu_validation/validate.py checkpoint --config multi_gpu_validation/config.yaml
```

Run every supported GPU count sequentially:

```bash
bash multi_gpu_validation/run_matrix.sh 2 4 8
```

`run_matrix.sh` automatically records the same-host one-GPU baseline first and
creates `artifacts/multigpu/summary.json` with speedup and efficiency.

Do not compare scaling results across different machine types. Keep the
generated preflight file, reports, stdout/stderr, and exact Git commit together.

