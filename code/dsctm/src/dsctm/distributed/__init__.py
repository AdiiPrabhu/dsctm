"""Distributed execution layer for D-MSTCN on PARAM Utkarsh.

Scope: full-model DistributedDataParallel. This is the CONTROL implementation — SAP
(Gate 8) and TCP (Gate 9) are separate execution modes built on top of it, and neither may
be claimed better than something that does not exist. See DECISIONS.md D-007.

Target hardware: PARAM Utkarsh `gpu` partition, 2 x NVIDIA V100 SXM2 (sm_70, 16 GB HBM2)
per node, 2 x Intel Xeon Gold 6248 (40 cores), Mellanox InfiniBand HDR, SLURM 20.11.8.
"""
from .checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    state_digest,
)
from .ddp import (
    EarlyStopCoordinator,
    EarlyStopDecision,
    assert_replicas_agree,
    build_grad_scaler,
    has_lazy_parameters,
    materialize_lazy_parameters,
    unwrap,
    wrap_ddp,
)
from .sap import (
    BRANCH_ORDER, CommStats, Placement, SAPModel, plan_placement,
    predicted_bytes_per_sample, recv_activation, replicate_gradients, sap_step,
    send_activation,
)
from .tcp_real import (
    BranchAction, ExecutionMode, MODE_DESCRIPTIONS, SyncReason, TCPState,
    TemporalConsistencyProtocol, describe_modes,
)
from .errors import (
    DistributedError,
    EvaluationCoverageError,
    PreflightFailure,
    RankFailure,
    all_ranks_ok,
    fail_together,
    hard_abort,
)
from .gather import (
    PredictionRecord,
    assert_exact_coverage,
    build_records,
    gather_and_validate,
    gather_predictions,
    records_to_arrays,
)
from .logging import (
    REQUIRED_RUN_FILES,
    RunLogger,
    audit_run_directory,
    rank_zero_only,
    write_json_atomic,
)
from .runtime import (
    BatchSemantics,
    DistContext,
    all_reduce_scalar,
    assert_agrees_across_ranks,
    assert_v100_ready,
    autocast_dtype,
    barrier,
    broadcast_object,
    cleanup,
    compute_capability,
    init_distributed,
    is_initialized,
    resolve_batch_semantics,
    seed_everything,
    select_backend,
)
from .sampler import (
    UnpaddedDistributedSampler,
    audit_sampler_partition,
    loader_kwargs_for_param,
    make_eval_sampler,
    make_train_sampler,
)

__all__ = [
    "BRANCH_ORDER", "BranchAction", "CommStats", "ExecutionMode", "MODE_DESCRIPTIONS",
    "Placement", "SAPModel", "SyncReason", "TCPState", "TemporalConsistencyProtocol",
    "describe_modes", "plan_placement", "predicted_bytes_per_sample", "recv_activation",
    "replicate_gradients", "sap_step", "send_activation",
    "BatchSemantics", "DistContext", "DistributedError", "EarlyStopCoordinator",
    "EarlyStopDecision", "EvaluationCoverageError", "PredictionRecord", "PreflightFailure",
    "REQUIRED_RUN_FILES", "RankFailure", "RunLogger", "UnpaddedDistributedSampler",
    "all_reduce_scalar", "all_ranks_ok", "assert_agrees_across_ranks",
    "assert_exact_coverage", "assert_replicas_agree", "assert_v100_ready",
    "audit_run_directory", "audit_sampler_partition", "autocast_dtype", "barrier",
    "broadcast_object", "build_grad_scaler", "build_records", "cleanup",
    "compute_capability", "fail_together", "find_latest_checkpoint", "gather_and_validate",
    "gather_predictions", "hard_abort", "has_lazy_parameters", "init_distributed",
    "is_initialized", "load_checkpoint", "loader_kwargs_for_param",
    "make_eval_sampler", "make_train_sampler", "materialize_lazy_parameters",
    "rank_zero_only", "records_to_arrays", "resolve_batch_semantics", "save_checkpoint",
    "seed_everything", "select_backend", "state_digest", "unwrap", "wrap_ddp",
    "write_json_atomic",
]
