"""Campaign orchestration: deterministic plan, run-directory contract, admission audit."""
from .plan import (
    FAMILIES,
    MODELS,
    Task,
    ablation_tasks,
    build_plan,
    confirmation_tasks,
    get_task,
    plan_digest,
    sbatch_array_spec,
    summarize,
    tuning_tasks,
)
from .audit import AuditResult, aggregate_family, audit_family
from .rundir import REQUIRED_FILES, RunDirectory

__all__ = ["AuditResult", "FAMILIES", "MODELS", "REQUIRED_FILES", "RunDirectory", "Task",
           "aggregate_family", "audit_family", "ablation_tasks", "build_plan", "confirmation_tasks", "get_task",
           "plan_digest", "sbatch_array_spec", "summarize", "tuning_tasks"]
