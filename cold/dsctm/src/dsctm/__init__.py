"""D-MSTCN resubmission experimentation harness.

Package layout:
  repro       - seeding, determinism modes, environment capture
  config      - YAML config load / deep-merge / resolve / hash
  registry    - immutable run registry (run identity + per-run directories)
  data        - data contract, synthetic generators, leakage-safe splits, loaders
  models      - D-MSTCN and baselines
  train       - training loop (scientific & systems modes)
  eval        - metrics + statistical analysis plan
  experiments - Phase 0-6 experiment drivers
"""
__version__ = "0.0.1"
