# D-MSTCN experiments

Correctness-first implementation and experiment harness for the D-MSTCN
resubmission. Development is currently on branch `experimentation2`.

## Environment

Both Claude and Codex should use the shared environment:

```bash
source /home/adii/venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Protected datasets must remain outside this repository. Dataset locations will
be supplied through local configuration or environment variables after access
and split provenance have been confirmed.

## Current scope

The package contains the single-device D-MSTCN model described in the rejected
manuscript: shared input projection, three residual causal TCN branches, CSAG
fusion, subject-conditioned FiLM, and a classification head. Tests cover shape,
causality, attention normalization, receptive field calculation, and input
validation. Distributed TCP/SAP behavior and dataset pipelines are intentionally
not claimed until their protocols and data are available.

