#!/usr/bin/env python
from dsctm.experiments.preprocessing import run_preprocessing_robustness

result = run_preprocessing_robustness()
print(result, flush=True)
print("EXP13_PREPROCESSING_DONE", flush=True)
