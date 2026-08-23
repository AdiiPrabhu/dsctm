#!/usr/bin/env python
from dsctm.experiments.delay_task import run_delay_task

result = run_delay_task()
print(result["criteria_results"], flush=True)
print("EXP33_DELAY_DONE", flush=True)
