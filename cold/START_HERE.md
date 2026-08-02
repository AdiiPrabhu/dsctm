# D-MSTCN Resubmission Package — Start Here

## Included files

```text
D_MSTCN_ONE_FILE_MASTER_PROMPT.md
reviews/
  D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx
  D_MSTCN_Rejected_Manuscript.pdf
```

The tracker already contains the detailed proposed resolutions and scientific justifications. Preserve it and require the agent to write future evidence-backed updates to a new workbook.

## Start the agent

Open the repository directory:

```bash
cd /media/adii/adissd/phd/dsctm-resubmission/code
```

Start either Claude Code or Codex from this directory. Give the agent this instruction:

```text
Read D_MSTCN_ONE_FILE_MASTER_PROMPT.md and execute it. The completed tracker is at reviews/D_MSTCN_IEEE_Access_Resubmission_Tracker_Completed.xlsx and the rejected manuscript is at reviews/D_MSTCN_Rejected_Manuscript.pdf. Preserve both originals and write updates to new output files. Begin Gate P now; do not merely summarize the prompt.
```

The first stage performs repository discovery, maps reviewer tasks, identifies missing inputs, estimates GPU-hours, and runs only cheap checks. The agent should ask for approval before costly or long experiments.

## Materials still needed for full execution

Keep these in the repository or provide their local paths when requested:

- actual D-MSTCN implementation and configurations;
- manuscript LaTeX/Word source, bibliography, and figures;
- original IEEE decision letter if not fully represented in the tracker;
- existing logs, checkpoints, predictions, and experiment configurations;
- authorized StudentLife and DAIC-WOZ dataset paths and official splits;
- DAIC test-evaluation access status;
- environment/container information;
- GPU model/count, topology, compute budget, and deadline.

Do not commit protected raw datasets, participant identifiers, passwords, or access tokens.

## If Claude Code reports a `bwrap` loopback error

This is a nested sandbox/runtime issue, not a prompt error. From the normal SSH shell, create or edit `.claude/settings.local.json`. Preserve existing settings and merge:

```json
{
  "sandbox": {
    "enabled": true,
    "enableWeakerNestedSandbox": true,
    "failIfUnavailable": false
  }
}
```

Restart Claude Code. If the trusted, already-isolated machine still cannot start the sandbox, set `"enabled": false` instead. Do not use a permissions-bypass flag.

For standalone Codex on an already-isolated trusted machine, the fallback is:

```bash
codex --sandbox danger-full-access
```

This disables Codex's additional OS sandbox, so use it only on a trusted personal machine/container.

