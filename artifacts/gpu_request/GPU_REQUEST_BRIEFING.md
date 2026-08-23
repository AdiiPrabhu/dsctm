# D-MSTCN — GPU Resource Requirement & Complexity Briefing

**Scholar:** Aditya Prabhu, CSE · **Supervisor:** Dr. Basavaraj N Hiremath
**Prepared:** 3 August 2026 · **Horizon:** 6 months
**For:** DSU Research Cell GPU allocation review

**Interactive deck:** https://claude.ai/code/artifact/9881519d-5cfc-4267-b425-b0685df726d6

> The deck is private by default. To let others open it, use the **Share** menu on the
> artifact page.

---

## THE ASK, IN ONE LINE

**4 × NVIDIA B200, configured as 2 nodes × 2 GPUs.**
≈ **600 B200-GPU-hours** across 6 months, plus **300 GB** scratch storage.

| | |
|---|---|
| **Minimum viable** | 4 GPUs (2 nodes × 2). Below this the 3-branch + aggregator topology cannot be instantiated at all. |
| **Preferred** | 8 GPUs (2 nodes × 4). Adds the N=8 scaling point the scalability claim needs. |
| **Scratch storage** | 300 GB — 226 GB staged + 40% working headroom |

**Hours are not the constraint — concurrency is.** Four GPUs held for six months is
17,280 GPU-hours of wall-clock. The programme needs about 600 of them, a duty cycle near
3%. The request is therefore **not for exclusive tenancy**: it is for 4 ranks across
2 nodes to be schedulable *together*, in roughly 24 windows over 6 months, each under
12 hours. A reservation or time-shared allocation serves this fully and leaves the
hardware free for other scholars between windows.

---

## ⚠ BLOCKING COMPATIBILITY FINDING

The project's pinned runtime is **`torch 2.1.2 + cu121`**. CUDA 12.1 does not emit code
for Blackwell (`sm_100`). **A B200 allocation requires the environment to be rebuilt** on:

- CUDA ≥ 12.8
- PyTorch ≥ 2.7 (cu128 wheels)
- NCCL ≥ 2.21
- Driver R570+

Roughly one day of work — but it must be budgeted *before* the first job, not discovered
at submission.

---

## 1. AREA OF RESEARCH

**Distributed deep learning for multi-timescale behavioural and physiological sensing.**
Three contributions, of which the third drives the compute request:

| Layer | Contribution |
|---|---|
| **Application domain** | Affective & mental-health computing — inferring depression and affect state from passive smartphone sensing, clinical interview audio, and EEG |
| **Modelling** | Multi-scale temporal CNN — three dilated causal branches resolving minutes, hours and days concurrently, fused by a learned cross-scale attention gate |
| **Systems** *(cost driver)* | Scale-aware partitioning — branch-parallel distribution with an asynchronous temporal coherence protocol |

### Why this needs an allocation rather than a workstation

The *science* half is small — a 1.37 M-parameter model over modest corpora; a single
modern GPU could grind through it. The *systems* half cannot run on one GPU at all: the
architecture assigns one temporal branch per rank plus a fusion aggregator, so the minimum
meaningful world size is four, and any claim about inter-node communication requires those
ranks to span at least two physical nodes over a real fabric.

### Current status, stated plainly

No experiment has yet executed on PARAM Utkarsh. All 20 installed V100s are unavailable —
10 drained to other projects, 6 held by the `nitk_res` reservation through 31 Dec 2026,
and the rest running other users' jobs. The campaign is **blocked on hardware, not on
code**: 316 tests pass and every job script is written and logic-verified. Live check:
`scripts/param/gpu_report.sh`.

---

## 2. METHODOLOGY BLUEPRINT

```
                    X ∈ ℝ^(B×T×F)          T timesteps, F sensor channels
                          │
                 Linear projection F → D    D = 128
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼─────┐     ┌─────▼────┐     ┌──────▼───┐
   │   SSB    │     │   MSB    │     │   LSB    │
   │  short   │     │  medium  │     │   long   │
   │ 1,2,4,8  │     │8,16,32,64│     │32,…,256  │
   │ RF 61    │     │ RF 481   │     │ RF 1921  │
   └────┬─────┘     └─────┬────┘     └──────┬───┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
          Cross-scale attention gate (softmax, temp √D)
                          │
              FiLM subject adapter (d_s = 8)
                          │
             Mean-pool → MLP head (D → 128 → C)
```

Each branch is **4 dilated residual blocks × 2 causal convolutions** = 8 convolutions;
24 in total across the three branches.

### Receptive field per branch — why three branches exist

| Branch | Receptive field | Span @ StudentLife 1-min sampling |
|---|---|---|
| SSB | 61 steps | 61 minutes |
| MSB | 481 steps | 8.0 hours |
| LSB | 1921 steps | 32.0 hours |

Derived and unit-tested from `RF = 1 + 2(K−1)·Σr` — **not** quoted from the manuscript.
The published figures (47 / 383 / 1535) assume one convolution per block and are incorrect
for this architecture.

### Distribution strategy under evaluation

| | Configuration |
|---|---|
| **Baseline** | Standard DDP — full model replicated per rank, gradient all-reduce |
| **Proposed** | Scale-aware partitioner — one branch per rank + aggregator rank (needs ≥ 4 ranks) |
| **Proposed +** | Asynchronous execution under a temporal coherence protocol, bounded staleness δ_max |
| **Measured against** | Strong & weak scaling at N = 1, 2, 4, 8 · 5 repetitions each |

---

## 3. DATA SIZE

| Corpus | Modality | Subjects | T × F | Raw on disk | Role |
|---|---|---:|---:|---:|---|
| **DAIC-WOZ** | Clinical interview audio + transcripts | 189 | 2000 × 88 | 86 GB | Primary depression benchmark |
| **E-DAIC** | Extended, pre-extracted features | 275 | 2000 × 88 | ~50 GB | Transfer / generalisation |
| **StudentLife** | Passive smartphone sensing | 48 | 60 × 8 | 1.5 GB | Longitudinal multi-scale |
| **SEED** | EEG, 62-channel | 15 | 2000 × 310 | ~50 GB | Cross-modality transfer *(optional)* |
| | | | | **187.5 GB** | |

| | |
|---|---|
| **Largest single tensor** | 44 MB — DAIC-WOZ batch of 32 at 2000×88, fp16 |
| **Total fits planned** | 372 — 294 scientific tasks + 78 ablation variants |
| **Model checkpoint** | 5.2 MB — 1.373 M parameters at fp32 |

The data does not stress any modern GPU. The **experiment count** does.

> **Open scoping decision, disclosed.** StudentLife windows are 60 timesteps, but the
> medium and long branches have receptive fields of 481 and 1921. On that corpus the two
> larger branches cannot observe their claimed timescales. Resolution — re-window,
> restrict the multi-scale claim to DAIC-WOZ, or report as a limitation — changes the
> ablation set and therefore the final task count by roughly ±20%.

---

## 4. COMPUTATION SIZE

Cost concentrates in 24 dilated causal convolutions (3 branches × 4 blocks × 2 convs),
each mapping D channels to D channels with kernel K:

```
F_fwd(T) ≈ 2 · L_conv · K · D² · T
         = 2 · 24 · 3 · 128² · T
         = 2.36 MFLOP · T
```

| Symbol | Value | Meaning |
|---|---|---|
| `L_conv` | 24 | total causal convolutions across all three branches |
| `K` | 3 | kernel width — dilation changes the receptive field, not the FLOP count |
| `D` | 128 | shared channel width |
| `T` | varies | sequence length |

Cost is **linear in T**, not quadratic. This is the architectural reason 2000-step
sequences are affordable here where self-attention at O(T²) would not be.

| Corpus | T | Forward / sample | Training step (3×, batched) | Peak activations |
|---|---:|---:|---:|---:|
| StudentLife | 60 | 0.142 GFLOP | 27.2 GFLOP @ B=64 | 0.03 GiB |
| DAIC-WOZ | 2000 | 4.767 GFLOP | 457.6 GFLOP @ B=32 | 0.48 GiB |
| DAIC-WOZ *(B200 batch-scaled)* | 2000 | 4.767 GFLOP | 3.66 TFLOP @ B=256 | 3.81 GiB |

### ★ The critical finding: this model is not FLOP-bound

Dividing step FLOPs by assumed step time gives achieved throughput, and hence model FLOP
utilisation against a V100's 125 TFLOPS fp16 peak:

| Corpus | Step FLOP | Step time | Achieved | MFU |
|---|---:|---:|---:|---:|
| StudentLife | 27.2 GFLOP | 8–25 ms | 1.09–3.40 TFLOPS | **0.9 – 2.7 %** |
| DAIC-WOZ | 457.6 GFLOP | 60–220 ms | 2.08–7.63 TFLOPS | **1.7 – 6.1 %** |

**Consequence for hardware selection.** At 1–6% utilisation the workload is limited by
memory bandwidth and kernel-launch occupancy, not arithmetic. A B200 delivers 18× the peak
FLOPS of a V100 but only **8.9× the bandwidth** — so the honest expected speedup is
**4–8×, not 18×**. Any request that scaled V100 hours by the FLOPS ratio would understate
the requirement by roughly a factor of three. This is why the figures below are bracketed
and why a calibration run precedes the full campaign.

---

## 5. THE GPU-HOUR FORMULA

Every quantity is either a property of the model, a property of the experiment plan, or a
stated assumption — nothing is a guess dressed as a measurement.

```
Step 1 · per-step cost
    F_step    = 3 · F_fwd(T) · B

Step 2 · steps per task
    S         = ceil(N_samples / B) · E_epochs · N_folds

Step 3 · seconds per task
    t_task    = (S · F_step) / (P_peak · MFU)

Step 4 · campaign total
    GPU-hours = η · Σ_families (n_tasks · t_task) / 3600

Step 5 · what to request
    Request   = GPU-hours · R_campaigns · I_iteration · (1 + c)
```

| Symbol | Meaning |
|---|---|
| `3 ·` | backward ≈ 2× forward; forward + backward ≈ 3× forward |
| `E_epochs` | epochs **actually reached under early stopping**, not the cap — the single largest source of error |
| `P_peak · MFU` | effective throughput. Use measured MFU; assume 2–6% for small-channel conv models, 35–50% for large transformers |
| `η = 1.25` | data loading, per-epoch evaluation, checkpoint I/O on shared storage |
| `R = 6` | campaign repeats over 6 months: reviewer rounds, corpus decision, windowing decision, transfer extension |
| `I = 2.0` | research iteration factor — failed, debug and exploratory runs |
| `c = 0.30` | contingency |

### Applied — V100 baseline, one campaign

| Family | Tasks | Corpus | GPU-h low | GPU-h high |
|---|---:|---|---:|---:|
| Tuning — StudentLife | 48 | StudentLife | 0.8 | 2.4 |
| Tuning — DAIC-WOZ | 48 | DAIC-WOZ | 0.1 | 0.3 |
| Confirmation — StudentLife | 60 | StudentLife | 1.0 | 3.0 |
| Confirmation — DAIC-WOZ | 60 | DAIC-WOZ | 0.1 | 0.4 |
| Ablations | 78 | StudentLife | 1.3 | 3.9 |
| **Systems** — DDP / partitioner / protocol sweep | — | both | **30.0** | **60.0** |
| **Campaign total** | **294** | | **33.2** | **70.1** |
| + 30% contingency | | | | **91 V100-GPU-h** |

Note the shape of this table: the 294 scientific fits cost **3–10 GPU-hours**. The systems
experiments cost **30–60**. Roughly **85% of the request exists to evaluate the
distributed protocol.**

### Converted — 6-month programme on B200

| Component | V100-GPU-h | Speedup | B200-GPU-h | Why that speedup |
|---|---:|---:|---:|---|
| Science (372 fits) | 3.2–10.1 | 6.0× | 0.5–1.7 | Bandwidth-bound; batch scales into 180 GB |
| Systems (DDP / SAP / protocol) | 30.0–60.0 | 1.7× | 17.6–35.3 | Communication-bound — compute speedup barely helps |
| **Per campaign** | **33.2–70.1** | | **18.2–37.0** | |
| × 6 campaigns × 2.0 iteration × 1.3 contingency | | | **≈ 577** | Round to **600 B200-GPU-hours** |

---

## 6. MEMORY — STAGING AND DOWNSTREAM

### GPU memory, per rank

| Component | DAIC-WOZ @ B=32 | @ B=256 (B200) | Notes |
|---|---:|---:|---|
| Parameters (fp32 master) | 5.2 MB | 5.2 MB | 1.373 M params |
| Gradients + Adam states | 15.7 MB | 15.7 MB | 3× parameter bytes |
| Activations (fp16, retained) | 0.48 GiB | 3.81 GiB | 24 conv layers × B × D × T |
| Workspace + fragmentation | ~0.5 GiB | ~1.2 GiB | cuDNN algorithm scratch |
| **Peak per rank** | **~1.0 GiB** | **~5.0 GiB** | Fits a 16 GB V100; uses 3% of a 180 GB B200 |

**This unlocks the real efficiency argument.** One training task needs under 5 GiB. A
180 GB B200 can therefore host **~28 concurrent tasks** under MPS, or 7 hard-isolated
tasks under MIG. The 372-fit scientific campaign — days of wall-clock on a 16 GB V100
running one job at a time — collapses to a few hours on a single B200 running the array
packed. **We are asking for 4 GPUs for the topology, not for the throughput.**

### Storage — staging in, results out

| Stage | Contents | Size |
|---|---|---:|
| **Staging** *(upstream, read-mostly)* | DAIC-WOZ raw — 189 sessions | 86.0 GB |
| | E-DAIC — 275 sessions, features + labels | 50.0 GB |
| | SEED EEG (optional scope) | 50.0 GB |
| | StudentLife | 1.5 GB |
| **Intermediate** | Extracted frame features, versioned per config | 20.0 GB |
| **Downstream** *(results, write-heavy)* | Checkpoints — 372 tasks × (best + last + optimiser) | 5.8 GB |
| | Predictions, metrics, learning curves, logs | 8.0 GB |
| | Systems traces — per-rank timing, protocol events | 5.0 GB |
| | Subtotal | **226.3 GB** |
| | + 40% working headroom | 90.5 GB |
| | **Scratch request** | **300 GB** |

### Host-side

- **System RAM:** 64 GB per training job — 8 dataloader workers × 8 GB. Feature extraction
  is the outlier: 48 cores, ~180 GB, CPU partition only.
- **Result egress:** under 20 GB total leaves the cluster. Aggregation and figure
  generation run locally; the cluster keeps raw artefacts.
- **Filesystem behaviour:** checkpoints written per epoch across a 372-task array generate
  many small writes — a known Lustre stress pattern, mitigated by writing to node-local
  scratch and syncing once at task exit.

---

## 7. NVIDIA EXECUTION ENVIRONMENT

| Layer | Current (V100 target) | Required for B200 |
|---|---|---|
| **Compute capability** | `sm_70` (Volta) | `sm_100` (Blackwell) |
| **CUDA toolkit** | `12.1` | **`≥ 12.8`** ⚠ breaking |
| **PyTorch** | `2.1.2+cu121` | **`≥ 2.7` with cu128 wheels** ⚠ breaking |
| **NCCL** | `2.18` | `≥ 2.21` |
| **Driver** | `R525+` | `R570+` |
| **Container** | bare conda env | `nvcr.io/nvidia/pytorch:25.xx-py3` *(preferred)* |
| **Numeric format** | fp16 + loss scaling | bf16 native — no loss scaling; fp8 available for the systems study |
| **Intra-node fabric** | PCIe | NVLink 5 / NVSwitch |
| **Inter-node fabric** | InfiniBand | InfiniBand NDR + GPUDirect RDMA *(required for systems claims)* |

### Scheduling and packing

| | |
|---|---|
| **Scheduler** | SLURM with `--gres=gpu:N`, array jobs throttled. Max walltime 12 h per task, well inside a 72 h limit |
| **Container runtime** | Enroot + Pyxis, or Apptainer/Singularity — NGC image avoids rebuilding CUDA on the cluster |
| **Packing** | CUDA MPS for the 372-fit array — ~28 tasks per B200. MIG (7 instances) where hard isolation is preferred |
| **Multi-node launch** | `torchrun` under `srun`, NCCL over IB. 2 nodes minimum for any inter-node claim |
| **Monitoring** | DCGM / `nvidia-smi dmon` for occupancy and utilisation — validates the MFU assumption against reality |

> **⚠ MIG cannot substitute for real nodes.** Partitioning one B200 into 7 instances
> yields 7 addressable ranks and is a legitimate way to *functionally* test the 4-rank
> topology. It is **not** valid for the scalability measurements: MIG instances share one
> physical device, so inter-rank communication never traverses the network the paper's
> protocol is designed to tolerate. Performance claims require ≥ 2 physical nodes.

---

## 8. HARDWARE OPTIONS COMPARED

| Device | BF16 peak | HBM | Bandwidth | Realistic speedup | Tasks / GPU | GPU-h for 6 months |
|---|---:|---:|---:|---:|---:|---:|
| V100-SXM2 *(PARAM today)* | 125 TF | 16 GB | 0.90 TB/s | 1.0× ref | 2 | ~2,800 |
| A100-SXM *(the earlier ask)* | 312 TF | 80 GB | 2.04 TB/s | 1.0–2.0× | 12 | ~1,700 |
| H100-SXM | 989 TF | 80 GB | 3.35 TB/s | 1.7–3.4× | 12 | ~1,000 |
| **B200-SXM** | **2,250 TF** | **180 GB** | **8.00 TB/s** | **4.0–8.0×** | **28** | **~600** |

Speedup is bracketed as 0.45–0.90 × the bandwidth ratio, because the workload runs at
1–6% FLOP utilisation. The lower bound assumes today's batch sizes; the upper assumes
batches scaled to fill the larger HBM. Bracketed deliberately — a single number here would
be false precision.

### Recommendation

- **4 × B200 as 2 nodes × 2** satisfies every requirement in this document and matches the
  topology already requested in A100 terms. This is the minimum that permits the systems
  contribution to be evaluated at all.
- **8 × B200 as 2 nodes × 4** additionally delivers the N=8 scaling point. The
  manuscript's original N=16 target is not schedulable on any allocation realistically
  available and should be restated.
- **A single B200 would be sufficient for all 372 scientific fits** — and would finish them
  in hours. It would leave the paper's central claim untested. If only one GPU can be
  granted, the honest outcome is a scoped-down paper, not a delayed one.
- **Access pattern matters more than quantity:** ~24 scheduled 4-GPU windows over 6 months,
  each under 12 hours, rather than continuous exclusive tenancy.

---

## 9. GENERIC TEMPLATE FOR ANY SCHOLAR

Fill the left column from your own model and plan; the arithmetic is identical regardless
of architecture.

| Input | How to obtain it | Example — D-MSTCN |
|---|---|---|
| **F_fwd** — forward FLOP per sample | Analytic count, or `thop` / `torch.profiler` on one batch | `2·L·K·D²·T = 4.77 GFLOP` |
| **B** — batch size | Largest that fits; measure, never assume | `32 (V100) → 256 (B200)` |
| **N, E, folds** | Experiment plan; E is epochs *under early stopping* | `163, 18, 1` |
| **n_tasks** | Configurations × seeds × folds, summed per family | `372` |
| **MFU** | Measure on one real task. 2–6% small conv/RNN; 35–50% large transformer | `1.7–6.1 %` |
| **η, R, I, c** | Overhead 1.25 · repeats · iteration 2.0 · contingency 0.30 | `1.25, 6, 2.0, 0.30` |
| **Peak memory** | params×16 bytes + activations(B·D·T·layers·2) + workspace | `1.0–5.0 GiB` |
| **Storage** | raw + features + checkpoints + logs, × 1.4 | `300 GB` |
| **Min world size** | Dictated by the *architecture*, not by throughput | `4 ranks / 2 nodes` |

### Three rules that prevent most bad estimates

1. **Never scale hours by peak-FLOPS ratio** — scale by *measured* throughput, or by
   bandwidth ratio when utilisation is low.
2. **Calibrate before committing** — run a handful of real tasks, measure, extrapolate,
   and only then submit the full array.
3. **Separate the *throughput* ask from the *topology* ask.** They are different numbers
   with different justifications, and conflating them is how requests get either rejected
   as excessive or granted at a size that cannot answer the question.

---

## PROVENANCE

Parameter counts, FLOP counts, receptive fields and task counts are **derived from the
source** and reproduce on demand. Per-step timings and MFU are **declared assumptions**,
bracketed low/high, pending a calibration run on the target hardware — no figure in this
document is presented as a measurement that is not one.

Reproduce with:

```bash
python scripts/param/estimate_compute.py     # GPU-hour brackets
python scripts/param/rf_report.py            # receptive fields
bash   scripts/param/gpu_report.sh           # live cluster availability
```

Repository: https://github.com/AdiiPrabhu/dsctm (branch `param-main`)
