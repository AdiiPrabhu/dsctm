# Draft reply — Dr. Basavaraj N Hiremath, "Gpu run", 3 Aug 2026

Subject: Re: Gpu run — B200 sizing, complexity model and resource template

Dear Sir,

Please find below the consolidated answers. I have also prepared a full briefing deck
covering every point you raised — area of research, methodology blueprint, data size,
computation size, the GPU-hour formula, memory for staging and downstream, and the
required NVIDIA execution environment:

**<PASTE ARTIFACT LINK HERE>**

---

## 1. If it is NVIDIA B200 — how much do we need

**4 × B200, configured as 2 nodes × 2 GPUs. Approximately 600 B200-GPU-hours over
6 months, plus 300 GB of scratch storage.**

The important nuance is that **hours are not the constraint — concurrency is.**
Four GPUs held for six months is 17,280 GPU-hours of wall-clock; we need about 600 of
them, a duty cycle near 3%. So the request is not for exclusive tenancy. It is for
**4 ranks across 2 nodes to be schedulable together, in roughly 24 windows across
6 months**, each under 12 hours. A reservation or time-shared allocation serves this
fully and leaves the hardware free for other scholars in between.

The reason it must be 2 nodes rather than 1 is architectural, not throughput-driven.
The contribution under review is a *distributed training protocol*: the model assigns one
temporal branch per rank plus a fusion aggregator, so the minimum meaningful world size
is 4, and any claim about inter-node communication requires those ranks to span at least
two physical machines over a real fabric. On a single node the claim cannot be tested at
all.

For comparison against the earlier A100 request:

| Device | Realistic speedup | Tasks per GPU | GPU-h for 6 months |
|---|---|---|---|
| V100 (PARAM today) | 1.0× reference | 2 | ~2,800 |
| A100 (earlier ask) | 1.0–2.0× | 12 | ~1,700 |
| H100 | 1.7–3.4× | 12 | ~1,000 |
| **B200** | **4.0–8.0×** | **28** | **~600** |

## 2. One finding that must be flagged before any B200 allocation

Our pinned runtime is `torch 2.1.2 + cu121`. **CUDA 12.1 does not emit code for Blackwell
(`sm_100`)** — the current environment will not produce a binary that runs on a B200.
A B200 allocation requires rebuilding on **CUDA ≥ 12.8, PyTorch ≥ 2.7, NCCL ≥ 2.21,
driver R570+**, ideally via the NGC PyTorch container rather than a hand-built conda
environment. This is about a day of work, but it needs to be budgeted before the first
job rather than discovered at submission.

## 3. Area of research

Distributed deep learning for multi-timescale behavioural and physiological sensing —
inferring depression and affect state from passive smartphone sensing, clinical interview
audio, and EEG. Three contributions: an application-domain result in affective computing,
a multi-scale temporal CNN architecture, and — the part that drives the compute request —
a scale-aware partitioning scheme with an asynchronous temporal coherence protocol.

## 4. Computational complexity

Cost concentrates in 24 dilated causal convolutions (3 branches × 4 blocks × 2 convs):

    F_fwd(T) ≈ 2 · L_conv · K · D² · T = 2 · 24 · 3 · 128² · T = 2.36 MFLOP · T

Linear in sequence length, not quadratic — that is the architectural reason 2000-step
sequences are affordable here where self-attention at O(T²) would not be.

- StudentLife (T=60): 0.142 GFLOP forward per sample
- DAIC-WOZ (T=2000): 4.767 GFLOP forward per sample; 457.6 GFLOP per training step at B=32

**The critical finding: this model is not FLOP-bound.** Measured against a V100's
125 TFLOPS peak, it runs at **1–6% FLOP utilisation** — it is limited by memory bandwidth
and kernel-launch occupancy, not arithmetic. A B200 has 18× the peak FLOPS of a V100 but
only 8.9× the bandwidth, so the honest expected speedup is **4–8×, not 18×**. Any estimate
that scaled V100 hours by the FLOPS ratio would understate the requirement by roughly
threefold. This is why our figures are bracketed and why a calibration run precedes the
full campaign.

## 5. The formula used to compute GPU hours

    F_step    = 3 · F_fwd(T) · B                                  (backward ≈ 2× forward)
    S         = ceil(N_samples / B) · E_epochs · N_folds
    t_task    = (S · F_step) / (P_peak · MFU)
    GPU-hours = η · Σ_families (n_tasks · t_task) / 3600
    Request   = GPU-hours · R_campaigns · I_iteration · (1 + c)

with η = 1.25 (loading, per-epoch eval, checkpoint I/O), R = 6 campaign repeats over
6 months, I = 2.0 research-iteration factor for failed and exploratory runs, c = 0.30
contingency. E_epochs must be the epochs *actually reached under early stopping*, not the
cap — that is the single largest source of error in this kind of estimate.

Applied to our frozen plan (294 scientific tasks + 78 ablation variants = 372 fits), on
V100 the campaign costs 33–70 GPU-hours, or 91 with contingency. Note the shape: the 372
scientific fits cost 3–10 GPU-hours; the systems experiments cost 30–60. **Roughly 85% of
the request exists to evaluate the distributed protocol.**

## 6. Data size

| Corpus | Subjects | T × F | Raw on disk |
|---|---|---|---|
| DAIC-WOZ | 189 | 2000 × 88 | 86 GB |
| E-DAIC | 275 | 2000 × 88 | ~50 GB |
| StudentLife | 48 | 60 × 8 | 1.5 GB |
| SEED (optional) | 15 | 2000 × 310 | ~50 GB |

## 7. Memory — staging and downstream

**GPU:** peak ~1.0 GiB per rank at current batch sizes, ~5.0 GiB with batches scaled for
a B200. One task therefore uses about 3% of a 180 GB B200 — which means a single B200 can
host **~28 concurrent tasks under MPS**, or 7 hard-isolated under MIG. The 372-fit
campaign, which is days of wall-clock on a 16 GB V100 running one job at a time, collapses
to a few hours on one B200 running the array packed. We are asking for 4 GPUs for the
**topology**, not for the throughput.

**Storage:** 187.5 GB raw corpora + 20 GB extracted features + 5.8 GB checkpoints
(372 × best/last/optimiser at 5.2 MB each) + 13 GB logs, predictions and systems traces
= 226 GB, plus 40% working headroom → **300 GB scratch**.

**Host:** 64 GB RAM per training job (8 dataloader workers × 8 GB). Feature extraction is
the outlier — 48 cores, ~180 GB, CPU partition only. Under 20 GB of results leave the
cluster; aggregation and figures are produced locally.

## 8. Execution environment (NVIDIA flavour)

CUDA ≥ 12.8 · PyTorch ≥ 2.7 (cu128) · NCCL ≥ 2.21 · driver R570+ · NGC container
`nvcr.io/nvidia/pytorch:25.xx-py3` under Enroot/Pyxis or Apptainer · bf16 native (no loss
scaling) · SLURM with `--gres=gpu:N` and throttled array jobs · `torchrun` under `srun`
for multi-node · NVLink 5 intra-node, **InfiniBand NDR with GPUDirect RDMA inter-node** ·
DCGM for occupancy monitoring.

One caveat worth recording: **MIG cannot substitute for real nodes.** Partitioning one
B200 into 7 instances gives 7 addressable ranks and is a legitimate way to *functionally*
test the 4-rank topology, but it is not valid for scalability measurements — the instances
share one physical device, so inter-rank traffic never traverses the network the protocol
is designed to tolerate.

## 9. Generic template for other scholars

Section 09 of the deck is a fill-in-the-blank table any scholar can use: forward FLOPs per
sample, batch size, samples/epochs/folds, task count, MFU, overhead and contingency
factors, peak memory, storage, and minimum world size — with guidance on how to obtain
each. Three rules prevent most bad estimates: never scale hours by peak-FLOPS ratio;
calibrate on a handful of real tasks before committing to a full array; and keep the
*throughput* ask separate from the *topology* ask, because they are different numbers with
different justifications.

## 10. Current PARAM status

No experiment has yet executed on PARAM Utkarsh. All 20 installed V100s are unavailable —
10 drained to other projects, 6 held by the `nitk_res` reservation through 31 Dec 2026,
and the remainder running other users' jobs. The campaign is blocked on hardware, not on
code: 316 tests pass and every job script is written and logic-verified. The live check is
`scripts/param/gpu_report.sh` in the repository.

---

Happy to present the deck in person or to adjust the scope if the allocation available is
smaller than the above — in that case the honest outcome is a scoped-down paper rather
than a delayed one, and I would rather make that call deliberately with you.

Regards,
Aditya Prabhu
