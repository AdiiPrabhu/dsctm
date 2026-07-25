# D-MSTCN — Reviewer Reproducibility Package

This anonymized package accompanies the D-MSTCN manuscript resubmission. It
contains the model implementation, the experiment/validation harness, and the
label/split artifacts needed to reproduce the reported experiments.

It deliberately does **not** contain the raw interview corpora, which are
access-restricted (see *Data availability* below). Reviewers who wish to run the
full pipeline obtain those datasets directly from their official custodians
under the same license the authors used.

## Contents

```
code/   D-MSTCN model, tests, and the multi-GPU validation harness
data/   PHQ-8 labels, train/dev/test splits, and the metadata mapping
```

## Reproducing the code-level results

```bash
cd code
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest        # shape, causality, attention-normalization, receptive-field, input-validation tests
```

The dataset pipeline expects the raw corpora to be present locally and its path
supplied via configuration/environment variables; see `code/README.md`.

## Data availability

The experiments use two access-restricted datasets. Neither is redistributed
here; both must be requested from their official sources:

- **DAIC-WOZ / E-DAIC** (Distress Analysis Interview Corpus) — distributed by
  USC Institute for Creative Technologies under an End User License Agreement
  that prohibits redistribution. Request access at
  <https://dcapswoz.ict.usc.edu/>.
- **StudentLife** — Dartmouth College mobile-sensing dataset, publicly
  available with citation at <https://studentlife.cs.dartmouth.edu/>.

The `data/` folder includes only the small label and split artifacts required to
reproduce the exact partitioning used in the paper. See `data/PROVENANCE.md`.

## Citation

Citation details are withheld for anonymous review and will be provided in the
camera-ready version.
