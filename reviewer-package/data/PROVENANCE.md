# Data provenance and sharing note

These files are the **labels and split definitions** used in the experiments,
not raw interview data:

- `train_split.csv`, `dev_split.csv`, `test_split.csv` — participant-ID lists
  defining the exact train/dev/test partition.
- `Detailed_PHQ8_Labels.csv`, `detailed_lables.csv` — PHQ-8 depression labels
  keyed by de-identified participant ID.
- `metadata_mapped.csv` — mapping/metadata used by the pipeline.

**Origin:** these artifacts derive from the USC E-DAIC distribution. They contain
no audio, video, or transcripts — only de-identified participant IDs and their
associated PHQ-8 scores/splits, which reproducibility repositories in this area
commonly share.

**Before publishing:** confirm that sharing these label/split files is permitted
under the E-DAIC End User License Agreement you signed and your institution's
data-sharing policy. If you prefer maximum caution, delete this `data/` folder
and instead point reviewers to the official split definitions — the `code/` and
top-level README still stand on their own.
