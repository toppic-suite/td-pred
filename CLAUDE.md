# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TD-Pred predicts top-down MS/MS fragment spectra from proteoform sequences with a CNN + transformer
encoder/decoder (PyTorch). There is no package, build system, test suite, or linter — it is a set of
standalone scripts under `src/`. Runtime dependencies (no requirements file exists): `torch`, `numpy`,
`h5py`, `pandas`, `torchinfo`.

## Commands

```bash
# 1. Annotated msalign -> HDF5 training data (writes all three target types)
python3 src/msalign/msalign_anno_to_hdf5.py --msalign spectra_anno_ms2.msalign --out spectra.hdf [--max_length 200]

# 2. Train (single GPU / CPU). Checkpoints: <out>_<epoch> per epoch and <out>_final; also writes similarity_<epoch>.tsv to cwd
python3 src/model/train_td_pred.py --train spectra_train.hdf --validate spectra_val.hdf --out td_pred_model.pth \
    [--target pep_bond|b_y|charge] [--batch 32] [--epochs 50] [--lr 3e-4] [--load_model ckpt] [--no_amp] [--deterministic]

# 2b. Multi-GPU DDP (LOCAL_RANK env var triggers DDP; without torchrun, >1 GPU falls back to DataParallel)
torchrun --nproc_per_node=N src/model/train_td_pred.py --train ... --validate ...

# 3. Predict msalign spectra from a TSV of sequences + MS metadata
python3 src/model/td_pred.py --input CID_HCD_VAL_1.2.1.tsv --model td_pred_backbone_model_1.2.1.pth --output spectra_pred.msalign

# Helper: extract per-scan metadata from an msalign into the TSV format td_pred.py consumes
python3 src/msalign/convert_msalign_to_tsv.py in.msalign out.tsv
```

Sample data and pretrained models are linked in README.md (Tulane Box).

### How imports resolve (read before importing these modules from anywhere new)

`src/` is not an installed package and has no `__init__.py` files. The modules in `src/model/` import each
other by bare top-level name (`import model_data`, `import td_pred_model`, `import hdf5_generator`), and
`src/msalign/msalign_anno_to_hdf5.py` does `import msalign_anno_generator`. Those names only resolve if the
module's own directory is on `sys.path`.

When you run `python3 path/to/script.py`, Python puts `path/to/` at `sys.path[0]`, which is what makes the
bare imports work. Any other entry point puts the *current directory* (or nothing) there instead, so:

| Invocation (from repo root)                    | Result                                       |
|------------------------------------------------|----------------------------------------------|
| `python3 src/model/train_td_pred.py ...`       | works                                        |
| `python3 -m src.model.train_td_pred ...`       | `ModuleNotFoundError: No module named 'model_data'` |
| `import src.model.td_pred_model` (test/notebook) | same error, raised from inside `td_pred_model.py` |
| `PYTHONPATH=src/model python3 -c "import td_pred_model"` | works                              |
| `cd src/model && python3 -c "import td_pred_model"`      | works                              |

`src/msalign/msalign_anno_generator.py` is the one exception: it inserts `src/` (its parent) into `sys.path`
at import time and then does `import model.model_data`, treating `src/model` as an implicit namespace package.
That makes it importable from `src/msalign/` but ties it to the on-disk layout.

Practical rules:
- Launch the CLIs as files, exactly as in the commands above. Do not convert them to `python -m` calls.
- To use these modules from a test, notebook, or new script, either place the new file in the same directory,
  or add `src/model` (and `src/msalign` if needed) to `sys.path`/`PYTHONPATH` first.
- If you add a new module under `src/model/`, import it by bare name to match the existing convention, and be
  aware that `sys.path.insert` in `msalign_anno_generator.py` resolves it as `model.<name>` instead. Both
  spellings load the file as separate module objects, so avoid module-level state that must be shared.

## Architecture

**Data pipeline:** annotated `.msalign` → `MsalignAnnoBatchGenerator` (parses `BEGIN IONS`/`END IONS`
blocks; peak lines must have exactly 10 whitespace-separated columns, ion type is the first char of column 5,
position is column 6) → HDF5 → `Hdf5BatchGenerator` (a `Dataset` that lazily opens the file per DataLoader
worker) → `TransformerSeq2Seq`.

**`src/model/model_data.py` is the single source of truth for all tensor layouts.** Both the HDF5 converter
and the prediction script call it, so the model, the stored data, and inference must agree with it:

- Sequence encoding: length `max_length + 2` (start token `@`, end token `[`), dimension 27 =
  24 one-hot chars (22 amino acids + 2 tokens) + residue mass/200 + proteoform length/200 + position in [-1, 1].
  Note the mass table has lowercase `m` (oxidized Met) but the one-hot alphabet does not.
- Meta vector: 46 = 30 precursor-charge one-hot + 9 instrument one-hot + 5 activation one-hot
  (unknown/cid/etd/hcd/ethcd) + proteoform mass/10000 + NCE/100 (0.25 when NCE missing).
- Mask: 1.0 over residue positions only (excludes start/end tokens).
- Charge mask: 60 = 30 N-terminal + 30 C-terminal slots, 1.0 for fragment charges ≤ precursor charge.
- Max fragment charge is 30 (`get_max_fragment_charge`).

**Targets (all three are stored in every HDF5; `--target` picks one at train time):** each is indexed by
peptide bond from the N-terminus (`max_length - 1 = 199` bonds), max-normalized to [0, 1].
`pep_bond` → output_dim 1; `b_y` → 2 (N-term, C-term); `charge` → 60 (N-term ions charge 1..30, then
C-term ions charge 1..30). b/c ions map to index `pos-1`; y/z ions map to `seq_len - pos - 1`.

**Model (`src/model/td_pred_model.py`):** 8 parallel `Conv1d` branches (kernels 1–8, 8 channels each) over
the sequence encoding are concatenated with the raw encoding and a repeated 8-dim meta embedding, projected
to `d_model=256`, run through a 6-layer encoder and a 6-layer *non-autoregressive* decoder whose input is a
learned `tgt_embedding` parameter. Output is sigmoid, multiplied by the sequence mask (and charge mask when
`output_dim > 2`), then sliced `[1 : output_len+1]` to drop the start token and flattened to
`(batch, output_len * output_dim)`. Loss is MSE on that flat vector; the reported metric is cosine similarity.

**Prediction (`src/model/td_pred.py`):** hardcodes `output_dim=60`, so it only works with a `charge`-target
model. It reshapes output to `(199, 60)`, keeps intensities > 0.01, computes b/y (or c/z· for ETD, with
fixed neutral shifts) masses from `get_mono_mass_list`, and appends `BEGIN IONS` blocks to the output file.
Input TSV columns: `DATASET_ID, MZML_FILE_NAME, MSALIGN_FILE_NAME, DATABASE_SEQUENCE, PRECURSOR_CHARGE,
INSTRUMENT, ACTIVATION, COLLISION_ENERGY`.

**Checkpoints** are dicts with `model_state_dict` (always the unwrapped module, DDP/DataParallel stripped)
plus loss/similarity histories. Loading a checkpoint into a DataParallel-wrapped model (as `td_pred.py` does
on a multi-GPU machine) will therefore fail on the `module.` prefix; run prediction on a single GPU or load
into the bare model first.
