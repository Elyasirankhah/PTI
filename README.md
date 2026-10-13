# Parallel-Then-Integrated (PTI)

Computational workflow for two representations of the same participant-response text, then mother-level association checks.

1. **Parallel quantitizing.** Ontology-guided Patient Voice (PV) sub-code densities, and unsupervised non-negative matrix factorization (NMF) themes, fit separately on the same paragraphs.
2. **Cross-stream alignment.** Qualitative. Not produced by these scripts.
3. **Overlap and external-anchor checks.** Feature overlap, leave-one-mother-out logistic regression with feature selection inside each training fold, and fold-selection frequencies.
4. **Integrated interpretation.** Qualitative joint display. Not produced by these scripts.

## Requirements

- Python 3.10+

```bash
pip install -r requirements.txt
```

`pv_metrics` is optional. Use it only when validating TAB-PO / PV Miner JSON against the Patient Voice ontology. Install it from the PV Miner documentation, or place `pv_metrics.py` on `PYTHONPATH`.

## Layout

```text
PTI_reproduction_bundle/
  README.md
  requirements.txt
  config.py
  data/
    units.jsonl                  # paragraph pool
    cohort_units.jsonl           # participant-turn subset (generated)
    participant_outcomes.csv     # mother-level outcome
    schema/pv_predictions_schema.json
  scripts/
  _lib/
  outputs/                       # created on run; not tracked
```

The files under `data/` are synthetic format examples (`DemoCorpus`, ids `P001`, `R001`). Replace them with the analysis corpus under the applicable data-use agreement.

Participant rows kept for modeling have `turn_role=participant` or a leading `P:` / `R:` speaker prefix. Turns with fewer than five alphabetic words are dropped.

## Run

From `PTI_reproduction_bundle/`:

### 1. Participant paragraphs

```bash
python scripts/prepare_cohort.py \
  --input data/units.jsonl \
  --outcomes data/participant_outcomes.csv \
  --output data/cohort_units.jsonl
```

### 2. NMF themes

```bash
python scripts/derive_nmf_themes.py --seed 42 --k 6
```

Writes `outputs/derived_themes_participant.csv` and the theme summaries. Topic count is chosen from `k = 2` to `k = 8` by mean Jaccard similarity of top-term matches across bootstrap refits. The outcome is not used.

### 3. PV labels

Produce paragraph-level PV labels with TAB-PO / PV Miner. Save them as `outputs/pv_predictions.json` using `data/schema/pv_predictions_schema.json`.

### 4. PV model

```bash
python scripts/run_pv_nested_model.py \
  --predictions outputs/pv_predictions.json \
  --input data/cohort_units.jsonl \
  --n-perm 500
```

Within each training fold the script ranks sub-code densities by absolute distance of the training AUC from 0.5 and keeps the top five. The script filename contains `nested`. The procedure is leave-one-mother-out with that selection inside the training fold.

### 5. Combined model

```bash
python scripts/run_integrated_model.py \
  --predictions outputs/pv_predictions.json \
  --input data/cohort_units.jsonl \
  --outcome depressed \
  --blocks 5,4 5,6 \
  --n-perm 500
```

`5,4` selects up to five PV sub-codes and up to four NMF themes. `5,6` raises the NMF cap to six. Non-informative columns are dropped before ranking.

Reduced block:

```bash
python scripts/run_integrated_model.py \
  --predictions outputs/pv_predictions.json \
  --input data/cohort_units.jsonl \
  --blocks 3,2 \
  --n-perm 500
```

### 6. Overlap and fold stability

```bash
python scripts/compute_feature_overlap.py \
  --table outputs/integrated_patient_table.csv

python scripts/summarize_fold_stability.py \
  --predictions outputs/pv_predictions.json \
  --input data/cohort_units.jsonl \
  --k-pv 5 --k-ash 4
```

`--k-ash` is the NMF block cap (up to four themes).

### 7. Classifier comparison (optional)

```bash
python scripts/run_classifier_sensitivity.py \
  --predictions outputs/pv_predictions.json \
  --input data/cohort_units.jsonl \
  --n-perm 500
```

## Outputs

| Result | File |
| --- | --- |
| NMF theme summary | `outputs/derived_themes_summary.txt` |
| PV, NMF, and combined AUC | `outputs/integrated_model.txt` |
| Mean absolute correlation between streams | `outputs/feature_overlap.json` |
| PV sub-codes selected inside each fold | `outputs/pv_fold_stability.txt` |
| Classifier comparison | `outputs/classifier_sensitivity.txt` |

## Notes

- Supervised models are fit at the mother level. Paragraphs are not treated as independent participants.
- Each permutation shuffles the mother-level outcome and refits the same cross-validation procedure. The commands above use 500 permutations.
- NMF is fit on the full paragraph pool before cross-validation, so NMF and combined estimates are transductive.
- Alignment of codes to themes, and the joint display, are done outside this repository.
