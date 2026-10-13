"""Paths for PTI demonstration pipeline."""

from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PIPELINE_ROOT / "data"
OUTPUT_DIR = PIPELINE_ROOT / "outputs"
CODEBOOK_DIR = PIPELINE_ROOT / "codebooks"

UNITS_JSONL = DATA_DIR / "units.jsonl"
COHORT_UNITS_JSONL = DATA_DIR / "cohort_units.jsonl"
PARTICIPANT_OUTCOMES = DATA_DIR / "participant_outcomes.csv"
CODING_SHEET = DATA_DIR / "coding_sheet.csv"
PV_PREDICTIONS_JSON = OUTPUT_DIR / "pv_predictions.json"

DEMO_SUBCORPUS = "DemoCorpus"
