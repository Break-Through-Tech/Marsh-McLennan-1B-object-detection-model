"""Build the cleaned NEISS 2025 master dataset: one row per incident, no rows dropped.

Cleaning decisions and why:

- Keep every incident. Problems are flagged in new columns instead of deleted,
  so each later step can decide what to exclude and the choice stays visible.
- Read only populated rows (rows with a case number). Rows are never dropped
  for having blanks, because optional fields such as Other_Race and
  Other_Diagnosis_2 are legitimately empty.
- narrative_raw is the untouched original, kept for traceability.
- narrative_clean only strips leading "^"/whitespace and collapses repeated
  whitespace. Punctuation, numbers, negations, abbreviations, the age/sex
  prefix and the "DX:" text all stay; spelling is not corrected and
  abbreviations are not expanded, so the model sees what real descriptions
  look like.
- narrative_key is lowercase narrative_clean, used only for duplicate
  detection and dictionary matching, never as model input.
- Product codes are the targets and the narrative is the input, so product
  names are kept in their own columns and never merged into the text.
  product_codes/product_names list every product on the incident, not only
  Product_1. Code 0 is a "no product" placeholder and is left out; it is
  never a category. Every nonzero code must exist in the NEISS_FMT lookup.
- incident_text is narrative_clean without the leading age/sex prefix
  ("47YOF", "23YO UNK GENDER") and without everything from the first "DX"
  marker onward, so extractors see only how the injury happened and do not
  pick up diagnosis words as objects. Case, abbreviations and spelling are
  untouched. incident_text_empty marks results too short to describe an
  incident; those rows are kept but should be skipped when sampling.
- possible_truncation marks narratives of exactly 400 characters, the field
  limit. It is a hint, not proof that text was cut off.
- has_redaction_marker marks narratives containing "***".
- duplicate_group is shared by records with the same narrative_key. Repeats
  are kept; duplicate_labels_differ marks groups whose product codes disagree,
  since those are ambiguous training examples.
- split is assigned per duplicate_group, so identical narratives can never
  appear in both training and evaluation data.
- age_decoded is age in years. Codes 201-223 mean 1-23 months for children
  under 2 (per the AGELTTWO lookup) and are converted, not treated as invalid.
  Code 0 is unknown age and becomes missing.
- Location, Body_Part and Diagnosis get *_label columns from NEISS_FMT.
  Location 0 stays as its own "UNK" (unknown) value rather than missing.
- All other columns (dates, demographics, survey fields) are unchanged.

product_codes and product_names are stored as JSON lists in the CSV; read them
back with json.loads.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "neiss2025.xlsx"
CLEAN_PATH = DATA_DIR / "processed" / "neiss_2025_clean.csv"

PRODUCT_COLUMNS = ["Product_1", "Product_2", "Product_3"]
NARRATIVE_FIELD_LIMIT = 400
REDACTION_MARKER = "***"
INCIDENT_TEXT_MIN_LENGTH = 8
SPLIT_SEED = 42
# Cumulative upper bounds: 80% train, 10% val, 10% test.
SPLIT_BOUNDS = {"train": 0.8, "val": 0.9, "test": 1.0}

# Sex written after the age: glued ("YOF"), spaced ("YO M"), unknown or redacted.
_SEX = (
	r"(?:[MF](?:[MF]\b)?|\s+(?:FEMALE|MALE|[MF])\b"
	r"|\s+UNK(?:NOWN)?\s+(?:SEX|GENDER)\b|\s+\*\*\*(?:\s+(?:FEMALE|MALE|[MF])\b)?)?"
)
# "47YOF", "64 YOM", "16MOM", "6WOF", "27Y0M", "21YF", "9 DAY OLD MALE",
# "23YO UNK GENDER", "20YO *** F", "UNK AGE/SEX", plus trailing punctuation.
AGE_SEX_PREFIX = re.compile(
	r"^(?:\d+[\s/,;']*"
	r"(?:(?:Y[O0IPY]{0,2}|Y/O|M/?[O0]|W/?O|WKO|D/?O"
	r"|[-\s]*(?:YEAR|YR|MONTH|WEEK|WK|DAY)S?[-\s]*OLD)" + _SEX
	+ r"|WK[MF]\b|WEEKO[MF]\b)"
	r"|UNK(?:NOWN)?\s+AGE(?:\s*/\s*SEX\b|\s*/?\s*(?:FEMALE|MALE|[MF])\b)?)"
	r"[\s,.:;\-]*"
)
# "DX:", "DX ", "DX-", "DX;", and DX glued to a neighbouring word ("HIPDX", "DXHEAD").
DIAGNOSIS_START = re.compile(r"\bDX|DX\b")


def read_workbook(raw_path: Path) -> tuple:
	"""Return (incidents, formats), keeping only populated rows."""
	with pd.ExcelFile(raw_path) as workbook:
		incidents = pd.read_excel(workbook, sheet_name="NEISS_2025")
		formats = pd.read_excel(workbook, sheet_name="NEISS_FMT", dtype=str)

	incidents = incidents[incidents["CPSC_Case_Number"].notna()].reset_index(drop=True)
	formats.columns = ["format", "start", "end", "label"]
	formats = formats[formats["format"].notna()].reset_index(drop=True)
	for column in formats.columns:
		formats[column] = formats[column].str.strip()
	return incidents, formats


def code_labels(formats: pd.DataFrame, format_name: str) -> dict:
	"""Return the code -> label mapping for one NEISS format."""
	rows = formats[formats["format"] == format_name]
	# Labels look like "1807 - FLOORS OR FLOORING MATERIALS"; keep the name only.
	labels = rows["label"].str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()
	return dict(zip(rows["start"].astype(int), labels))


def decode_age(age: pd.Series, formats: pd.DataFrame) -> pd.Series:
	"""Convert NEISS age codes to years using the AGELTTWO lookup."""
	rows = formats[formats["format"] == "AGELTTWO"]
	months = rows["label"].str.extract(r"^(\d+) MONTH")[0]
	month_codes = dict(zip(rows["start"][months.notna()].astype(int), months.dropna().astype(int)))

	decoded = age.astype(float)
	decoded[age == 0] = np.nan
	in_months = age.isin(list(month_codes))
	decoded[in_months] = (age[in_months].map(month_codes) / 12).round(3)
	return decoded


def add_products(data: pd.DataFrame, formats: pd.DataFrame) -> None:
	labels = code_labels(formats, "PROD")
	codes = data[PRODUCT_COLUMNS].to_numpy()
	unmapped = set(codes[codes != 0].tolist()) - set(labels)
	assert not unmapped, f"Product codes missing from NEISS_FMT: {sorted(unmapped)}"

	product_codes = [[int(code) for code in row if code != 0] for row in codes]
	data["product_codes"] = product_codes
	data["product_names"] = [[labels[code] for code in row] for row in product_codes]
	data["product_count"] = [len(row) for row in product_codes]


def add_incident_text(data: pd.DataFrame) -> None:
	"""Add the narrative with the age/sex prefix and diagnosis section removed."""
	clean_text = data["narrative_clean"]
	without_prefix = clean_text.str.replace(AGE_SEX_PREFIX, "", n=1, regex=True).str.strip()
	data["prefix_removed"] = without_prefix != clean_text
	incident = without_prefix.str.split(DIAGNOSIS_START, n=1, regex=True).str[0].str.strip()
	data["dx_removed"] = incident != without_prefix
	data["incident_text"] = incident
	data["incident_text_empty"] = incident.str.len() < INCIDENT_TEXT_MIN_LENGTH


def add_duplicate_groups(data: pd.DataFrame) -> None:
	data["duplicate_group"] = pd.factorize(data["narrative_key"])[0]
	groups = data.groupby("duplicate_group")
	data["duplicate_group_size"] = groups["duplicate_group"].transform("size")
	# Compare products as sets so the Product_1/2/3 order does not matter.
	product_sets = data["product_codes"].map(lambda codes: tuple(sorted(codes)))
	data["duplicate_labels_differ"] = (
		product_sets.groupby(data["duplicate_group"]).transform("nunique") > 1
	)


def add_split(data: pd.DataFrame) -> None:
	"""Assign train/val/test per duplicate group, not per row."""
	group_count = data["duplicate_group"].max() + 1
	draws = np.random.default_rng(SPLIT_SEED).random(group_count)
	names = np.array(list(SPLIT_BOUNDS))
	group_split = names[np.searchsorted(list(SPLIT_BOUNDS.values()), draws, side="right")]
	data["split"] = group_split[data["duplicate_group"]]


def clean(raw_path: Path) -> tuple:
	"""Return (cleaned data, number of incident rows read)."""
	incidents, formats = read_workbook(raw_path)
	data = incidents.rename(columns={"Narrative_1": "narrative_raw"})

	raw = data["narrative_raw"].astype(str)
	data["narrative_clean"] = (
		raw.str.replace(r"^[\s^]+", "", regex=True)
		.str.replace(r"\s+", " ", regex=True)
		.str.strip()
	)
	data["narrative_key"] = data["narrative_clean"].str.lower()
	data["possible_truncation"] = raw.str.len() == NARRATIVE_FIELD_LIMIT
	data["has_redaction_marker"] = raw.str.contains(REDACTION_MARKER, regex=False)

	add_incident_text(data)
	add_products(data, formats)
	add_duplicate_groups(data)
	add_split(data)

	data["age_decoded"] = decode_age(data["Age"], formats)
	for column, format_name in [("Location", "LOC"), ("Body_Part", "BDYPT"), ("Diagnosis", "DIAG")]:
		data[column.lower() + "_label"] = data[column].map(code_labels(formats, format_name))

	assert len(data) == len(incidents), "Cleaning must not add or drop incidents."
	return data, len(incidents)


def print_summary(data: pd.DataFrame, rows_in: int) -> None:
	repeated = data["duplicate_group_size"] > 1
	print(f"Rows in: {rows_in:,}  rows out: {len(data):,}")
	print(f"possible_truncation: {data['possible_truncation'].sum():,}")
	print(f"has_redaction_marker: {data['has_redaction_marker'].sum():,}")
	print(f"Incidents with two product codes: {(data['product_count'] == 2).sum():,}")
	print(f"Incidents with three product codes: {(data['product_count'] == 3).sum():,}")
	# add_products() asserts that every nonzero code is in the lookup.
	print("Unmapped product codes: 0")
	print(
		f"Records in repeated-text groups: {repeated.sum():,} "
		f"({data.loc[repeated, 'duplicate_group'].nunique():,} groups, "
		f"{data['duplicate_labels_differ'].sum():,} records in groups whose products differ)"
	)
	print(f"Age/sex prefix removed: {data['prefix_removed'].sum():,}")
	print(f"DX section removed: {data['dx_removed'].sum():,}")
	print(f"incident_text_empty: {data['incident_text_empty'].sum():,}")
	print(f"Ages decoded from month codes: {data['Age'].between(201, 223).sum():,}")
	print(f"Split sizes: {data['split'].value_counts().to_dict()}")


def main() -> None:
	data, rows_in = clean(RAW_PATH)

	output = data.copy()
	for column in ["product_codes", "product_names"]:
		output[column] = output[column].map(json.dumps)
	CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
	output.to_csv(CLEAN_PATH, index=False)

	print_summary(data, rows_in)
	print(f"Saved {CLEAN_PATH.name} with {len(output.columns)} columns.")


if __name__ == "__main__":
	main()
