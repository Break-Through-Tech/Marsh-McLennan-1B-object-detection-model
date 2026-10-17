import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_SOURCE = DATA_DIR / "gold" / "gold_sample_final.csv"
DEFAULT_OUTPUT = DATA_DIR / "gold" / "gold_sample.csv"

# The team's labeling sheet has been exported with either header style.
COLUMN_NAMES = {
	"injury description": "incident_text",
	"case id": "case_id",
	"key object terms": "gold_objects",
	"team label": "labeler",
	"incident_text": "incident_text",
	"case_id": "case_id",
	"gold_objects": "gold_objects",
	"labeler": "labeler",
}


def normalize_objects(cell: str) -> str:
	"""Lowercase and trim each object; blank or NONE becomes NONE."""
	objects = [item.strip().lower() for item in str(cell).split(";")]
	objects = [item for item in objects if item and item != "none"]
	return "; ".join(objects) if objects else "NONE"


def convert(source: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
	source = source.rename(columns=lambda name: COLUMN_NAMES.get(name.strip().lower(), name))
	# Repeated header rows and blank lines have no numeric case ID.
	case_ids = pd.to_numeric(source["case_id"].str.strip(), errors="coerce")
	source = source[case_ids.notna()].assign(case_id=case_ids.dropna().astype("int64"))

	gold = pd.DataFrame(
		{
			"case_id": source["case_id"],
			"incident_text": source["incident_text"],
			"gold_objects": source["gold_objects"].fillna("").map(normalize_objects),
			"labeler": "team",
		}
	)
	# Spreadsheet exports can alter quotes and escape characters; restore the
	# exact text from the sampled file where the case is known.
	original_text = original.set_index("case_id")["incident_text"]
	gold["incident_text"] = gold["case_id"].map(original_text).fillna(gold["incident_text"])
	return gold.reset_index(drop=True)


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert the team's labeled sheet to the scorer's gold file.")
	parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
	parser.add_argument("--original", type=Path, default=DEFAULT_OUTPUT, help="Unlabeled sample file with the exact text.")
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	args = parser.parse_args()

	source = pd.read_csv(args.source, dtype=str, keep_default_na=False)
	original = pd.read_csv(args.original, usecols=["case_id", "incident_text"])
	gold = convert(source, original)
	assert gold["case_id"].is_unique, "Duplicate case IDs in the labeled file."

	gold.to_csv(args.output, index=False)
	print(f"Saved {len(gold)} labeled rows to {args.output.name}.")


if __name__ == "__main__":
	main()
