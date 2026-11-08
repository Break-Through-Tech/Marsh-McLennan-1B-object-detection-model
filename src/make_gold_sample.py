import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CLEAN_PATH = DATA_DIR / "processed" / "neiss_2025_clean.csv"
GOLD_PATH = DATA_DIR / "gold" / "gold_sample.csv"
SAMPLE_SIZE = 200
RANDOM_SEED = 42


def main() -> None:
	parser = argparse.ArgumentParser(description="Sample narratives for hand labeling.")
	parser.add_argument("--output", type=Path, default=GOLD_PATH)
	parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
	parser.add_argument("--seed", type=int, default=RANDOM_SEED)
	parser.add_argument(
		"--exclude",
		type=Path,
		nargs="*",
		default=[],
		help="Sample files whose case_ids must not be drawn again.",
	)
	args = parser.parse_args()

	# Refuse to overwrite: this file holds the team's hand labels.
	if args.output.exists():
		raise SystemExit(f"{args.output} already exists; delete it to resample.")

	data = pd.read_csv(
		CLEAN_PATH,
		usecols=[
			"CPSC_Case_Number",
			"incident_text",
			"incident_text_empty",
			"has_redaction_marker",
		],
	)
	# Redacted ("***") narratives often hide the object itself, so skip them.
	usable = data[~data["incident_text_empty"] & ~data["has_redaction_marker"]]
	for path in args.exclude:
		used = pd.read_csv(path, usecols=["case_id"])["case_id"]
		usable = usable[~usable["CPSC_Case_Number"].isin(used)]
	sample = usable.sample(args.size, random_state=args.seed)

	gold = pd.DataFrame(
		{
			"case_id": sample["CPSC_Case_Number"],
			"incident_text": sample["incident_text"],
			"gold_objects": "",
			"labeler": "",
		}
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	gold.to_csv(args.output, index=False)
	print(f"Saved {len(gold)} narratives to label in {args.output.name}.")


if __name__ == "__main__":
	main()
