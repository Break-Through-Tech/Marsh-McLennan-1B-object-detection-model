import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_GOLD = DATA_DIR / "gold" / "gold_sample.csv"
DEFAULT_PREDICTIONS = DATA_DIR / "processed" / "predictions_spacy.csv"


def parse_objects(cell) -> set:
	"""Turn "Ladder; car seats" into {"ladder", "car seat"}."""
	if pd.isna(cell):
		return set()
	objects = set()
	for item in str(cell).lower().split(";"):
		item = " ".join(item.split())
		# Crude singular form so "stairs" and "stair" count as a match.
		if item.endswith("s") and not item.endswith("ss"):
			item = item[:-1]
		# Labelers write NONE when a narrative has no object.
		if item and item != "none":
			objects.add(item)
	return objects


def evaluate(gold: pd.DataFrame, predictions: pd.DataFrame) -> tuple:
	"""Score predictions on the labeled rows; return (metrics, per-row results)."""
	labeled = gold[gold["labeler"].notna() & (gold["labeler"] != "")]
	merged = labeled.merge(predictions, on="case_id", how="left")

	true_positives = false_positives = false_negatives = exact_matches = 0
	partial_matches = 0
	results = []
	for row in merged.itertuples():
		gold_objects = parse_objects(row.gold_objects)
		predicted_objects = parse_objects(row.objects)
		partial_match_pass = bool(gold_objects & predicted_objects) if gold_objects else not predicted_objects
		partial_matches += partial_match_pass
		true_positives += len(gold_objects & predicted_objects)
		false_positives += len(predicted_objects - gold_objects)
		false_negatives += len(gold_objects - predicted_objects)
		if gold_objects == predicted_objects:
			exact_matches += 1
		results.append(
			{
				"case_id": row.case_id,
				"incident_text": row.incident_text,
				"gold_objects": row.gold_objects,
				"objects": row.objects,
				"wrongly_predicted": "; ".join(sorted(predicted_objects - gold_objects)),
				"missed": "; ".join(sorted(gold_objects - predicted_objects)),
				"exact_match_pass": gold_objects == predicted_objects,
				"partial_match_pass": partial_match_pass,
			}
		)

	predicted_total = true_positives + false_positives
	gold_total = true_positives + false_negatives
	precision = true_positives / predicted_total if predicted_total else 0.0
	recall = true_positives / gold_total if gold_total else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
	metrics = {
		"labeled_narratives": len(merged),
		"precision": round(precision, 3),
		"recall": round(recall, 3),
		"micro_f1": round(f1, 3),
		"exact_match_accuracy": round(exact_matches / len(merged), 3) if len(merged) else 0.0,
		"at_least_one_match_accuracy": round(partial_matches / len(merged), 3) if len(merged) else 0.0,
	}
	return metrics, pd.DataFrame(results)


def main() -> None:
	parser = argparse.ArgumentParser(description="Score predicted objects against the gold set.")
	parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
	parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
	args = parser.parse_args()

	metrics, results = evaluate(pd.read_csv(args.gold), pd.read_csv(args.predictions))
	if not metrics["labeled_narratives"]:
		raise SystemExit("No labeled rows yet: fill in gold_objects and labeler in the gold file.")
	for name, value in metrics.items():
		print(f"{name}: {value}")

	results_path = args.predictions.with_name(args.predictions.stem + "_results.csv")
	results.to_csv(results_path, index=False)
	print(f"Saved {len(results)} narrative results to {results_path.name}.")

	errors = results[~results["exact_match_pass"]]
	errors_path = args.predictions.with_name(args.predictions.stem + "_errors.csv")
	errors.to_csv(errors_path, index=False)
	print(f"Saved {len(errors)} narratives with errors to {errors_path.name}.")


if __name__ == "__main__":
	main()
