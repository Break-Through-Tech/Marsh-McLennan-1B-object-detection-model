import pandas as pd
from pathlib import Path

# Keep the input and generated report beside this script.
DATA_PATH = Path(__file__).resolve().with_name("neiss2025.xlsx")
INVENTORY_PATH = DATA_PATH.with_name("neiss2025_column_inventory.csv")
SAMPLE_VALUE_LIMIT = 3
SAMPLE_TEXT_LIMIT = 120


def build_column_inventory(data_path: Path) -> pd.DataFrame:
	"""Profile columns in the first worksheet without modifying the source."""
	with pd.ExcelFile(data_path) as workbook:
		# Start with the first sheet; profile other sheets separately if needed.
		sheet_name = workbook.sheet_names[0]
		data = pd.read_excel(workbook, sheet_name=sheet_name)

	row_count = len(data)
	records = []

	for column_name in data.columns:
		values = data[column_name]
		non_missing = values.dropna()
		samples = []

		# Limit examples and text length so the inventory stays easy to scan.
		for value in non_missing.drop_duplicates().head(SAMPLE_VALUE_LIMIT):
			sample = str(value).replace("\n", " ").replace("\r", " ")
			if len(sample) > SAMPLE_TEXT_LIMIT:
				sample = sample[:SAMPLE_TEXT_LIMIT] + "..."
			samples.append(sample)

		missing_count = int(values.isna().sum())
		records.append(
			{
				"column": column_name,
				"inferred_type": str(values.dtype),
				"row_count": row_count,
				"non_missing_count": int(values.notna().sum()),
				"missing_count": missing_count,
				"missing_percent": round(missing_count / row_count * 100, 2)
				if row_count
				else 0.0,
				"distinct_non_missing_count": int(non_missing.nunique()),
				"sample_values": " | ".join(samples),
			}
		)

	return pd.DataFrame(records)


def main() -> None:
	inventory = build_column_inventory(DATA_PATH)
	# Write a separate review file; the source workbook remains unchanged.
	inventory.to_csv(INVENTORY_PATH, index=False)
	print(f"Profiled {len(inventory)} columns from {DATA_PATH.name}.")
	print(f"Inventory saved to {INVENTORY_PATH.name}.")
	print(inventory.to_string(index=False))


if __name__ == "__main__":
	main()