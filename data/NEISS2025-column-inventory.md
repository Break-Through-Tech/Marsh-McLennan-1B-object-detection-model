# NEISS 2025 Column Inventory

## Purpose

`data/data_cleaning.py` creates a review table to help the team understand the NEISS workbook before deciding which fields are useful. Despite the script's filename, this first step profiles the data; it does not remove columns, change values, or overwrite the source workbook.

The script currently reads the first worksheet in `data/neiss2025.xlsx`. The profiled copy used for this initial inventory contained 410,201 rows and 25 columns. Rerun the script to refresh the report after replacing or updating the workbook.

## Output

The script writes `data/neiss2025_column_inventory.csv`, with one row for each source column:

| Report field | Meaning |
| --- | --- |
| `column` | Source column name. |
| `inferred_type` | Type inferred by pandas when it reads the worksheet; this is not necessarily the NEISS codebook's conceptual type. |
| `row_count` | Number of records read from the worksheet. This value is repeated for each column. |
| `non_missing_count` | Number of values pandas read as non-missing. |
| `missing_count` | Number of values pandas read as missing. |
| `missing_percent` | Missing values as a percentage of the worksheet row count. |
| `distinct_non_missing_count` | Number of distinct non-missing values in the column. |
| `sample_values` | Up to three distinct example values. Newlines are replaced with spaces and each example is limited to 120 characters. |

Sample values are included to make coded and text fields easier to recognize. They can include short narrative excerpts, so treat the generated report with the same care as the source data and follow project data-sharing rules.

## Source Columns Observed

The initial workbook contained these columns:

`CPSC_Case_Number`, `Treatment_Date`, `Age`, `Sex`, `Race`, `Other_Race`, `Hispanic`, `Body_Part`, `Diagnosis`, `Other_Diagnosis`, `Body_Part_2`, `Diagnosis_2`, `Other_Diagnosis_2`, `Disposition`, `Location`, `Fire_Involvement`, `Product_1`, `Product_2`, `Product_3`, `Alcohol`, `Drug`, `Narrative_1`, `Stratum`, `PSU`, and `Weight`.

Numeric values in coded fields should be interpreted using the NEISS documentation/codebook, rather than assumed to be continuous measurements. For this project's text task, `Narrative_1` is the main narrative field, and the product fields may be candidate labels to investigate. They should not be assumed to identify every object mention in a narrative without validation.

## Interpreting the Inventory

- Use missingness and distinct counts to guide investigation, not as automatic rules for dropping columns.
- Check the NEISS codebook before interpreting numeric category codes.
- Keep identifiers such as `CPSC_Case_Number` out of model inputs, even if they are useful for tracking or deduplication.
- Confirm how `Product_1`, `Product_2`, and `Product_3` relate to the narrative before using them as training targets.
