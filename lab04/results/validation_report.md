# Lab 4: Validation Report

## Ingestion and schema

Source: OpenML Adult, version 2, dataset 1590. Ingestion returned 48,842 rows and 15 columns. OpenML names the target class; the validation view calls it income. The raw frame is not changed.

Integer checks require numeric, finite, whole-number values. Category checks require allowed strings from the UCI dictionary. The token ? is allowed as a missing marker. OpenML has already represented the source missing markers as NaN. Missing feature values are reported and imputed later; missing labels are an error.

| Column | Required type | Observed dtype | Required range | Missing | Violations |
| --- | --- | --- | --- | --- | --- |
| age | int | int64 | 17-90 | 0 | 0 |
| workclass | category | category | not specified | 2799 | 0 |
| fnlwgt | int | int64 | not specified | 0 | 0 |
| education | category | category | not specified | 0 | 0 |
| education-num | int | int64 | not specified | 0 | 0 |
| marital-status | category | category | not specified | 0 | 0 |
| occupation | category | category | not specified | 2809 | 0 |
| relationship | category | category | not specified | 0 | 0 |
| race | category | category | not specified | 0 | 0 |
| sex | category | category | not specified | 0 | 0 |
| capital-gain | int | int64 | 0-99999 | 0 | 0 |
| capital-loss | int | int64 | 0-4356 | 0 | 0 |
| hours-per-week | int | int64 | 1-99 | 0 | 0 |
| native-country | category | category | not specified | 857 | 0 |
| income | category | category | not specified | 0 | 0 |

Schema, category, range and label violations: 0. Missing cells: 6,465. Missing values are handled by the declared imputation policy, not by deleting rows.

## Duplicates and labels

- Exact duplicate rows beyond the first in the raw dataset: 52.
- Duplicate feature groups with conflicting income labels: 5.
- Rows in conflicting groups: 10.
- Initial stratified split: 34,189 training / 14,653 test rows.
- Exact duplicates removed within training: 34.
- Final training rows: 34,155.
- Test rows removed: 0.
- Raw feature groups shared by training and test: 0.
- Cleaned feature groups shared by training and test: 4.

The split keeps identical raw feature groups in one partition. It starts with a seeded group split, then moves only singleton groups to match the exact test class counts of a 70/30 stratified split. All raw rows are assigned before cleaning; no extra training rows are deleted using the held-out set. Conflicting labels are reported, not guessed or corrected. Clipping and imputation can make different raw records identical after cleaning; these collisions are counted without removing test rows. Removed source row IDs and reasons are recorded in config.yaml. The conflicting records are in [duplicate_conflicts.csv](duplicate_conflicts.csv).

## Fitted transformation

Numeric medians, categorical modes, clipping bounds, scaling and one-hot categories were fitted only on the final training rows. Test data received transform only. The raw data and the held-out test rows were checked for changes after processing; both remained unchanged.

| Feature | Train 1st percentile | Train 99th percentile |
| --- | --- | --- |
| age | 17 | 74 |
| fnlwgt | 27319.6 | 509738 |
| education-num | 3 | 16 |
| capital-gain | 0 | 15024 |
| capital-loss | 0 | 1980 |
| hours-per-week | 8 | 80 |

Clipping counts refer to values after train-fitted imputation and before clipping:

| Split | Feature | Below lower bound | Above upper bound |
| --- | --- | --- | --- |
| train | age | 0 | 304 |
| train | fnlwgt | 342 | 342 |
| train | education-num | 227 | 0 |
| train | capital-gain | 0 | 264 |
| train | capital-loss | 0 | 327 |
| train | hours-per-week | 330 | 216 |
| test | age | 0 | 141 |
| test | fnlwgt | 166 | 144 |
| test | education-num | 99 | 0 |
| test | capital-gain | 0 | 127 |
| test | capital-loss | 0 | 164 |
| test | hours-per-week | 125 | 102 |

## Manual check of 50 rows

The sample uses random_state=42 and contains 50 distinct source row IDs from the raw dataset. The source_row field is the zero-based OpenML row index. Missing values are displayed as ?.

Review status: 50 valid, 0 issue, 0 pending.

The original records in [manual_review.csv](manual_review.csv) were checked individually against the UCI dictionary and the lab ranges. The check covered integer values, numeric ranges, allowed categories, education codes and missing markers. Three sampled records contain allowed missing markers; these are described in notes. No impossible values were found in the 50 records. The pipeline preserves the recorded review on later runs.
