# Lab 4: Data Pipeline and Quality

## 1. Goal

I wanted to build a reproducible pipeline for predicting whether a person's annual income is above $50,000 using the Adult dataset. Each record has 14 input features: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. The output is a binary income label, with >50K as the positive class and <=50K as the negative class.

I used accuracy and F1 to evaluate the predictions. Accuracy describes the overall proportion of correct predictions. F1 combines precision and recall for the >50K class, which matters because the two income classes are not equally common. I compared the original features with the same features plus capital_net and hours_per_age, using Logistic Regression and the same held-out test rows for both experiments.

The main constraints were to preserve the raw data, prevent leakage and make the pipeline reproducible. I needed to split the data before cleaning, learn preprocessing values only from training, and keep every test row. I also needed to report missing values, duplicates and conflicting labels instead of silently removing them. A fixed seed, saved configuration, fitted transformer and processed-data hash make the steps traceable and repeatable. The manual review checks whether sampled records contain impossible values; it cannot verify a person's true income from these records alone.

## 2. Method

I used the existing Python 3.11.8 environment and saved all installed versions in results/versions.txt. I loaded Adult through fetch_openml with name='adult', version=2 and as_frame=True. Ingestion returned 48,842 rows and 15 columns. OpenML calls the target class; I used income in a separate validation view.

I checked the schema, the allowed categorical values and the four ranges in the manual. The ? marker is treated as missing. OpenML already represents the source markers as NaN. I split the raw rows 70/30 with stratification and seed 42 before cleaning, keeping identical feature groups together. A seeded group split is adjusted using singleton groups only to reach exactly the same per-class test counts as a 70/30 stratified split. I then removed 34 exact duplicate rows from training only. No test rows were removed and no test values were used to decide which training rows to delete.

The final split contained 34,155 training rows and 14,653 test rows. I fitted numeric median imputation and categorical mode imputation on training only. I then clipped the six original numeric features using the training 1st and 99th percentiles. I calculated capital_net = capital-gain - capital-loss and hours_per_age = hours-per-week / (age - 16) from these cleaned values. The derived features were not separately clipped.

I standardized numeric features and one-hot encoded categorical features, fitting both on training only. Unknown test categories are ignored by the encoder. Both experiments used LogisticRegression with solver='liblinear', C=1, max_iter=1000 and random_state=42. Both fits completed without a convergence warning. Seeds for Python, NumPy, PyTorch and CUDA were set to 42. The fitted preprocessor for the derived-feature experiment is saved in results/preprocessor.joblib.

The processed dataset is a cleaned table with both derived features, income, source_row and split. It is saved locally under data/lab04/processed.csv, outside Git. Its SHA256 is stored with the resolved configuration in results/config.yaml. The hash covers the CSV bytes before scaling and one-hot encoding. The source config is lab04/config.yaml.

Run everything from the repository root with one command: `.\.venv\Scripts\python.exe lab04\src\run_pipeline.py`.

## 3. Results

| Features | Input columns | Encoded columns | Test accuracy | Test F1 (>50K) |
| --- | --- | --- | --- | --- |
| original_features | 14 | 105 | 0.8543 | 0.6650 |
| with_derived_features | 16 | 107 | 0.8579 | 0.6737 |

Adding the two derived features increased the test F1 by 0.0087. This is one fixed split. capital_net is a linear combination of two existing inputs, so it does not give a linear model entirely new information. hours_per_age introduces a nonlinear relationship. The comparison does not identify which derived feature caused the difference.

| Quality check | Result |
|---|---:|
| Schema, category, range and label violations | 0 |
| Missing cells | 6,465 |
| Raw exact duplicates beyond the first | 52 |
| Conflicting feature groups | 5 |
| Rows in conflicting groups | 10 |
| Raw feature groups shared by training and test | 0 |
| Cleaned feature groups shared by training and test | 4 |
| Manual rows still pending | 0 |

The detailed checks are in [validation_report.md](results/validation_report.md). The conflicting records are in [duplicate_conflicts.csv](results/duplicate_conflicts.csv). The 50-row manual review is in [manual_review.csv](results/manual_review.csv); current status: 50 valid, 0 issue, 0 pending. It remains separate from the automated checks.

Processed dataset SHA256: `df5fb23892d890d325fb5eab85fc568e83b7d9c7d2ba7f25931c74d27221ae80`.

## 4. Conclusions

1. I learned that missing values and invalid values are different. The missing markers need an imputation policy, while an impossible age or unknown category needs to be reported as a validation problem.
2. Splitting first is not enough by itself. Identical feature rows can still appear in both sets, and preprocessing statistics must also be learned only from training. I kept identical feature groups in one partition and removed exact duplicates only from training.
3. Derived features did not automatically guarantee better predictions. Their effect has to be checked on the same test set. The saved configuration and data hash also make it possible to check that another run used the same processed data.

Data source, license, dictionary and limitations: [data_card.md](data_card.md).

Sources: supplied Lab 4 manual; [UCI Adult dataset and dictionary](https://archive.ics.uci.edu/dataset/2/adult); [OpenML dataset 1590 metadata](https://www.openml.org/api/v1/json/data/1590); [scikit-learn 1.5 data-leakage guidance](https://scikit-learn.org/1.5/common_pitfalls.html).
