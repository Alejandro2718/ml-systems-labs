# Adult Data Card

## Source and license

Adult was extracted by Barry Becker from 1994 US Census data and donated to UCI by Becker and Ronny Kohavi. The pipeline loads OpenML Adult version 2, dataset 1590, which combines the original training and test files. It creates its own split as required by the lab.

- [UCI dataset and data dictionary](https://archive.ics.uci.edu/dataset/2/adult)
- [Dataset DOI and attribution](https://doi.org/10.24432/C5XW20): Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository.
- [OpenML version metadata](https://www.openml.org/api/v1/json/data/1590)
- UCI license: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution when sharing derived data. OpenML's metadata labels its license field Public; the UCI license is the specific license used here.

## Size and task

The raw dataset has 48,842 rows, 14 input features and one target. Six inputs are numeric and eight are categorical. There are 11,687 >50K labels (23.93%) and 37,155 <=50K labels. Accuracy and binary F1 are reported, treating >50K as positive.

The split keeps identical feature groups together and preserves the exact class counts of a 70/30 stratified split. The initial split is 34,189/14,653. Training-only duplicate removal leaves 34,155 training rows; every test row is retained. The cleaned table has 48,808 rows and 19 columns: 16 inputs, income, source_row and split. The two model matrices have 105 and 107 columns after encoding.

## Dictionary and missing values

| Feature or target | Meaning | Allowed values or lab range |
| --- | --- | --- |
| age | Age in years | 17-90 |
| workclass | Employment category | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, ? |
| fnlwgt | Census sampling weight | integer; no range specified in the lab |
| education | Education category | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool, ? |
| education-num | Education-level code | integer; no range specified in the lab |
| marital-status | Marital status | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse, ? |
| occupation | Occupation category | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces, ? |
| relationship | Household relationship | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried, ? |
| race | Recorded race category | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black, ? |
| sex | Recorded binary sex category | Female, Male, ? |
| capital-gain | Reported capital gain | 0-99999 |
| capital-loss | Reported capital loss | 0-4356 |
| hours-per-week | Hours worked per week | 1-99 |
| native-country | Country category | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands, ? |
| income | Binary income target | <=50K, >50K, ? |

The token ? denotes missing data, not an additional substantive category. OpenML has converted these markers to NaN. The label must be present. Source row IDs in the review and audit files use zero-based OpenML indexing.

For the personal review, use the categories and lab ranges above and check that numeric fields contain whole numbers. The UCI extraction condition requires fnlwgt > 1; education-num is an education-level code with observed values 1-16. Compare education names with their codes. Missing markers are handled by imputation and are not automatically impossible values.

## Processing and reproducibility

The schema and category lists are stored in config.yaml. Training medians and modes are fitted after the split and training-only duplicate removal. Original numeric values are clipped to training 1st/99th percentiles. capital_net and hours_per_age are calculated after this cleaning. Scaling and one-hot encoding are fitted on training only. No target values are included among the model inputs.

Raw cached data remains under data/openml. The cleaned table is stored at data/lab04/processed.csv. These data files are ignored by Git. The fitted transformer, configuration, hash and validation evidence are under results/. The source configuration is separate from the resolved run configuration.

Processed CSV SHA256: `df5fb23892d890d325fb5eab85fc568e83b7d9c7d2ba7f25931c74d27221ae80`.

The CSV uses UTF-8, LF line endings and %.17g float formatting, sorted by split and source_row. It contains imputed and clipped values before model scaling and encoding. Its hash verifies exact file identity for this configuration and environment; it does not prove that labels are true or that sampling is unbiased.

## Known biases and limitations

This is a historical, selected US sample, not a current or globally representative population. The raw data contains 32,650 Male records (66.85%) and 41,762 White records (85.50%). The higher-income class is also a minority. These imbalances may affect performance across groups; this lab does not measure group-level fairness.

Education and education-num encode overlapping information. fnlwgt is treated as an input because the manual requires all 14 features; metrics are unweighted row-level scores, not population estimates. Label conflict checks found 5 feature groups containing 10 rows with different income labels. These conflicts do not identify which label is wrong. Identical observable attributes also do not prove two records refer to the same person.

The record set includes missing cells and duplicate rows. Clipping removes information from valid extreme values as well as potential outliers. The results describe a single seeded split and two fixed Logistic Regression experiments.

## Privacy concerns

The fields include income, race, sex, age, household information and country of origin. Although names and direct identifiers are not provided, combinations of attributes can be sensitive and may support linkage to other records. Public availability does not guarantee anonymity. The data is used here for a course experiment; no attempt is made to identify individuals.

## Manual review status

0 valid, 0 issue, 50 pending. 50 of 50 sampled records still need personal review. Decisions and notes are stored in results/manual_review.csv. This check concerns impossible values against the dictionary, not verification of actual income against census records.
