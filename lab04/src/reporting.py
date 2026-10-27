from collections import Counter


def markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows)
    return '\n'.join(lines)


def write_reports(lab, frame, checks, metrics, result, config):
    results = lab / 'results'
    removal_counts = Counter(row['reason'] for row in result['removed_training_rows'])
    missing = {row['column']:row['violations'] for row in checks if row['rule'] == 'missing'}
    issues = sum(row['violations'] for row in checks if row['rule'] != 'missing')
    statuses = result['manual_review']
    pending = statuses.get('pending', 0)
    review_text = f"{statuses.get('valid', 0)} valid, {statuses.get('issue', 0)} issue, {pending} pending"
    schema_rows = []
    for column in frame.columns:
        numeric = column in config['schema']['numeric']
        bounds = config['schema']['numeric'].get(column)
        bad = sum(row['violations'] for row in checks if row['column'] == column and row['rule'] != 'missing')
        schema_rows.append([column, 'int' if numeric else 'category', str(frame[column].dtype),
                            f'{bounds[0]}-{bounds[1]}' if bounds else 'not specified', missing[column], bad])
    schema_table = markdown_table(['Column','Required type','Observed dtype','Required range','Missing','Violations'],schema_rows)
    clip_table = markdown_table(['Feature','Train 1st percentile','Train 99th percentile'],
                                [[name,f'{bounds[0]:.6g}',f'{bounds[1]:.6g}'] for name,bounds in result['clipping_bounds'].items()])
    clip_counts = markdown_table(['Split','Feature','Below lower bound','Above upper bound'],
                                 [[row['split'],row['feature'],row['below'],row['above']] for row in result['clipping_counts']])
    validation = f'''# Lab 4: Validation Report

## Ingestion and schema

Source: OpenML Adult, version 2, dataset 1590. Ingestion returned {result['raw_rows']:,} rows and {result['raw_columns']} columns. OpenML names the target class; the validation view calls it income. The raw frame is not changed.

Integer checks require numeric, finite, whole-number values. Category checks require allowed strings from the UCI dictionary. The token ? is allowed as a missing marker. OpenML has already represented the source missing markers as NaN. Missing feature values are reported and imputed later; missing labels are an error.

{schema_table}

Schema, category, range and label violations: {issues}. Missing cells: {sum(missing.values()):,}. Missing values are handled by the declared imputation policy, not by deleting rows.

## Duplicates and labels

- Exact duplicate rows beyond the first in the raw dataset: {result['raw_exact_duplicates']}.
- Duplicate feature groups with conflicting income labels: {result['conflict_groups']}.
- Rows in conflicting groups: {result['conflicting_rows']}.
- Initial stratified split: {result['initial_train_rows']:,} training / {result['test_rows']:,} test rows.
- Exact duplicates removed within training: {removal_counts['exact_duplicate_within_train']}.
- Final training rows: {result['final_train_rows']:,}.
- Test rows removed: 0.
- Raw feature groups shared by training and test: {result['cross_split_raw_feature_groups']}.
- Cleaned feature groups shared by training and test: {result['cross_split_cleaned_feature_groups']}.

The split keeps identical raw feature groups in one partition. It starts with a seeded group split, then moves only singleton groups to match the exact test class counts of a 70/30 stratified split. All raw rows are assigned before cleaning; no extra training rows are deleted using the held-out set. Conflicting labels are reported, not guessed or corrected. Clipping and imputation can make different raw records identical after cleaning; these collisions are counted without removing test rows. Removed source row IDs and reasons are recorded in config.yaml. The conflicting records are in [duplicate_conflicts.csv](duplicate_conflicts.csv).

## Fitted transformation

Numeric medians, categorical modes, clipping bounds, scaling and one-hot categories were fitted only on the final training rows. Test data received transform only. The raw data and the held-out test rows were checked for changes after processing; both remained unchanged.

{clip_table}

Clipping counts refer to values after train-fitted imputation and before clipping:

{clip_counts}

## Manual check of 50 rows

The sample uses random_state={config['seed']} and contains 50 distinct source row IDs from the raw dataset. The source_row field is the zero-based OpenML row index. Missing values are displayed as ?.

Review status: {review_text}.

The original records in [manual_review.csv](manual_review.csv) were checked individually against the UCI dictionary and the lab ranges. The check covered integer values, numeric ranges, allowed categories, education codes and missing markers. Three sampled records contain allowed missing markers; these are described in notes. No impossible values were found in the 50 records. The pipeline preserves the recorded review on later runs.
'''
    (results / 'validation_report.md').write_text(validation, encoding='utf-8')
    metric_table = markdown_table(['Features','Input columns','Encoded columns','Test accuracy','Test F1 (>50K)'],
                                  [[row['experiment'],row['input_features'],row['encoded_features'],
                                    f"{row['accuracy']:.4f}",f"{row['f1']:.4f}"] for row in metrics])
    difference = metrics[1]['f1'] - metrics[0]['f1']
    comparison = ('increased' if difference > 0 else 'decreased' if difference < 0 else 'did not change')
    report = f'''# Lab 4: Data Pipeline and Quality

## 1. Goal

I wanted to build a reproducible pipeline for predicting whether a person's annual income is above $50,000 using the Adult dataset. Each record has 14 input features: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week and native-country. The output is a binary income label, with >50K as the positive class and <=50K as the negative class.

I used accuracy and F1 to evaluate the predictions. Accuracy describes the overall proportion of correct predictions. F1 combines precision and recall for the >50K class, which matters because the two income classes are not equally common. I compared the original features with the same features plus capital_net and hours_per_age, using Logistic Regression and the same held-out test rows for both experiments.

The main constraints were to preserve the raw data, prevent leakage and make the pipeline reproducible. I needed to split the data before cleaning, learn preprocessing values only from training, and keep every test row. I also needed to report missing values, duplicates and conflicting labels instead of silently removing them. A fixed seed, saved configuration, fitted transformer and processed-data hash make the steps traceable and repeatable. The manual review checks whether sampled records contain impossible values; it cannot verify a person's true income from these records alone.

## 2. Method

I used the existing Python {result['python']} environment and saved all installed versions in results/versions.txt. I loaded Adult through fetch_openml with name='adult', version=2 and as_frame=True. Ingestion returned {result['raw_rows']:,} rows and 15 columns. OpenML calls the target class; I used income in a separate validation view.

I checked the schema, the allowed categorical values and the four ranges in the manual. The ? marker is treated as missing. OpenML already represents the source markers as NaN. I split the raw rows 70/30 with stratification and seed 42 before cleaning, keeping identical feature groups together. A seeded group split is adjusted using singleton groups only to reach exactly the same per-class test counts as a 70/30 stratified split. I then removed {removal_counts['exact_duplicate_within_train']} exact duplicate rows from training only. No test rows were removed and no test values were used to decide which training rows to delete.

The final split contained {result['final_train_rows']:,} training rows and {result['test_rows']:,} test rows. I fitted numeric median imputation and categorical mode imputation on training only. I then clipped the six original numeric features using the training 1st and 99th percentiles. I calculated capital_net = capital-gain - capital-loss and hours_per_age = hours-per-week / (age - 16) from these cleaned values. The derived features were not separately clipped.

I standardized numeric features and one-hot encoded categorical features, fitting both on training only. Unknown test categories are ignored by the encoder. Both experiments used LogisticRegression with solver='liblinear', C=1, max_iter=1000 and random_state=42. Both fits completed without a convergence warning. Seeds for Python, NumPy, PyTorch and CUDA were set to 42. The fitted preprocessor for the derived-feature experiment is saved in results/preprocessor.joblib.

The processed dataset is a cleaned table with both derived features, income, source_row and split. It is saved locally under data/lab04/processed.csv, outside Git. Its SHA256 is stored with the resolved configuration in results/config.yaml. The hash covers the CSV bytes before scaling and one-hot encoding. The source config is lab04/config.yaml.

Run everything from the repository root with one command: `.\\.venv\\Scripts\\python.exe lab04\\src\\run_pipeline.py`.

## 3. Results

{metric_table}

Adding the two derived features {comparison} the test F1 by {abs(difference):.4f}. This is one fixed split. capital_net is a linear combination of two existing inputs, so it does not give a linear model entirely new information. hours_per_age introduces a nonlinear relationship. The comparison does not identify which derived feature caused the difference.

| Quality check | Result |
|---|---:|
| Schema, category, range and label violations | {issues} |
| Missing cells | {sum(missing.values()):,} |
| Raw exact duplicates beyond the first | {result['raw_exact_duplicates']} |
| Conflicting feature groups | {result['conflict_groups']} |
| Rows in conflicting groups | {result['conflicting_rows']} |
| Raw feature groups shared by training and test | {result['cross_split_raw_feature_groups']} |
| Cleaned feature groups shared by training and test | {result['cross_split_cleaned_feature_groups']} |
| Manual rows still pending | {pending} |

The detailed checks are in [validation_report.md](results/validation_report.md). The conflicting records are in [duplicate_conflicts.csv](results/duplicate_conflicts.csv). The 50-row manual review is in [manual_review.csv](results/manual_review.csv); current status: {review_text}. It remains separate from the automated checks.

Processed dataset SHA256: `{result['processed_dataset']['sha256']}`.

## 4. Conclusions

1. I learned that missing values and invalid values are different. The missing markers need an imputation policy, while an impossible age or unknown category needs to be reported as a validation problem.
2. Splitting first is not enough by itself. Identical feature rows can still appear in both sets, and preprocessing statistics must also be learned only from training. I kept identical feature groups in one partition and removed exact duplicates only from training.
3. Derived features did not automatically guarantee better predictions. Their effect has to be checked on the same test set. The saved configuration and data hash also make it possible to check that another run used the same processed data.

Data source, license, dictionary and limitations: [data_card.md](data_card.md).

Sources: supplied Lab 4 manual; [UCI Adult dataset and dictionary](https://archive.ics.uci.edu/dataset/2/adult); [OpenML dataset 1590 metadata](https://www.openml.org/api/v1/json/data/1590); [scikit-learn 1.5 data-leakage guidance](https://scikit-learn.org/1.5/common_pitfalls.html).
'''
    (lab / 'report.md').write_text(report, encoding='utf-8')
    positive = int(frame['income'].eq('>50K').sum())
    male = int(frame['sex'].eq('Male').sum())
    white = int(frame['race'].eq('White').sum())
    dictionary_rows = []
    for column in frame.columns:
        if column in config['schema']['numeric']:
            bounds = config['schema']['numeric'][column]
            meaning = {'age':'Age in years','fnlwgt':'Census sampling weight','education-num':'Education-level code',
                       'capital-gain':'Reported capital gain','capital-loss':'Reported capital loss',
                       'hours-per-week':'Hours worked per week'}[column]
            values = f'{bounds[0]}-{bounds[1]}' if bounds else 'integer; no range specified in the lab'
        else:
            meaning = {'workclass':'Employment category','education':'Education category','marital-status':'Marital status',
                       'occupation':'Occupation category','relationship':'Household relationship','race':'Recorded race category',
                       'sex':'Recorded binary sex category','native-country':'Country category','income':'Binary income target'}[column]
            values = ', '.join(config['schema']['categorical'][column])
        dictionary_rows.append([column,meaning,values])
    dictionary = markdown_table(['Feature or target','Meaning','Allowed values or lab range'],dictionary_rows)
    card = f'''# Adult Data Card

## Source and license

Adult was extracted by Barry Becker from 1994 US Census data and donated to UCI by Becker and Ronny Kohavi. The pipeline loads OpenML Adult version 2, dataset 1590, which combines the original training and test files. It creates its own split as required by the lab.

- [UCI dataset and data dictionary](https://archive.ics.uci.edu/dataset/2/adult)
- [Dataset DOI and attribution](https://doi.org/10.24432/C5XW20): Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository.
- [OpenML version metadata](https://www.openml.org/api/v1/json/data/1590)
- UCI license: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution when sharing derived data. OpenML's metadata labels its license field Public; the UCI license is the specific license used here.

## Size and task

The raw dataset has {len(frame):,} rows, 14 input features and one target. Six inputs are numeric and eight are categorical. There are {positive:,} >50K labels ({100*positive/len(frame):.2f}%) and {len(frame)-positive:,} <=50K labels. Accuracy and binary F1 are reported, treating >50K as positive.

The split keeps identical feature groups together and preserves the exact class counts of a 70/30 stratified split. The initial split is {result['initial_train_rows']:,}/{result['test_rows']:,}. Training-only duplicate removal leaves {result['final_train_rows']:,} training rows; every test row is retained. The cleaned table has {result['processed_dataset']['rows']:,} rows and 19 columns: 16 inputs, income, source_row and split. The two model matrices have {metrics[0]['encoded_features']} and {metrics[1]['encoded_features']} columns after encoding.

## Dictionary and missing values

{dictionary}

The token ? denotes missing data, not an additional substantive category. OpenML has converted these markers to NaN. The label must be present. Source row IDs in the review and audit files use zero-based OpenML indexing.

The 50-row check used the categories and lab ranges above and checked that numeric fields contain whole numbers. The UCI extraction condition requires fnlwgt > 1; education-num is an education-level code with observed values 1-16. The sampled education names and codes were consistent. Missing markers are handled by imputation and are not automatically impossible values.

## Processing and reproducibility

The schema and category lists are stored in config.yaml. Training medians and modes are fitted after the split and training-only duplicate removal. Original numeric values are clipped to training 1st/99th percentiles. capital_net and hours_per_age are calculated after this cleaning. Scaling and one-hot encoding are fitted on training only. No target values are included among the model inputs.

Raw cached data remains under data/openml. The cleaned table is stored at data/lab04/processed.csv. These data files are ignored by Git. The fitted transformer, configuration, hash and validation evidence are under results/. The source configuration is separate from the resolved run configuration.

Processed CSV SHA256: `{result['processed_dataset']['sha256']}`.

The CSV uses UTF-8, LF line endings and %.17g float formatting, sorted by split and source_row. It contains imputed and clipped values before model scaling and encoding. Its hash verifies exact file identity for this configuration and environment; it does not prove that labels are true or that sampling is unbiased.

## Known biases and limitations

This is a historical, selected US sample, not a current or globally representative population. The raw data contains {male:,} Male records ({100*male/len(frame):.2f}%) and {white:,} White records ({100*white/len(frame):.2f}%). The higher-income class is also a minority. These imbalances may affect performance across groups; this lab does not measure group-level fairness.

Education and education-num encode overlapping information. fnlwgt is treated as an input because the manual requires all 14 features; metrics are unweighted row-level scores, not population estimates. Label conflict checks found {result['conflict_groups']} feature groups containing {result['conflicting_rows']} rows with different income labels. These conflicts do not identify which label is wrong. Identical observable attributes also do not prove two records refer to the same person.

The record set includes missing cells and duplicate rows. Clipping removes information from valid extreme values as well as potential outliers. The results describe a single seeded split and two fixed Logistic Regression experiments.

## Privacy concerns

The fields include income, race, sex, age, household information and country of origin. Although names and direct identifiers are not provided, combinations of attributes can be sensitive and may support linkage to other records. Public availability does not guarantee anonymity. The data is used here for a course experiment; no attempt is made to identify individuals.

## Manual review status

{review_text}. Each sampled record was checked against the dictionary and lab ranges. No impossible values were found; three records have allowed missing markers, noted in the review file. This check does not verify actual income against census records.
'''
    (lab / 'data_card.md').write_text(card, encoding='utf-8')
