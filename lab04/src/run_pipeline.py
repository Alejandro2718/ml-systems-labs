import hashlib
import io
import platform
import random
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from preprocessing import AdultPreprocessor
from reporting import write_reports
from validation import audit_conflicts, split_and_deduplicate, validate

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / 'results'
DATA = LAB.parent / 'data'


def frame_hash(frame):
    digest = hashlib.sha256()
    digest.update('|'.join(frame.columns).encode())
    digest.update('|'.join(map(str, frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.astype('<u8').tobytes())
    return digest.hexdigest()


def prepare_manual_review(frame, config):
    sample = frame.sample(n=config['manual_sample_size'], random_state=config['seed']).copy()
    sample.insert(0, 'source_row', sample.index)
    path = RESULTS / 'manual_review.csv'
    raw_text = sample.to_csv(index=False, na_rep='?', lineterminator='\n')
    expected = pd.read_csv(io.StringIO(raw_text), dtype=str, keep_default_na=False)
    if path.exists():
        review = pd.read_csv(path, dtype=str, keep_default_na=False)
        if not review[expected.columns].equals(expected):
            raise ValueError('The manual review sample was changed; original values must be preserved')
    else:
        review = expected.copy()
        review['review_status'] = 'pending'
        review['notes'] = ''
        review.to_csv(path, index=False, lineterminator='\n')
    if not review['review_status'].isin(['pending', 'valid', 'issue']).all():
        raise ValueError('Manual review statuses must be pending, valid or issue')
    if (review['review_status'].eq('issue') & review['notes'].str.strip().eq('')).any():
        raise ValueError('Explain each manual issue in the notes column')
    return review['review_status'].value_counts().to_dict()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((LAB / 'config.yaml').read_text(encoding='utf-8-sig'))
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('1. Reading Adult version 2 from OpenML', flush=True)
    adult = fetch_openml(name=config['dataset']['name'], version=config['dataset']['version'],
                         as_frame=True, data_home=DATA / 'openml')
    if int(adult.details['id']) != config['dataset']['openml_id']:
        raise ValueError('Unexpected OpenML dataset ID')
    if config['preprocessing']['derived_features'] != ['capital_net', 'hours_per_age']:
        raise ValueError('The lab requires capital_net and hours_per_age')
    raw = adult.frame
    if raw.shape != (config['dataset']['rows'], config['dataset']['columns']):
        raise ValueError(f'Unexpected input shape: {raw.shape}')
    original_hash = frame_hash(raw)
    print(f'Raw rows: {len(raw)}; raw columns: {len(raw.columns)}', flush=True)
    frame = raw.rename(columns={config['dataset']['target']: config['dataset']['schema_target']})
    checks, violations = validate(frame, config['schema'])
    if violations:
        pd.DataFrame(violations).to_csv(RESULTS / 'validation_violations.csv', index=False)
        raise ValueError('Schema or range violations found; see validation_violations.csv')
    print('2. Schema and range validation passed', flush=True)
    conflict_count, conflicts = audit_conflicts(frame)
    conflicts.to_csv(RESULTS / 'duplicate_conflicts.csv', index=False, na_rep='?', lineterminator='\n')
    manual_counts = prepare_manual_review(frame, config)
    train, test, initial_train, removed = split_and_deduplicate(frame, config)
    untouched_test = frame.loc[test.index].copy(deep=True)
    print(f'3. Split: {initial_train} training / {len(test)} test; {len(removed)} training rows removed', flush=True)
    numeric = list(config['schema']['numeric'])
    categorical = [name for name in config['schema']['categorical'] if name != 'income']
    X_train, X_test = train.drop(columns='income'), test.drop(columns='income')
    y_train = train['income'].eq(config['dataset']['positive_label']).astype(np.int64)
    y_test = test['income'].eq(config['dataset']['positive_label']).astype(np.int64)
    preprocessors, metrics = {}, []
    with threadpool_limits(limits=1):
        for name, derived in [('original_features', False), ('with_derived_features', True)]:
            preprocessor = AdultPreprocessor(numeric, categorical,
                                              quantiles=tuple(config['preprocessing']['clip_quantiles']),
                                              derived=derived)
            train_values = preprocessor.fit_transform(X_train)
            test_values = preprocessor.transform(X_test)
            model = LogisticRegression(**config['model'], random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(train_values, y_train)
            predicted = model.predict(test_values)
            metrics.append(dict(experiment=name, input_features=14 + 2 * int(derived),
                                encoded_features=train_values.shape[1],
                                accuracy=accuracy_score(y_test, predicted),
                                f1=f1_score(y_test, predicted, pos_label=1, zero_division=0)))
            preprocessors[name] = preprocessor
            print(f"4. {name}: accuracy={metrics[-1]['accuracy']:.4f}, F1={metrics[-1]['f1']:.4f}", flush=True)
    pd.DataFrame(metrics).to_csv(RESULTS / 'model_comparison.csv', index=False, lineterminator='\n')
    joblib.dump(preprocessors['with_derived_features'], RESULTS / 'preprocessor.joblib', compress=3)
    loaded = joblib.load(RESULTS / 'preprocessor.joblib')
    if (loaded.transform(X_test) != preprocessors['with_derived_features'].transform(X_test)).nnz:
        raise AssertionError('Reloaded transformer changed the processed data')
    processed_parts = []
    preprocessor = preprocessors['with_derived_features']
    for split_name, part in [('train', train), ('test', test)]:
        cleaned = preprocessor.clean_frame(part.drop(columns='income'))
        cleaned['income'] = part['income'].astype(object)
        cleaned.insert(0, 'source_row', part.index)
        cleaned.insert(1, 'split', split_name)
        processed_parts.append(cleaned)
    processed = pd.concat(processed_parts).sort_values(['split', 'source_row'], kind='stable')
    processed_path = DATA / 'lab04' / 'processed.csv'
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False, float_format='%.17g', lineterminator='\n')
    processed_hash = hashlib.sha256(processed_path.read_bytes()).hexdigest()
    if not test.equals(untouched_test) or frame_hash(raw) != original_hash:
        raise AssertionError('Raw data or the held-out test rows were modified')
    fitted = preprocessors['with_derived_features']
    clipping = []
    for split_name, values in [('train', X_train), ('test', X_test)]:
        imputed = fitted.numeric_imputer_.transform(fitted._numeric_values(values))
        for i, column in enumerate(numeric):
            clipping.append(dict(split=split_name, feature=column,
                                 below=int((imputed[:, i] < fitted.lower_[i]).sum()),
                                 above=int((imputed[:, i] > fitted.upper_[i]).sum())))
    feature_columns = list(X_train.columns)
    raw_partitions = pd.concat([train.assign(split='train'), test.assign(split='test')])
    raw_overlap = int(raw_partitions.groupby(feature_columns, dropna=False, observed=True)
                      ['split'].nunique().gt(1).sum())
    cleaned_columns = feature_columns + config['preprocessing']['derived_features']
    cleaned_overlap = int(processed.groupby(cleaned_columns, dropna=False, observed=True)
                          ['split'].nunique().gt(1).sum())
    result = dict(raw_rows=len(raw), raw_columns=len(raw.columns), raw_frame_sha256=original_hash,
                  openml_id=int(adult.details['id']), openml_md5=adult.details['md5_checksum'],
                  initial_train_rows=initial_train, final_train_rows=len(train), test_rows=len(test),
                  raw_exact_duplicates=int(frame.duplicated().sum()),
                  conflict_groups=conflict_count, conflicting_rows=len(conflicts),
                  removed_training_rows=removed, manual_review=manual_counts,
                  processed_dataset=dict(path='data/lab04/processed.csv', rows=len(processed),
                                         columns=list(processed.columns), sha256=processed_hash,
                                         format='CSV UTF-8, LF, index excluded, floats %.17g, split/source_row order'),
                  clipping_bounds={column:[float(fitted.lower_[i]), float(fitted.upper_[i])]
                                   for i,column in enumerate(numeric)},
                  clipping_counts=clipping,
                  categorical_modes=dict(zip(categorical, map(str, fitted.categorical_imputer_.statistics_))),
                  numeric_medians=dict(zip(numeric, map(float, fitted.numeric_imputer_.statistics_))),
                  preprocessing_settings=dict(numeric_imputation=fitted.numeric_imputer_.strategy,
                                              categorical_imputation=fitted.categorical_imputer_.strategy,
                                              quantile_method='linear', scale_numeric=fitted.scaler_.with_std,
                                              unknown_categories=fitted.encoder_.handle_unknown,
                                              saved_transformer='with_derived_features'),
                  package_versions={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib','PyYAML']},
                  python=platform.python_version(), raw_unchanged=True, test_rows_unchanged=True,
                  cross_split_raw_feature_groups=raw_overlap,
                  cross_split_cleaned_feature_groups=cleaned_overlap)
    resolved = dict(config)
    resolved['run'] = result
    (RESULTS / 'config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False, width=100), encoding='utf-8')
    write_reports(LAB, frame, checks, metrics, result, config)
    from print_versions import main as print_versions
    print_versions()
    print(f'5. Processed dataset SHA256: {processed_hash}', flush=True)
    print(f'6. Manual review: {manual_counts}', flush=True)
    print('Lab 4 results and reports saved.', flush=True)


if __name__ == '__main__':
    main()
