import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def validate(frame, schema):
    checks, violations = [], []
    expected = set(schema['numeric']) | set(schema['categorical'])
    if set(frame.columns) != expected:
        raise ValueError('Columns do not match the required schema')

    def record(column, rule, bad):
        bad = pd.Series(bad, index=frame.index).fillna(False)
        checks.append(dict(column=column, rule=rule, violations=int(bad.sum())))
        for row_id in frame.index[bad]:
            violations.append(dict(source_row=int(row_id), column=column,
                                   rule=rule, value=str(frame.at[row_id, column])))

    for column, bounds in schema['numeric'].items():
        series = frame[column]
        missing = series.isna() | series.eq('?')
        values = pd.to_numeric(series.mask(missing), errors='coerce')
        invalid = ~missing & (~np.isfinite(values) | values.ne(np.floor(values)))
        record(column, 'integer_type', invalid)
        if bounds:
            record(column, f'range_{bounds[0]}_{bounds[1]}', ~missing & ~values.between(*bounds))
        checks.append(dict(column=column, rule='missing', violations=int(missing.sum())))
    for column, allowed in schema['categorical'].items():
        series = frame[column].astype(object)
        missing = series.isna() | series.eq('?')
        record(column, 'category_type', ~missing & ~series.map(lambda value: isinstance(value, str)))
        record(column, 'allowed_category', ~missing & ~series.isin(allowed))
        checks.append(dict(column=column, rule='missing', violations=int(missing.sum())))
    record('income', 'required_label', frame['income'].isna() | frame['income'].eq('?'))
    return checks, violations


def feature_groups(frame):
    features = frame.drop(columns='income')
    keys = features.astype(object).where(features.notna(), '<MISSING>')
    return keys.groupby(list(keys.columns), dropna=False, sort=True).ngroup()


def audit_conflicts(frame):
    groups = feature_groups(frame)
    labels_per_group = frame['income'].groupby(groups).nunique()
    conflicting = labels_per_group.index[labels_per_group > 1]
    rows = frame.loc[groups.isin(conflicting)].copy()
    rows.insert(0, 'conflict_group', groups.loc[rows.index].to_numpy())
    rows.insert(0, 'source_row', rows.index)
    return len(conflicting), rows


def split_and_deduplicate(frame, config):
    if config['split']['strategy'] != 'stratified_feature_groups':
        raise ValueError('The split must preserve class proportions and keep feature groups separate')
    if config['split']['duplicates'] != 'exact_train_only':
        raise ValueError('Only exact training duplicates may be removed')
    seed = config['seed']
    test_size = config['split']['test_size']
    groups = feature_groups(frame).to_numpy()
    _, quota_rows = train_test_split(frame.index, test_size=test_size,
                                      stratify=frame['income'], random_state=seed)
    quotas = frame.loc[quota_rows, 'income'].value_counts()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_positions = next(splitter.split(frame, groups=groups))
    in_test = np.zeros(len(frame), dtype=bool)
    in_test[test_positions] = True
    singletons = ~frame.drop(columns='income').duplicated(keep=False).to_numpy()
    rng = np.random.default_rng(seed)
    for label, count in quotas.items():
        belongs_to_class = frame['income'].eq(label).to_numpy()
        difference = int(count) - int((in_test & belongs_to_class).sum())
        if difference:
            candidates = np.flatnonzero(singletons & belongs_to_class &
                                         (~in_test if difference > 0 else in_test))
            selected = rng.choice(candidates, size=abs(difference), replace=False)
            in_test[selected] = difference > 0
    train = frame.iloc[rng.permutation(np.flatnonzero(~in_test))].copy()
    test = frame.iloc[rng.permutation(np.flatnonzero(in_test))].copy()
    if test['income'].value_counts().to_dict() != quotas.to_dict():
        raise AssertionError('The test set does not meet the stratified class counts')
    if len(train) + len(test) != len(frame) or set(train.index) & set(test.index):
        raise AssertionError('Every raw row must belong to exactly one partition')
    if np.intersect1d(groups[in_test], groups[~in_test]).size:
        raise AssertionError('A feature group occurs in both partitions')
    initial_train = len(train)
    exact = train.duplicated(keep='first')
    removed = [dict(source_row=int(i), reason='exact_duplicate_within_train') for i in train.index[exact]]
    train = train.loc[~exact].copy()
    return train, test, initial_train, removed
