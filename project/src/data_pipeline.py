import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
RESULTS = PROJECT / 'results'
DATA = ROOT / 'data' / 'project'


def load_config():
    return yaml.safe_load((PROJECT / 'config.yaml').read_text(encoding='utf-8'))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def raw_bytes(config):
    values = config['dataset']
    path = DATA / 'raw' / values['file']
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = urllib.request.urlopen(values['url'], timeout=60).read()
        if sha256(archive) != values['zip_sha256']:
            raise ValueError('Downloaded archive hash does not match the configuration')
        with zipfile.ZipFile(io.BytesIO(archive)) as file:
            path.write_bytes(file.read(values['file']))
    data = path.read_bytes()
    if sha256(data) != values['file_sha256']:
        raise ValueError('SMSSpamCollection hash does not match the configuration')
    return data


def parse(data, config):
    rows, problems = [], []
    for number, line in enumerate(data.decode('utf-8').splitlines()):
        label, separator, text = line.partition('\t')
        if not separator or label not in config['dataset']['labels'] or not text.strip():
            problems.append(dict(source_row=number, line=line[:80]))
            continue
        rows.append(dict(source_row=number, label=label, text=text))
    frame = pd.DataFrame(rows)
    if problems or len(frame) != config['dataset']['rows']:
        raise ValueError(f'Validation failed: {len(problems)} malformed rows, {len(frame)} parsed rows')
    return frame


def deduplicate(frame):
    labels_per_text = frame.groupby('text')['label'].nunique()
    conflicting = set(labels_per_text.index[labels_per_text > 1])
    conflicts = frame.loc[frame.text.isin(conflicting)]
    kept = frame.loc[~frame.text.isin(conflicting)].drop_duplicates('text', keep='first')
    audit = dict(raw_rows=len(frame), exact_duplicate_rows=int(frame.duplicated(['label', 'text']).sum()),
                 duplicate_text_rows=int(frame.duplicated('text').sum()),
                 conflicting_texts=len(conflicting), conflicting_rows=len(conflicts),
                 rows_after_deduplication=len(kept))
    return kept.reset_index(drop=True), conflicts, audit


def split(frame, config):
    seed, values = config['seed'], config['split']
    train, rest = train_test_split(frame, train_size=values['train'], stratify=frame.label, random_state=seed)
    validation_share = values['validation'] / (values['validation'] + values['test'])
    validation, test = train_test_split(rest, train_size=validation_share, stratify=rest.label, random_state=seed)
    return {'train': train.sort_values('source_row'), 'validation': validation.sort_values('source_row'),
            'test': test.sort_values('source_row')}


def run(config):
    data = raw_bytes(config)
    frame = parse(data, config)
    kept, conflicts, audit = deduplicate(frame)
    parts = split(kept, config)
    if set.intersection(*(set(p.text) for p in parts.values())):
        raise AssertionError('The same message text appears in more than one split')
    RESULTS.mkdir(parents=True, exist_ok=True)
    processed = DATA / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    assignment = pd.concat([p[['source_row']].assign(split=name) for name, p in parts.items()]).sort_values('source_row')
    assignment.to_csv(RESULTS / 'split_assignment.csv', index=False, lineterminator='\n')
    conflicts.to_csv(RESULTS / 'label_conflicts.csv', index=False, lineterminator='\n')
    hashes = {}
    for name, part in parts.items():
        path = processed / f'{name}.csv'
        part.to_csv(path, index=False, lineterminator='\n')
        hashes[name] = sha256(path.read_bytes())
    audit.update(file_sha256=sha256(data), split_sha256=hashes,
                 split_rows={n: len(p) for n, p in parts.items()},
                 split_spam={n: int(p.label.eq('spam').sum()) for n, p in parts.items()},
                 spam_rows_raw=int(frame.label.eq('spam').sum()))
    print(f"Data: {audit['raw_rows']} rows, {audit['duplicate_text_rows']} duplicate texts removed, "
          f"{audit['conflicting_texts']} conflicting texts; split {audit['split_rows']}", flush=True)
    return parts, audit


if __name__ == '__main__':
    run(load_config())
